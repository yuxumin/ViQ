import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector, NormalizedDwPooler
import os
import math
import torch.nn.functional as F

if 'DINO2_SEP_MLP' in os.environ:
    print("DINO2_SEP_MLP is set")
    DINO2_SEP_MLP = True
else:
    DINO2_SEP_MLP = False


if 'VISION2_DIM' in os.environ:
    VISION2_DIM = os.environ['VISION2_DIM']
    print(f"VISION2_DIM is set as {VISION2_DIM}")
    VISION2_DIM = int(VISION2_DIM)
else:
    VISION2_DIM = 1024

if 'REGIONAL_POOL' in os.environ:
    REGIONAL_POOL = os.environ['REGIONAL_POOL']
else:
    REGIONAL_POOL = '2x'
print(f"REGIONAL_POOL is set as {REGIONAL_POOL}")

if 'ADAPTIVE_TWO_VIEWS' in os.environ:
    print("ADAPTIVE_TWO_VIEWS is set")
    ADAPTIVE_TWO_VIEWS = True
else:
    ADAPTIVE_TWO_VIEWS = False


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class SepProjConv(nn.Module):
    def __init__(self, in_channels, out_channels, extra_channels=1024):
        super().__init__()
        base_channels = in_channels - extra_channels
        in_channels = base_channels * 2

        self.base_channels = base_channels
        self.extra_channels = extra_channels

        self.proj_extra = nn.Sequential(
            nn.Conv2d(1024, base_channels, 1)
        )

        self.proj_base = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1)
        )

        self.proj = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.split([self.base_channels, self.extra_channels], dim=1)
        x1 = self.proj_base(x1)
        x2 = self.proj_extra(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x)
        return x

class MultiPathConvMlp(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, projector_type):
        super().__init__()
        self.projector_type = projector_type
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        if DINO2_SEP_MLP:
            self.vision_projector1 = SepProjConv(in_channels1, out_channels, 1024)
        else:
            self.vision_projector1 = nn.Sequential(
                nn.Conv2d(in_channels1, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                #downsample conv
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            )
        if 'woconv' in projector_type:
            print("Building multipath conv mlp without conv...")
            self.vision_projector2 = nn.Sequential(
                nn.Linear(in_channels2, out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.vision_projector2 = nn.Sequential(
                nn.Conv2d(in_channels2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                #downsample conv
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            )
    
    def forward1(self, x, size=(24,24)):
        if size is None:
            size = (24, 24)
        h, w = size
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
        vision1 = self.vision_projector1(x)
        if 'woconv' not in self.projector_type:
            x1 = torch.zeros(x.shape[0], self.in_channels2, h, w).to(x.device, x.dtype)
        else:
            x1 = torch.zeros(x.shape[0], vision1.shape[-1]*vision1.shape[-2], self.in_channels2).to(vision1.device, vision1.dtype)
        vision2 = self.vision_projector2(x1)
        if 'woconv' in self.projector_type:
            vision2 = vision2.permute(0, 2, 1).reshape(x.shape[0], -1, vision1.shape[-2], vision1.shape[-1])
        vision = vision1 + vision2 * 0
        return vision

    def forward2(self, x, size=(24,24)):
        if size is None:
            size = (24, 24)
        h, w = size
        
        if 'woconv' not in self.projector_type:
            x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
            x1 = torch.zeros(x.shape[0], self.in_channels1, h, w).to(x.device, x.dtype)
        else:
            x1 = torch.zeros(x.shape[0], self.in_channels1, 2*h, 2*w).to(x.device, x.dtype)

        vision1 = self.vision_projector1(x1)
        vision2 = self.vision_projector2(x)
        if 'woconv' in self.projector_type:
            vision2 = vision2.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
        vision = vision1 * 0 + vision2
        return vision

class SimpleConvNextResModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4, down=False):
        super().__init__()
        mid_channel = ratio * out_channels
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ) if in_channels != out_channels else nn.Identity()

        self.layer = nn.Sequential(
            LayerNorm2d(out_channels, eps=1e-4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, mid_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channel, out_channels, kernel_size=1),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
            ) if down else nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        y = x + self.layer(x)
        return self.downsample(y)

class ConvMlp(nn.Module):
    def __init__(self, in_channels, out_channels, param=''):
        super().__init__()
        param = param.replace('_', '')
        if param == '':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            )
        elif param == 'patch2x2hy':
            mid_channel = in_channels * 2
            self.proj = nn.Sequential(
                LayerNorm2d(in_channels, eps=1e-6),
                nn.Conv2d(in_channels, mid_channel, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel, mid_channel, 1),
                nn.GELU(),
                nn.Conv2d(mid_channel, out_channels, 1),
            )
        elif param == 'patch2x2':
            mid_channel = in_channels * 2
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel, mid_channel, 1),
                LayerNorm2d(mid_channel, eps=1e-6),
                nn.Conv2d(mid_channel, out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        
        elif param == 'patch4x4':
            mid_channel = in_channels * 2
            mid_channel2 = in_channels * 4
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel, mid_channel, 1),
                LayerNorm2d(mid_channel, eps=1e-6),
                nn.Conv2d(mid_channel, mid_channel2, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel2, mid_channel2, 1),
                LayerNorm2d(mid_channel2, eps=1e-6),
                nn.Conv2d(mid_channel2, out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1),
            )


        elif param == 'patch4x4pool':
            mid_channel = in_channels * 2
            mid_channel2 = in_channels * 4
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel, mid_channel2, 1),
                nn.AvgPool2d(2, 2),
                LayerNorm2d(mid_channel2, eps=1e-6),
                nn.Conv2d(mid_channel2, out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1),
            )
            
        elif param == 'patch2x2from4x4':
            mid_channel = in_channels * 2
            mid_channel2 = in_channels * 4
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, mid_channel, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channel, mid_channel2, 1),
                nn.Identity(),
                LayerNorm2d(mid_channel2, eps=1e-6),
                nn.Conv2d(mid_channel2, out_channels, 1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1),
            )

        else:
            depth, ratio = param.split('x')
            depth = int(depth)
            ratio = int(ratio)
            mid_channel = out_channels * ratio
            if depth < 2:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1),
                )
            else:
                modules = [SimpleConvNextResModule(in_channels, out_channels, ratio)]
                for _ in range(1, depth-1):
                    modules.append(SimpleConvNextResModule(out_channels, out_channels, ratio))
                modules.append(SimpleConvNextResModule(out_channels, out_channels, ratio, down=True))
                self.proj = nn.Sequential(*modules)

        embed_std = 1 / torch.sqrt(torch.tensor(out_channels, dtype=torch.float))
        self.image_newline = nn.Parameter(
            torch.randn(out_channels, dtype=torch.float) * embed_std
        )

    def forward(self, x, size=(24,24)):
        if size is None:
            size = (24, 24)
        h, w = size
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
        x = self.proj(x) #b,c,h,w
        b, c, h, w = x.shape
        x = torch.cat([
            x,
            self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1)
        ], dim=-1)
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        return x

class MultiPathProjector(nn.Module):
    def __init__(self, config, vision_cfg, projector_type):
        super().__init__()
        self.vision_cfg = vision_cfg

        modules1 = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        modules1.append(nn.GELU())
        modules1.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.vision_projector1 = nn.Sequential(*modules1)
        modules2 = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        modules2.append(nn.GELU())
        modules2.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.vision_projector2 = nn.Sequential(*modules2)
        

    def forward1(self, vision):
        vision1 = self.vision_projector1(vision)
        vision2 = self.vision_projector2(vision)
        vision = vision1 + vision2 * 0
        return vision
    
    def forward2(self, vision):
        vision1 = self.vision_projector1(vision)
        vision2 = self.vision_projector2(vision)
        vision = vision1 * 0 + vision2
        return vision

class OryxMLPv2(nn.Module):
    def __init__(self, in_channels, out_channels, twoview=False):
        super().__init__()
        
        self.proj1 = nn.Linear(in_channels, out_channels)
        self.proj2 = nn.Linear(out_channels, out_channels)
        self.act = nn.GELU()
        self.pooler = NormalizedDwPooler(out_channels)

        embed_std = 1 / math.sqrt(out_channels)
        self.image_newline = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        self.image_begin = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        self.image_end = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )

        self.out_channels = out_channels
        
        if twoview:
            self.image_sep = nn.Parameter(
                torch.randn(out_channels) * embed_std
            )


    def forward_mlp_for_list(self, x, size, new_line_param):
        if REGIONAL_POOL == '2x':
            split_lens = [h//2 * w//2 for h, w in size]
            dtype = x[0].dtype
            all_x = []
            for i, (h, w) in enumerate(size):
                now_x = x[i]
                now_x = now_x.reshape(1, h//2, 2, w//2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(h//2 * w//2, 2, 2, -1)
                all_x.append(now_x)
            x = torch.cat(all_x, dim=0) # b, 2, 2, c
            x = self.proj1(x)
            x = self.pooler(x, forward_type=REGIONAL_POOL)
            x = self.act(x)
            x = self.proj2(x)
            c = x.shape[-1]
            x = torch.split(x, split_lens, dim=0)

            xs = []
            for i, (h, w) in enumerate(size):
                now_x = x[i]
                now_x = now_x.reshape(1, h//2, w//2, -1)
                _, now_h, now_w, c = now_x.shape
                now_x = torch.cat([
                    now_x,
                    new_line_param.expand(1, now_h, 1, c).to(dtype)
                ], dim=2)
                now_x = now_x.reshape(1, -1, c)
                xs.append(now_x)
        elif REGIONAL_POOL == '1x':
            split_lens = [h * w for h, w in size]
            dtype = x[0].dtype
            all_x = []
            for i, (h, w) in enumerate(size):
                now_x = x[i]
                now_x = now_x.reshape(1, h, 1, w, 1, -1).permute(0, 1, 3, 2, 4, 5).reshape(h * w, 1, 1, -1)
                all_x.append(now_x)
            x = torch.cat(all_x, dim=0)
            x = self.proj1(x)
            x = self.pooler(x, forward_type=REGIONAL_POOL)
            x = self.act(x)
            x = self.proj2(x)
            c = x.shape[-1]
            x = torch.split(x, split_lens, dim=0)

            xs = []
            for i, (h, w) in enumerate(size):
                now_x = x[i]
                now_x = now_x.reshape(1, h, w, -1)
                _, now_h, now_w, c = now_x.shape
                now_x = torch.cat([
                    now_x,
                    new_line_param.expand(1, now_h, 1, c).to(dtype)
                ], dim=2)
                now_x = now_x.reshape(1, -1, c)
                xs.append(now_x)

        return xs

    def forward(self, x, size=(16,16), x2=None, size2=(16, 16)):
        if type(x) == list:
            assert REGIONAL_POOL in ['1x','2x'], 'Only 1x and 2x pooling is supported for batching MLP now'
            new_line_param = self.image_newline.reshape(1, 1, 1, self.out_channels) * 1.0
            dtype = x[0].dtype

            xs = self.forward_mlp_for_list(x, size, new_line_param)            
            if x2 is not None:
                xs2 = self.forward_mlp_for_list(x2, size2, new_line_param)
                sep = self.image_sep.reshape(1, 1, -1).expand(1, 1, self.out_channels).to(dtype)

                if ADAPTIVE_TWO_VIEWS:
                    out = []
                    for x, x2, s, s2 in zip(xs, xs2, size, size2):
                        n1 = s[0] * s[1]
                        n2 = s2[0] * s2[1]
                        ratio = n2 * 1.0 / n1
                        if ratio > 0.5 and ratio < 1.5:
                            x2 = x2 + 0.0 * x.mean() + 0.0 * sep.mean()
                            out.append(x2)
                        else:
                            out.append(torch.cat([x, sep, x2], dim=1))
                    xs = out
                else:
                    xs = [torch.cat([x, sep, x2], dim=1) for x, x2 in zip(xs, xs2)]

            begin = self.image_begin.reshape(1, 1, -1).expand(1, 1, self.out_channels).to(dtype)
            end = self.image_end.reshape(1, 1, -1).expand(1, 1, self.out_channels).to(dtype)
            xs = [torch.cat([begin, x, end], dim=1) for x in xs]
            return xs
        else:
            h, w = size
            dtype = x.dtype
            x = x.reshape(x.shape[0], h, w, -1)
            x = self.proj1(x)
            x = self.pooler(x, forward_type=REGIONAL_POOL)
            x = self.act(x)
            x = self.proj2(x)


            b, h, w, c = x.shape
            x = torch.cat([
                x,
                self.image_newline.reshape(1, 1, 1, c).expand(b, h, 1, c).to(dtype)
            ], dim=2)
            x = x.reshape(b, -1, c)

            if x2 is not None:
                h2, w2 = size2
                x2 = x2.reshape(x2.shape[0], h2, w2, -1)
                ## x2 = self.proj(x2) #b,h,w, c
                x2 = self.proj1(x2)
                x2 = self.pooler(x2, forward_type=REGIONAL_POOL)
                x2 = self.act(x2)
                x2 = self.proj2(x2)

                b2, h2, w2, c2 = x2.shape
                x2 = torch.cat([
                    x2,
                    self.image_newline.reshape(1, 1, 1, c).expand(b, h2, 1, c).to(dtype)
                ], dim=2)
                x2 = x2.reshape(b, -1, c)

                if ADAPTIVE_TWO_VIEWS:
                    n1 = h * w
                    n2 = h2 * w2
                    ratio = n2 * 1.0 / n1
                    if ratio > 0.5 and ratio < 1.5:
                        x = x2 + 0.0 * x.mean()
                    else:
                        sep = self.image_sep.reshape(1, 1, -1).expand(b, 1, c2).to(dtype)
                        x = torch.cat([x, sep, x2], dim=1)
                else:
                    sep = self.image_sep.reshape(1, 1, -1).expand(b, 1, c2).to(dtype)
                    x = torch.cat([x, sep, x2], dim=1)
            
            begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, c).to(dtype)
            end = self.image_end.reshape(1, 1, -1).expand(b, 1, c).to(dtype)
            x = torch.cat([begin, x, end], dim=1)
            return x

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'pooler':
        return PoolerProjector(config, kwargs['vision_cfg'])
    
    if projector_type == 'multipath':
        return MultiPathProjector(config, kwargs['vision_cfg'], projector_type)

    if projector_type == 'oryx_mlp_v2_twoview':
        return OryxMLPv2(config.mm_hidden_size, config.hidden_size, twoview=False)
    
    if projector_type == 'multipath_conv_mlp' or projector_type == 'multipath_conv_mlp_woconv':
        return MultiPathConvMlp(config.mm_hidden_size, VISION2_DIM, config.hidden_size, projector_type)
    
    
    if projector_type.startswith('conv_mlp'):
        param = projector_type.split('conv_mlp')[-1]
        return ConvMlp(config.mm_hidden_size, config.hidden_size, param)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r'^mlp(\d+)x_res(\d+)x_gelu$', projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
