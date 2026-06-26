# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import models
from collections import namedtuple
import argparse, os, sys, datetime, glob, importlib
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from PIL import Image
import torch.distributed as dist
from timm.layers import Mlp, resample_abs_pos_embed
from ..layers.model_utils import *

try:
    from llava_viq._paths import PROJECT_ROOT as _PROJECT_ROOT
except ImportError:
    _PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    )

if 'THIS_EXP_NAME' in os.environ:
    THIS_EXP_NAME = os.environ['THIS_EXP_NAME']
else:
    THIS_EXP_NAME = 'unknown'


####################### LPIPS ####################### 
class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'model', 'lpips'))
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("[VQVIT] loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

class ScalingLayer(nn.Module):
    # make Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class ViTDecoder(nn.Module):
    def __init__(
                self, 
                in_dims: int,
                dim: int = 768, 
                depth: int = 12, 
                heads: int = 12, 
                mlp_dim: int = 3072, 
                image_size: int = 512,
                patch_size: int = 16,
                channels: int = 3,
                pos_drop_rate: float = 0.0,
                disable_perceptual_loss=False
        ) -> None:
        super().__init__()
        self.grad_checkpointing = False
        self.in_conv = Mlp(
                in_features=in_dims,
                hidden_features=mlp_dim,
                out_features=dim,
                act_layer=nn.GELU
            )
        self.pos_embed = nn.Parameter(torch.randn(1, image_size * image_size // patch_size // patch_size, dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        layers = [
            Block(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_dim//dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.Tanh,
                mlp_layer=Mlp,
            ) for _ in range(depth)
        ] 
        self.transformer = nn.Sequential(*layers)

        self.to_pixel = nn.ConvTranspose2d(dim, channels, kernel_size=patch_size, stride=patch_size)
        self.apply(init_weights)
        self.register_buffer('global_step', torch.tensor(0), persistent = True)

        self.disable_perceptual_loss = disable_perceptual_loss
        if not disable_perceptual_loss:
            self.perceptual_loss = LPIPS().eval()


    def _pos_embed(self, x: torch.Tensor, dynamic_img_size=True) -> torch.Tensor:
        if dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        return self.pos_drop(x)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward_batch(self, input_feat, input_images, batch_limit=32, only_return_feature=False):
        # input feat = b H W C
        input_images = input_images[:batch_limit]
        input_feat = input_feat[:batch_limit]

        target = input_images # B 3 H W

        input_feat = self.in_conv(input_feat)
        B, H, W, C = input_feat.shape
        input_feat = self._pos_embed(input_feat.view(B, -1, C), dynamic_img_size=False)

        # transformer
        for idx, blk in enumerate(self.transformer):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                input_feat = checkpoint(blk, input_feat, use_reentrant=True)
            else:
                input_feat = blk(input_feat)

        if only_return_feature:
            return input_feat

        input_feat = input_feat.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        recon = self.to_pixel(input_feat) # B 3 H W

        rec_loss = (target.contiguous() - recon.contiguous()) ** 2
        if self.disable_perceptual_loss:
            p_loss = 0.
        else:
            p_loss = self.perceptual_loss(target.contiguous(), recon.contiguous())
        nll_loss = rec_loss + p_loss
        nll_loss = torch.mean(nll_loss)

        self.global_step += 1

        if self.global_step % 100 == 0 and dist.get_rank() == 0:
            recon = recon.permute(0, 2, 3, 1)
            recon_image_numpy = ((recon.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)[0]
            target = target.permute(0, 2, 3, 1)
            target_image_numpy = ((target.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)[0]
            
            combined = np.hstack([recon_image_numpy, target_image_numpy])
            save_dir = os.path.join(os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plot'), THIS_EXP_NAME)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
            image = Image.fromarray(combined, mode='RGB')
            image.save(save_path)

        return nll_loss


    def forward(self, x, input_images, slen, cu_slens, image_sizes, num_prefix_tokens=0, plot=True, only_return_loss=True, only_return_feature=False, image_limit=10000):

        # add pos_emb
        x = self.in_conv(x)
        xs = x.split(slen, dim=1)
        x_all = []
        for _x, _image_size in zip(xs, image_sizes):
            assert num_prefix_tokens == 0
            _x = _x[:, num_prefix_tokens:].reshape(len(_x), _image_size[0], _image_size[1], -1)
            _x = self._pos_embed(_x)
            x_all.append(_x)
        x = torch.cat(x_all, dim=1)

        # transformer
        for idx, blk in enumerate(self.transformer):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, cu_slens, use_reentrant=True)
            else:
                x = blk(x, cu_slens=cu_slens)

        if only_return_feature:
            return x

        xs = x.split(slen, dim=1)
        nll_loss = 0
        num_images = 0
        recon_image_list = []
        target_image_list = []

        for _x, input_image, _image_size in zip(xs, input_images, image_sizes):
            #     # placeholder image and empty image
            #     continue
            if num_images > image_limit:
                break
            _x = _x[:, num_prefix_tokens:].reshape(len(_x), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) 
            _x = self.to_pixel(_x)

            target = input_image
            recon = _x

            recon_image = recon.permute(0, 2, 3, 1)
            recon_image_numpy = ((recon_image.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            recon_image_list.append(recon_image_numpy[0]) # bs = 1

            target_image = target.permute(0, 2, 3, 1)
            target_image_numpy = ((target_image.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            target_image_list.append(target_image_numpy[0]) # bs = 1


            rec_loss = (target.contiguous() - recon.contiguous()) ** 2
            if self.disable_perceptual_loss:
                p_loss = 0.
            else:
                p_loss = self.perceptual_loss(target.contiguous(), recon.contiguous())
                
            nll_loss += torch.mean(rec_loss + p_loss)
            num_images += 1
            self.global_step += 1

            if self.global_step % 1000 == 0 and dist.get_rank() == 0 and plot:
                recon = recon.permute(0, 2, 3, 1)
                recon_image_numpy = ((recon.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)[0]
                target = target.permute(0, 2, 3, 1)
                target_image_numpy = ((target.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)[0]
                
                combined = np.hstack([recon_image_numpy, target_image_numpy])
                save_dir = os.path.join(os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plot'), THIS_EXP_NAME)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
                image = Image.fromarray(combined, mode='RGB')
                image.save(save_path)

        if num_images == 0:
            nll_loss = torch.tensor(0.0, device=x.device)
        else:
            nll_loss = nll_loss / num_images

        if only_return_loss:
            return nll_loss
        else:
            return recon_image_list, target_image_list
