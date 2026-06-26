# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import os, hashlib
import requests
from tqdm import tqdm
from torchvision import models
from collections import namedtuple
import argparse, os, sys, datetime, glob, importlib
import torch.nn.functional as F
from PIL import Image
import torch.distributed as dist

try:
    from llava_viq._paths import PROJECT_ROOT as _PROJECT_ROOT
except ImportError:
    _PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    )

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

if 'NO_LIMIT' in os.environ:
    NO_LIMIT = True
else:
    NO_LIMIT = False


####################### download utils ####################### 

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")    
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

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
        print("[MOVQ] loaded pretrained LPIPS loss from {}".format(ckpt))

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

####################### Decode ####################### 

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class SpatialNorm(nn.Module):
    def __init__(self, f_channels, zq_channels, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False, **norm_layer_params):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, f, zq):
        f_size = f.shape[-2:]
        zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

def Normalize(in_channels, zq_ch, add_conv):
    return SpatialNorm(in_channels, zq_ch, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=add_conv, num_groups=32, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, zq):
        h = x
        h = self.norm1(h, zq)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h, zq)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, zq):
        h_ = x
        h_ = self.norm(h_, zq)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class MOVQDecoder(nn.Module):
    def __init__(self, 
                z_channels=4, 
                # default from https://github.com/ai-forever/MoVQGAN/blob/main/configs/movqgan_67M.yaml
                ch=128,  # channel base
                ch_mult=(1, 2, 2, 4, 4),  # channel mult factor
                out_ch=3,  # output channel, always rgb
                num_res_blocks=2, # resual conv block number
                resolution=256, # do not know how to use
                attn_resolutions=[ 32 ], # trigger to use attn
                dropout=0.0, 
                resamp_with_conv=True, 
                give_pre_end=False, 
                zq_ch=None, 
                add_conv=False, 
                **ignorekwargs
            ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv)
        self.mid.attn_1 = AttnBlock(block_in, zq_ch, add_conv=add_conv)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         zq_ch=zq_ch,
                                         add_conv=add_conv))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, zq_ch, add_conv=add_conv))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, zq, return_feat=False):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            

        feat = h
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if return_feat:
            return feat, h
        return h

class decoder_head_MoVQ(nn.Module):
    def __init__(self, quant_dim, zq_ch=None, ch_mult=(1, 1, 2, 2, 4), disable_perceptual_loss=False):
        super().__init__()
        if zq_ch is None:
            zq_ch = quant_dim
        self.decoder = MOVQDecoder(z_channels=quant_dim, zq_ch=zq_ch, ch_mult=ch_mult)
        self.post_quant_conv = torch.nn.Conv2d(quant_dim, quant_dim, 1)
        self.disable_perceptual_loss = disable_perceptual_loss
        if not disable_perceptual_loss:
            self.perceptual_loss = LPIPS().eval()
        self.register_buffer('global_step', torch.tensor(0), persistent = True)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        print('enable checkpointing')

    def forward(self, quant, return_feat=False):
        # quant: B C H W
        quant2 = self.post_quant_conv(quant) # B C H W
        dec = self.decoder(quant2, quant, return_feat=return_feat) # B 3 H W
        return dec

    def forward_loss(self, input_feat, input_images, batch_limit=32):
        # LPIPS need the image be normalized by 0.5 0.5 0.5, 0.5 0.5 0.5
        input_images = input_images[:batch_limit]
        input_feat = input_feat[:batch_limit]
        target = input_images

        recon = self.forward(input_feat.permute(0, 3, 1, 2)) # B C H W
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
    
    def forward_loss_varlen(
            self, 
            input_feat,
            input_images,
            slen,
            cu_slens,
            image_sizes,
            num_prefix_tokens,
            image_limit=10000
        ):
        feature_list = input_feat.split(slen, dim=1)

        nll_loss = 0.0
        num_images = 0
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            condition =  1024 * 1024 / 16 / 16 if NO_LIMIT else 768 * 768/ 16 / 16 
            if _image_size[0] * _image_size[1] > condition:
                continue
            if num_images > image_limit:
                break
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            target = image
            recon = self.forward(shaped_feat)
            rec_loss = (target.contiguous() - recon.contiguous()) ** 2
            if self.disable_perceptual_loss:
                p_loss = 0.
            else:
                p_loss = self.perceptual_loss(target.contiguous(), recon.contiguous())
                
            nll_loss += torch.mean(rec_loss + p_loss)
            num_images += 1
            self.global_step += 1

            if self.global_step % 1000 == 0 and dist.get_rank() == 0:
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
            nll_loss = torch.tensor(0.0, device=input_feat.device)
        else:
            nll_loss = nll_loss / num_images
        return nll_loss

    def forward_recon_varlen(
            self, 
            input_feat,
            input_images,
            slen,
            cu_slens,
            image_sizes,
            num_prefix_tokens
    ):
        feature_list = input_feat.split(slen, dim=1)
        recon_image_list = []
        target_image_list = []
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            target = image
            recon = self.forward(shaped_feat)

            recon = recon.permute(0, 2, 3, 1)
            recon_image_numpy = ((recon.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            recon_image_list.append(recon_image_numpy[0]) # bs = 1

            target = target.permute(0, 2, 3, 1)
            target_image_numpy = ((target.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            target_image_list.append(target_image_numpy[0]) # bs = 1

        return recon_image_list, target_image_list


    def forward_feat_varlen(
            self, 
            input_feat,
            input_images,
            slen,
            cu_slens,
            image_sizes,
            num_prefix_tokens
    ):
        feature_list = input_feat.split(slen, dim=1)
        recon_image_list = []
        target_image_list = []
        last_feat_list = []
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            target = image
            last_feat, recon = self.forward(shaped_feat, return_feat=True)

            print('perceptual loss is:', self.perceptual_loss(target.contiguous(), recon.contiguous()))

            last_feat_list.append(last_feat)

            recon = recon.permute(0, 2, 3, 1)
            recon_image_numpy = ((recon.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            recon_image_list.append(recon_image_numpy[0]) # bs = 1

            target = target.permute(0, 2, 3, 1)
            target_image_numpy = ((target.float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            target_image_list.append(target_image_numpy[0]) # bs = 1


        return last_feat_list, recon_image_list, target_image_list
