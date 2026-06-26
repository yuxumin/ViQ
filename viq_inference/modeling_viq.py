"""ViQ model definitions (parameterized for 2k/4k/8k/16k/64k).

Shared by convert_weight.py (training weight -> ViQ weight conversion + verify)
and ViQ.py (inference with the converted ViQ weights); VAE decoder imported
from viq_train.
"""
import os as _os
import sys as _sys
# Make `llava_viq` (the training package, which holds the VAE decoder) importable.
# Resolve <repo>/viq_train relative to this file (viq_inference/modeling_viq.py),
# unless VIQ_ROOT is set explicitly.
_VIQ_ROOT = _os.environ.get(
    "VIQ_ROOT",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "viq_train"),
)
_sys.path.insert(0, _os.path.abspath(_VIQ_ROOT))

import os
import math
import time
import random
import warnings
import requests
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from PIL import Image
from torch.amp import autocast
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps, partial
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
from transformers import CLIPImageProcessor
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
try:
    from timm.layers import (
        AttentionPoolLatent,
        DropPath,
        LayerType,
        Mlp,
        PatchDropout,
        PatchEmbed,
        to_2tuple,
        resample_abs_pos_embed,
    )
    from timm.models._manipulate import checkpoint_seq, named_apply
except:
    pass


try:
    from llava_viq.model.multimodal_encoder.vae.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(_VIQ_ROOT))
    from llava_viq.model.multimodal_encoder.vae.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


class QwenImageVAEHead(nn.Module):
    def __init__(self, in_dims):
        super().__init__()

        self.conv_norm_out = nn.GroupNorm(num_channels=in_dims, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dims, 4 * 16, 3, padding=1)
        self.vae = AutoencoderKLQwenImage()
        
    @torch.no_grad()
    def forward_dist(self, image):
        try:
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        except:
            print("Please update diffusers to 0.15.1 or above to use DiagonalGaussianDistribution")
        parameters = self.vae.encode(image).latent_dist.parameters.detach()
        return DiagonalGaussianDistribution(parameters)

    def recon(self, feat):
        latents = feat.bfloat16()
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0] # B 3 H W
        # image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
        image_numpy = ((image.permute(0, 2, 3, 1).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) #B H W 3
        return image_numpy

    def get_latent_feat(self, input_feat, image_size, num_prefix_tokens=0):
        shaped_feat = input_feat[:, num_prefix_tokens:].reshape(len(input_feat), image_size[0], image_size[1], -1).permute(0, 3, 1, 2)
        shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W
        B, C, H ,W = shaped_feat.shape

        shaped_feat = shaped_feat.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)
        return shaped_feat

    def latent_feat_2_image(self, shaped_feat):
        pred_mean = shaped_feat.unsqueeze(2) # for T-dimension
        recon_image_numpy = self.recon(pred_mean)
        return recon_image_numpy

    def forward_recon(self, input_feat, image_sizes, num_prefix_tokens=0):
        shaped_feat = input_feat[:, num_prefix_tokens:].reshape(len(input_feat), image_sizes[0], image_sizes[1], -1).permute(0, 3, 1, 2) # B C H W
        shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W
        B, C, H ,W = shaped_feat.shape

        shaped_feat = shaped_feat.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)

        pred_mean = shaped_feat.unsqueeze(2)
        recon_image_numpy = self.recon(pred_mean)
        return recon_image_numpy

    def forward_recon_list(self, input_feat, slen, cu_slens, image_sizes, num_prefix_tokens=0):
        feature_list = input_feat.split(slen, dim=1)
        recon_image_numpy_list = []
        for feat, _image_size in zip(feature_list, image_sizes):
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W

            B, C, H ,W = shaped_feat.shape
            shaped_feat = shaped_feat.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)

            # mean
            pred_mean = shaped_feat.unsqueeze(2)

            
            shaped_feat = pred_mean
            recon_image_numpy = self.recon(shaped_feat)
            recon_image_numpy_list.append(recon_image_numpy[0])
                
        return recon_image_numpy_list

class GoldenGateRoPE2d(nn.Module):
    def __init__(
        self,
        head_dim: int,
        n_heads: int = 1,
        min_freq: float = 0.2,
        max_freq: float = 20.0,
        p_zero_freqs: float = 0.25,
        direction_spacing: float = math.pi * (math.sqrt(5) - 1) / 2,
    ):
        """
        Args:
            image_size: expected height and width of (patchified) input
            n_heads: number of attention heads
            head_dim: attention head dimensionality
            min_freq, max_freq: lowest and highest nonzero frequency magnitudes
            p_zero_freqs: proportion of frequencies set to 0
            direction_spacing: difference in radians between adjacent directions along
                which position is measured
        
        Dimension key:
            N: batch size
            H: image_size[0]
            W: image_size[1]
            h: n_heads
            d: head_dim
            F: num_freqs == d // 2
        """
        super().__init__()
        assert head_dim % 2 == 0
        assert 0 <= p_zero_freqs <= 1
        n_freqs = head_dim // 2
        n_zero_freqs = round(p_zero_freqs * n_freqs)
        omega_F = torch.cat(
            (
                torch.zeros(n_zero_freqs),
                min_freq
                * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
            )
        )
        phi_hF = (
            torch.arange(n_heads * n_freqs).reshape(n_heads, n_freqs)
            * direction_spacing
        )
        directions_hF2 = torch.stack((torch.cos(phi_hF), torch.sin(phi_hF)), dim=-1)
        self.freqs_hF2 = omega_F.unsqueeze(-1) * directions_hF2 # h F 2

    def forward(self, input_NHWhd: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        H, W = image_size
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_HW = torch.linspace(-xlim, xlim, W).reshape(1, W).expand(H, W)
        y_HW = torch.linspace(-ylim, ylim, H).reshape(H, 1).expand(H, W)
        positions_HW112 = torch.stack((x_HW, y_HW), dim=-1).reshape(H, W, 1, 1, 2)

        theta_HWhF = (self.freqs_hF2 * positions_HW112).sum(dim=-1)
        cos_HWhF = torch.cos(theta_HWhF)
        sin_HWhF = torch.sin(theta_HWhF)

        x_NHWhF, y_NHWhF = input_NHWhd.float().chunk(2, dim=-1)
        x_out_NHWhF = x_NHWhF * cos_HWhF.to(x_NHWhF.device) - y_NHWhF * sin_HWhF.to(x_NHWhF.device)
        y_out_NHWhF = x_NHWhF * sin_HWhF.to(x_NHWhF.device) + y_NHWhF * cos_HWhF.to(x_NHWhF.device)
        output_NHWhd = torch.cat((x_out_NHWhF, y_out_NHWhF), dim=-1)
        return output_NHWhd.type_as(input_NHWhd)

    def get_cos_sin_theta(self, image_size: tuple[int, int]):
        H, W = image_size
        xlim, ylim = math.sqrt(W / H), math.sqrt(H / W)
        x_HW = torch.linspace(-xlim, xlim, W).reshape(1, W).expand(H, W)
        y_HW = torch.linspace(-ylim, ylim, H).reshape(H, 1).expand(H, W)
        positions_HW112 = torch.stack((x_HW, y_HW), dim=-1).reshape(H, W, 1, 1, 2)
        theta_HWhF = (self.freqs_hF2 * positions_HW112).sum(dim=-1) # H W h F
        # cos_HWhF = torch.cos(theta_HWhF)
        # sin_HWhF = torch.sin(theta_HWhF)
        return theta_HWhF
    
def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)

def init_weights_vit_timm(module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""The original timm.models.layers.weight_init.trunc_normal_ can not handle bfloat16 yet, here we first
    convert the tensor to float32, apply the trunc_normal_() in float32, and then convert it back to its orignal dtype.
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn
    from the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    zhat = z.floor()
    return z + (zhat - z).detach()

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        seperate_qv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

        if seperate_qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = False
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if cu_slens is not None:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            max_seqlen = torch.max(cu_slens[1:] - cu_slens[:-1]).item()
            x = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_slens,
                cu_seqlens_k=cu_slens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scale,
                causal=False,
                )

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        else:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(q, k, v, softmax_scale=self.scale) # -> b, n, h, c

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class RoPEAttentionHeadless(nn.Module):
    '''
        we directly input the multihead feature produced by ts,
        so we do not need split head here
    '''
    fused_attn: Final[bool]

    def __init__(
        self,
        head_dim: int,
        num_heads: int = 4,
        max_head_dim: int = 256,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        seperate_qv_bias: bool = False,
    ) -> None:
        super().__init__()
        # assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = head_dim

        if self.head_dim > max_head_dim:
            self.decrease_dim = nn.Linear(self.head_dim, max_head_dim)
            self.increase_dim = nn.Linear(max_head_dim, self.head_dim)
            self.head_dim = head_dim = max_head_dim
        else:
            self.decrease_dim = self.increase_dim = nn.Identity()

        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(head_dim, head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim, head_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

        if seperate_qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(head_dim))
            self.v_bias = nn.Parameter(torch.zeros(head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:

        x = self.decrease_dim(x)

        B, N, C = x.shape # here N is 4N produced by TS
        assert N % self.num_heads == 0, f"error seq len for x. {x.shape[1]} vs {self.num_heads}"
        if cu_slens is not None:
            assert cu_slens[-1] == N // self.num_heads, f"error cu_slens for x. {cu_slens[-1]} vs {N}//{self.num_heads}"
        if rope_theta is not None:
            assert rope_theta.size(1) == N // self.num_heads, f"error for theta shape. rope_theta.shape {rope_theta.shape} vs x.shape {x.shape}"
        qkv_bias = False
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        # qkv B 4N 3C 
        # because we use token shuffle to expand the sequence length, here we use a different reshape pattern
        qkv = qkv.reshape(B, N//self.num_heads, self.num_heads, 3, self.head_dim).permute(3, 0, 2, 1, 4)  # 3, B, n_head=4, N, C
        
        # qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope_theta is not None:
            rope_theta = rope_theta.to(q.dtype) # B, N, num_heads, C
            # rope_theta from B, N, h, C to B, h, N, C
            rope_theta = rope_theta.permute(0, 2, 1, 3)

            cos_theta = torch.cos(rope_theta).to(q.device)
            sin_theta = torch.sin(rope_theta).to(q.device)

            # calculate rope2d for q
            x_q, y_q = q.chunk(2, dim=-1)
            x_out = x_q * cos_theta - y_q * sin_theta
            y_out = x_q * sin_theta + y_q * cos_theta
            q = torch.cat((x_out, y_out), dim=-1)
            
            # calculate rope2d for k
            x_k, y_k = k.chunk(2, dim=-1)
            x_out = x_k * cos_theta - y_k * sin_theta
            y_out = x_k * sin_theta + y_k * cos_theta
            k = torch.cat((x_out, y_out), dim=-1)


        if cu_slens is not None:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            max_seqlen = torch.max(cu_slens[1:] - cu_slens[:-1]).item()
            x = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_slens,
                cu_seqlens_k=cu_slens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scale,
                causal=False,
                ) # -> b, n, h, c

            x = x.reshape(B, N, C) # -> b, (n, h), c
            x = self.proj(x)
            x = self.proj_drop(x)

        else:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(q, k, v, softmax_scale=self.scale) # -> b, n, h, c

            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        x = self.increase_dim(x)

        return x

class RoPEBlockHeadless(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        max_head_dim: int = 256,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(head_dim)
        self.attn = RoPEAttentionHeadless(
            head_dim,
            num_heads=num_heads,
            max_head_dim=max_head_dim,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(head_dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(head_dim)
        self.mlp = mlp_layer(
            in_features=head_dim,
            hidden_features=int(head_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(head_dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens, rope_theta=rope_theta)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ResualMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(in_features)
        self.mlp = mlp_layer(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x

class FSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry: bool = False,
        noise_dropout = 0.0,
        symmetry_vq=False,
        ts_factor=1
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent = False)
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        
        self.symmetry_vq = symmetry_vq
        if symmetry_vq:
            self.project_out = nn.Linear(effective_codebook_dim, dim) if has_projections else nn.Identity()
        else:
            self.project_out = nn.Identity()

        output_feat_dim = self.dim
        self.project_out = nn.Linear(effective_codebook_dim, output_feat_dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

        self.forward_times = 0
        self.analysis_code_collection = []
        self.history_code_collection = []

        self.ts_factor = ts_factor

        max_head_dim = 256
        self.fsq_pre_attn = nn.Sequential(
            *[RoPEBlockHeadless(
                head_dim=self.dim,
                num_heads=self.ts_factor,
                max_head_dim=max_head_dim,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_norm=False,
                init_values=None,
                proj_drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                mlp_layer=Mlp
            ) for _ in range(3)],
        )

        rope_dim = self.dim
        if self.dim > max_head_dim:
            rope_dim = max_head_dim
        self.rope2d = GoldenGateRoPE2d(rope_dim, n_heads=self.ts_factor)
            

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels - 1)
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        shape, device, noise_dropout, preserve_symmetry, half_width = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry, (self._levels // 2)
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # determine where to add a random offset elementwise
        # if using noise dropout

        if self.training and noise_dropout > 0.:
            offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
            offset = torch.rand_like(bounded_z) - 0.5
            bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return round_ste(bounded_z) / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        # is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes


    def get_indices(
        self, 
        z,
        image_sizes,
        slen=None
        ):

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        if slen:
            assert sum(slen) == z.shape[1], f"sum(slen) = {sum(slen)}, but z.shape[1] = {z.shape[1]}"

        # prepare rope theta for this sequence
        theta_list = []
        for i in range(len(image_sizes)):
            theta_HWhF = self.rope2d.get_cos_sin_theta(image_sizes[i])
            theta_flattened = theta_HWhF.reshape(-1, theta_HWhF.shape[-2], theta_HWhF.shape[-1]).unsqueeze(0)    # shape (1, H*W, h, F)
            theta_list.append(theta_flattened)
        theta = torch.cat(theta_list, dim=1)
        assert theta.size(1)  == z.size(1) // self.ts_factor, f"{theta.size()} vs {z.size()}"

        if slen:
            slen_for_headless_attn = [length // self.ts_factor for length in slen] 
            cu_indices = [0, ]
            for i in slen_for_headless_attn:
                cu_indices.append(cu_indices[-1] + i)
            cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(z.device)
            for idx, blk in enumerate(self.fsq_pre_attn):
                z = blk(z, cu_slens=cu_slens, rope_theta=theta)
        else:
            for idx, blk in enumerate(self.fsq_pre_attn):
                z = blk(z, rope_theta=theta)


        z = self.project_in(z)

        z_in = z

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            # if self.return_indices:
            indices = self.codes_to_indices(codes) 
            assert indices.max() < self.codebook_size, f"expected max index to be < {self.codebook_size} but found {indices.max()}"
            return indices, theta
    
    def translate_indices_to_feats(self, indices, orig_dtype, slen=None, theta=None): 
        codes = self.implicit_codebook[indices]
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        codes = codes.to(orig_dtype)
        # project out
        out = self.project_out(codes)
        return out

    def forward(
            self, 
            z,
            image_sizes,
            slen=None
        ):
        is_img_or_video = z.ndim >= 4
        should_pack = is_img_or_video #default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)
        if self.channel_first:
            z = rearrange(z, 'b d ... -> b ... d')
        if should_pack:
            z, ps = pack_one(z, 'b * d')


        with torch.no_grad():
            orig_dtype = z.dtype
            indices, theta = self.get_indices(z, image_sizes=image_sizes, slen=slen)  
            out = self.translate_indices_to_feats(indices, orig_dtype, slen=slen, theta=theta)


        if should_pack:
            out = unpack_one(out, ps, 'b * d')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if self.channel_first:
            out = rearrange(out, 'b ... d -> b d ...')


        # if not self.keep_num_codebooks_dim and self.return_indices:
        #     indices = maybe(rearrange)(indices, '... 1 -> ...')

        if self.symmetry_vq:
            assert out.size()[-1] == self.dim

        # return quantized output and indices
        if slen is None:
            return out, self.zero, {"indices": indices}
        else:
            return out, [self.zero] * len(slen), {"indices": indices}

class FusionBlock(nn.Module):
    def __init__(
            self, 
            input_dim, 
            num_heads: int = 16,
            mlp_ratio: float = 4.0, 
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path = 0.0,
            norm_layer: Optional[LayerType] = partial(nn.LayerNorm, eps=1e-6),
            act_layer: Optional[LayerType] = nn.GELU,
            mlp_layer: Type[nn.Module] = Mlp,
            fsq_levels: list = [8, 8, 8, 5, 5, 5],
        ):
        super().__init__()
        self.num_features = input_dim

        self.vq_low_pre_token_shuffle_head = Mlp(
            in_features=self.num_features,
            hidden_features=4 * self.num_features,
            out_features=4 * self.num_features,
            act_layer=nn.GELU
        )
        self.vq_low_post_token_shuffle_head = Mlp(
            in_features=self.num_features,
            hidden_features=self.num_features,
            out_features=self.num_features,
            act_layer=nn.GELU
        )

        self.vq_low_inverse_token_shuffle_head = Mlp(
            in_features=4 * self.num_features,
            hidden_features=4 * self.num_features,
            out_features=self.num_features,
            act_layer=nn.GELU
        )

        self.vq_low = FSQ(
            dim = 1536,
            levels = fsq_levels, # 64k
            # levels = [8, 8, 8, 6, 5], # 16k
            symmetry_vq = True,
            ts_factor = 4
        )

        self.vq_low_norm_trick_pre_quantize_norm_wo_affine = nn.LayerNorm(self.num_features, elementwise_affine=False)
        self.vq_low_norm_trick_post_quantize_affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.vq_low_norm_trick_post_quantize_affine_bias = nn.Parameter(torch.zeros(self.num_features))

        self.vq_low_preprocess_layers = nn.Sequential(
                *[Block(
                    dim=self.num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    ) for _ in range(3)],
            )

        self.vq_low_postprocess_layers = nn.Sequential(
                Mlp(
                    in_features=self.num_features,
                    hidden_features=self.num_features,
                    out_features=self.num_features,
                    act_layer=nn.GELU
                ),
                *[ResualMLP(in_features=self.num_features, hidden_features=2 * self.num_features, out_features=self.num_features) for _ in range(3)]
            )
        
        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
    
    def forward_indices(
        self,
        indice,
        dtype
    ):
        # indices: B N
        print(f"input indice shape:", indice.shape)
        feat = self.vq_low.translate_indices_to_feats(indice, dtype)
        B, N, C = feat.shape
        feat = feat.view(B, N//4, 4 * C)  #  1 4N C ->  1 N 4C
        feat = self.vq_low_inverse_token_shuffle_head(feat) # 1 N 4C -> 1 N C

        for idx, blk in enumerate(self.vq_low_postprocess_layers):
            if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                feat = blk(feat)
            else:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    feat = checkpoint(blk, feat, use_reentrant=True)
                else:
                    feat = blk(feat)
        feat = feat * self.vq_low_norm_trick_post_quantize_affine_weight + self.vq_low_norm_trick_post_quantize_affine_bias
        print(f"output feat shape:", feat.shape)
        return feat

    def forward(
            self,
            x,
            num_prefix_tokens,
            image_sizes,
            return_indices=True
        ):
        # x : [1, sum_i (N_i), C]
        x_low = x
        x_low = self.vq_low_norm_trick_pre_quantize_norm_wo_affine(x_low)

        for idx, blk in enumerate(self.vq_low_preprocess_layers):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x_low = checkpoint(blk, x_low, use_reentrant=True)
            else:
                x_low = blk(x_low)

        if num_prefix_tokens == 0:
            B, N, C = x_low.shape
            x_low = self.vq_low_pre_token_shuffle_head(x_low) # 1 N C  -> 1 N 4C
            x_low = x_low.view(B, 4 * N, C) # 1 N 4C -> 1 4N C
            x_low = self.vq_low_post_token_shuffle_head(x_low) #  1 4N C ->  1 4N C
        else:
            raise NotImplementedError(f"TS")

        x_low, _, info = self.vq_low(x_low, image_sizes=image_sizes)

        if num_prefix_tokens == 0:
            B, N, C = x_low.shape
            x_low = x_low.view(B, N//4, 4 * C)  #  1 4N C ->  1 N 4C
            x_low = self.vq_low_inverse_token_shuffle_head(x_low) # 1 N 4C -> 1 N C
        else:
            raise NotImplementedError(f"TS")


        for idx, blk in enumerate(self.vq_low_postprocess_layers):
            if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                x_low = blk(x_low)
            else:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x_low = checkpoint(blk, x_low, use_reentrant=True)
                else:
                    x_low = blk(x_low)

        x_low = x_low * self.vq_low_norm_trick_post_quantize_affine_weight + self.vq_low_norm_trick_post_quantize_affine_bias
        if return_indices:
            return x_low, info["indices"]
        return x_low, None

    def forward_list(
            self,
            x,
            num_prefix_tokens,
            image_sizes,
            slen,
            cu_slens,
            return_indices=True
        ):
        # x : [1, sum_i (N_i), C]
        x_low = x

        x_low = self.vq_low_norm_trick_pre_quantize_norm_wo_affine(x_low)

        for idx, blk in enumerate(self.vq_low_preprocess_layers):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
            else:
                x_low = blk(x_low, cu_slens=cu_slens)

        if num_prefix_tokens == 0:
            B, N, C = x_low.shape
            x_low = self.vq_low_pre_token_shuffle_head(x_low) # 1 N C  -> 1 N 4C
            x_low = x_low.view(B, 4 * N, C) # 1 N 4C -> 1 4N C
            x_low = self.vq_low_post_token_shuffle_head(x_low) #  1 4N C ->  1 4N C
        else:
            raise NotImplementedError(f"TS")

        xs = x_low.split([length * 4 for length in slen], dim=1)
        vq_x = torch.cat(xs, dim=1)
        vq_slen = [length * 4 for length in slen] 
        x_low, _, info = self.vq_low(vq_x, slen=vq_slen, image_sizes=image_sizes)

        if num_prefix_tokens == 0:
            B, N, C = x_low.shape
            x_low = x_low.view(B, N//4, 4 * C)  #  1 4N C ->  1 N 4C
            x_low = self.vq_low_inverse_token_shuffle_head(x_low) # 1 N 4C -> 1 N C
        else:
            raise NotImplementedError(f"TS")

        if isinstance(self.vq_low_postprocess_layers, torch.nn.Sequential):
            for idx, blk in enumerate(self.vq_low_postprocess_layers):
                if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                    x_low = blk(x_low)
                else:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                    else:
                        x_low = blk(x_low, cu_slens=cu_slens)
        else:
            x_low = self.vq_low_postprocess_layers(x_low)

        x_low = x_low * self.vq_low_norm_trick_post_quantize_affine_weight + self.vq_low_norm_trick_post_quantize_affine_bias
        if return_indices:
            return x_low, info["indices"]
        return x_low, None
        
class ViqEncoder(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000, # type: ignore
        global_pool: Literal["", "avg", "token", "map"] = "token", # type: ignore
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        strict_img_size: bool = False,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
        fsq_levels: list = [8, 8, 8, 5, 5, 5],
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            strict_img_size=strict_img_size,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # NOTE: newly added part
        self.fusion_block = FusionBlock(self.embed_dim, fsq_levels=fsq_levels)
        self.mllm_out_dims = self.embed_dim
        self.vae_preprocess_layers = nn.Sequential(
            Mlp(
                in_features=embed_dim,
                hidden_features=1536,
                out_features=1536,
                act_layer=nn.GELU
            ),
            norm_layer(1536),
            *[Block(
                dim=1536,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ) for _ in range(3)]
        )
        self.vae_head = QwenImageVAEHead(
            in_dims=1536
        )

        if weight_init != "skip":
            self.init_weights(weight_init)
        
    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        # head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @property
    def device(self):
        return self.pos_embed.device

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        self.fusion_block.set_grad_checkpointing(enable)

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        pos_embed_shape = int((self.pos_embed.shape[1]) ** 0.5)
        if (h, w) == (pos_embed_shape, pos_embed_shape):
            return self.pos_embed
        rescaled_positional_embedding = \
            self.pos_embed.new_zeros(1, h*w, self.pos_embed.shape[2])
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        if torch.__version__ == '2.0.0':
            dtype = pe_2d.dtype
            pe_2d = F.interpolate(pe_2d.float(), out_size, mode='bilinear', align_corners=False).to(dtype).view(-1, h*w)
        else:
            pe_2d = F.interpolate(pe_2d, out_size, mode='bilinear', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[0] = pe_2d.T.contiguous()
        return rescaled_positional_embedding

    def forward_features_list(self, x_list, plot=False):
        x_all = []
        image_sizes = []
        for x in x_list:
            bs, _, h, w = x.shape

            # fix patch size=14 in datasets 
            pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
            pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

            bs, _, h, w = x.shape

            h = h // self.patch_embed.patch_size[0]
            w = w // self.patch_embed.patch_size[1]

            x = self.patch_embed(x)
            x = x + self.rescale_positional_embedding(out_size=(h, w))
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            x_all.append(x)
            image_sizes.append((h, w))

        slen = [xi.size(1) for xi in x_all]
        x = torch.cat(x_all, dim=1)

        cu_indices = [0, ]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)

        cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(x.device)
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, cu_slens, use_reentrant=True)
            else:
                x = blk(x, cu_slens=cu_slens)

        x, indices = self.fusion_block.forward_list(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            image_sizes=image_sizes,
            slen=slen,
            cu_slens=cu_slens,
            return_indices=True
        )

        if plot:
            seq_feature = x
            for idx, blk in enumerate(self.vae_preprocess_layers):
                if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                    seq_feature = blk(seq_feature)
                else:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        seq_feature = checkpoint(blk, seq_feature, cu_slens, use_reentrant=True)
                    else:
                        seq_feature = blk(seq_feature, cu_slens=cu_slens)

            image_numpy_list = self.vae_head.forward_recon_list(
                seq_feature,
                slen,
                cu_slens,
                image_sizes,
                self.num_prefix_tokens
            )
        else:
            image_numpy_list = None

        x = x.split(slen, dim=1)
        indices = indices.split([s * 4 for s in slen], dim=1)

        return x, image_sizes, image_numpy_list, indices

    def forward_features(self, x, plot=False):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]

        x = self.patch_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, use_reentrant=True)
            else:
                x = blk(x)

        x, indices = self.fusion_block(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            image_sizes=[(h, w)],
            return_indices=True
        )

        if plot:
            seq_feature = x
            for idx, blk in enumerate(self.vae_preprocess_layers):
                if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                    seq_feature = blk(seq_feature)
                else:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        seq_feature = checkpoint(blk, seq_feature, use_reentrant=True)
                    else:
                        seq_feature = blk(seq_feature)

            image_numpy = self.vae_head.forward_recon(
                seq_feature,
                (h, w),
                self.num_prefix_tokens
            )
        else:
            image_numpy = None

        return  x, (h, w), image_numpy, indices

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
    
    def forward(self, x, cal_attn_pool=False, plot=False):
        if type(x) is list:
            x, image_sizes, image_numpy_list, indices = self.forward_features_list(x, plot=plot)
            if cal_attn_pool:
                cls_tokens = []
                for cur_x in x:
                    cls_tokens.append(self.forward_head(cur_x))
                cls_tokens = torch.cat(cls_tokens, dim=0)
                return x, image_sizes, cls_tokens, image_numpy_list, indices
            return x, image_sizes, None, image_numpy_list, indices
        else:
            x, image_sizes, image_numpy, indices = self.forward_features(x, plot=plot)
            if cal_attn_pool:
                cls_token = self.forward_head(x)
                return x, image_sizes, cls_token, image_numpy, indices
            return x, image_sizes, None, image_numpy, indices


    def get_image_indices(self, x_list):
        x_all = []
        image_sizes = []
        for x in x_list:
            bs, _, h, w = x.shape

            # fix patch size=14 in datasets 
            pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
            pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

            bs, _, h, w = x.shape

            h = h // self.patch_embed.patch_size[0]
            w = w // self.patch_embed.patch_size[1]

            x = self.patch_embed(x)
            x = x + self.rescale_positional_embedding(out_size=(h, w))
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            x_all.append(x)
            image_sizes.append((h, w))

        slen = [xi.size(1) for xi in x_all]
        x = torch.cat(x_all, dim=1)

        cu_indices = [0, ]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)

        cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(x.device)
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, cu_slens, use_reentrant=True)
            else:
                x = blk(x, cu_slens=cu_slens)

        _, indices = self.fusion_block.forward_list(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            image_sizes=image_sizes,
            slen=slen,
            cu_slens=cu_slens,
            return_indices=True
        )
        indices = indices.split([s * 4 for s in slen], dim=1)

        return indices, image_sizes

    def get_vae_feat(self, feat, image_size):
        seq_feature = feat
        for idx, blk in enumerate(self.vae_preprocess_layers):
            if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                seq_feature = blk(seq_feature)
            else:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    seq_feature = checkpoint(blk, seq_feature, use_reentrant=True)
                else:
                    seq_feature = blk(seq_feature)
        print("seq_feature", seq_feature.shape)
        vae_latent_feat = self.vae_head.get_latent_feat(seq_feature, image_size) # B C H W  torch.Size([1, 16, 128, 128]) # patch size = 8
        
        return vae_latent_feat
        
    def index_2_feat(self, indices, dtype, image_size, get_vae_feat=False):
        feat = self.fusion_block.forward_indices(indices, dtype)
        # feat : B N C, torch.Size([1, 4096, 1536]) # N = H*W/16/16
        if get_vae_feat:
            vae_latent_feat = self.get_vae_feat(feat, image_size)
            return feat, vae_latent_feat
        return feat, None

    def vae_feat_2_image(self, vae_latent_feat):
        image_numpy = self.vae_head.latent_feat_2_image(vae_latent_feat)
        return image_numpy
    
class IndexEmbeder(nn.Module):
    def __init__(self, weight=None, codedim=6, feature_dims=1536, codesize=64000):
        super().__init__()
        self.ori_codebook = nn.Embedding(codesize, codedim)
        self.projector_1 = nn.Linear(codedim, feature_dims, bias=True)
        self.projector_2 = Mlp(
            4 * feature_dims, 
            4 * feature_dims, 
            feature_dims, 
            act_layer=nn.GELU
        )
        self.projector_3 = nn.Sequential(
                Mlp(
                    in_features=feature_dims,
                    hidden_features=feature_dims,
                    out_features=feature_dims,
                    act_layer=nn.GELU
                ),
                *[ResualMLP(in_features=feature_dims, hidden_features=feature_dims*2, out_features=feature_dims) for _ in range(3)]
            )
        self._weight = nn.Parameter(torch.ones(feature_dims))
        self._bias = nn.Parameter(torch.zeros(feature_dims))


        if weight is not None:
            state_dict = torch.load(weight, map_location="cpu")
            incompatible_keys = self.load_state_dict(state_dict, strict=True)
            print(
                f"Embedder restores from {weight}\n"
                f"\tincompatible_keys:', {incompatible_keys}."
            )

    def init_weight_from_viq_weight(self, viq_weight, implicit_codebook, save_path):
        state_dict = torch.load(viq_weight, map_location="cpu")

        new_state_dict = {}
        for k in state_dict.keys():
            if 'vq_low.project_out' in k:
                new_k = k.replace('fusion_block.vq_low.project_out', 'projector_1')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_inverse_token_shuffle_head' in k:
                new_k = k.replace('fusion_block.vq_low_inverse_token_shuffle_head', 'projector_2')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_postprocess_layers' in k:
                new_k = k.replace('fusion_block.vq_low_postprocess_layers', 'projector_3')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_norm_trick_post_quantize_affine_weight' in k:
                new_k = k.replace('fusion_block.vq_low_norm_trick_post_quantize_affine_weight', '_weight')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_norm_trick_post_quantize_affine_bias' in k:
                new_k = k.replace('fusion_block.vq_low_norm_trick_post_quantize_affine_bias', '_bias')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_projectout' in k:
                new_k = k.replace('fusion_block.vq_low_projectout', 'projector_aux')
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict
        state_dict['ori_codebook.weight'] = implicit_codebook
        
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        print(
            f"Embedder restores from {viq_weight}\n"
            f"\tincompatible_keys:', {incompatible_keys}."
        )

        # save_path = os.path.join(os.path.dirname(viq_weight), 'embedder.pth')
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)

    def forward(self, indices):
        if isinstance(indices, list) or isinstance(indices, tuple):
            slen = [ind.size(1) // 4 for ind in indices]
            embedding_feat = self.forward(torch.cat(indices, dim=1))
            return embedding_feat.split(slen, dim=1)
        else:
            # indice (b, n )
            if len(indices.shape) == 3:
                indices = indices.squeeze(-1) # sometimes the indices will be B N 1
            assert len(indices.shape) == 2
            feat = self.projector_1(self.ori_codebook(indices)) # B N C
            B, N, C = feat.shape
            feat = feat.view(B, N//4, 4 * C)
            feat = self.projector_3(self.projector_2(feat))
            feat = feat * self._weight + self._bias
            return feat

class IndexDrawer(nn.Module):
    def __init__(self, weight=None, codedim=6, feature_dims=1536, codesize=64000):
        super().__init__()
        self.ori_codebook = nn.Embedding(codesize, codedim)
        self.projector_1 = nn.Linear(codedim, feature_dims, bias=True)
        self.projector_2 = Mlp(
            4 * feature_dims, 
            4 * feature_dims, 
            feature_dims, 
            act_layer=nn.GELU
        )
        self.projector_3 = nn.Sequential(
                Mlp(
                    in_features=feature_dims,
                    hidden_features=feature_dims,
                    out_features=feature_dims,
                    act_layer=nn.GELU
                ),
                *[ResualMLP(in_features=feature_dims, hidden_features=feature_dims*2, out_features=feature_dims) for _ in range(3)]
            )
        self._weight = nn.Parameter(torch.ones(feature_dims))
        self._bias = nn.Parameter(torch.zeros(feature_dims))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.vae_preprocess_layers = nn.Sequential(
            Mlp(
                in_features=1536,
                hidden_features=1536,
                out_features=1536,
                act_layer=nn.GELU
            ),
            norm_layer(1536),
            *[Block(
                dim=1536,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                qk_norm=False,
                init_values=None,
                proj_drop=0,
                attn_drop=0,
                drop_path=0,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
                mlp_layer=Mlp,
            ) for _ in range(3)]
        )
        self.vae_head = QwenImageVAEHead(in_dims=1536)


        if weight is not None:
            state_dict = torch.load(weight, map_location="cpu")
            incompatible_keys = self.load_state_dict(state_dict, strict=True)
            print(
                f"Drawer restores from {weight}\n"
                f"\tincompatible_keys:', {incompatible_keys}."
            )

    def init_weight_from_viq_weight(self, viq_weight, implicit_codebook, save_path):
        state_dict = torch.load(viq_weight, map_location="cpu")

        new_state_dict = {}
        for k in state_dict.keys():
            if 'vq_low.project_out' in k:
                new_k = k.replace('fusion_block.vq_low.project_out', 'projector_1')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_inverse_token_shuffle_head' in k:
                new_k = k.replace('fusion_block.vq_low_inverse_token_shuffle_head', 'projector_2')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_postprocess_layers' in k:
                new_k = k.replace('fusion_block.vq_low_postprocess_layers', 'projector_3')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_norm_trick_post_quantize_affine_weight' in k:
                new_k = k.replace('fusion_block.vq_low_norm_trick_post_quantize_affine_weight', '_weight')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_norm_trick_post_quantize_affine_bias' in k:
                new_k = k.replace('fusion_block.vq_low_norm_trick_post_quantize_affine_bias', '_bias')
                new_state_dict[new_k] = state_dict[k]
            if 'vq_low_projectout' in k:
                new_k = k.replace('fusion_block.vq_low_projectout', 'projector_aux')
                new_state_dict[new_k] = state_dict[k]
            if 'vae_preprocess_layers' in k:
                new_state_dict[k] = state_dict[k]
            if 'vae_head' in k:
                new_state_dict[k] = state_dict[k]

        state_dict = new_state_dict
        state_dict['ori_codebook.weight'] = implicit_codebook
        
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        print(
            f"Drawer restores from {viq_weight}\n"
            f"\tincompatible_keys:', {incompatible_keys}."
        )

        # save_path = os.path.join(os.path.dirname(viq_weight), 'index_drawer.pth')
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)

    def forward(self, indices, image_sizes):
        if isinstance(indices, list) or isinstance(indices, tuple):
            feats = []
            vae_latent_feats = []
            image_numpy_list = []
            for indice, image_size in zip(indices, image_sizes):
                feat, vae_latent_feat, image_numpy = self.forward_single(indice, image_size)
                feats.append(feat)
                vae_latent_feats.append(vae_latent_feat)
                image_numpy_list.append(image_numpy[0]) # remove batch size dimension
            return feats, vae_latent_feats, image_numpy_list
        else:
            return self.forward_single(indices, image_sizes)

    def forward_single(self, indice, image_size):
        if len(indice.shape) == 3:
            indice = indice.squeeze(-1)
        assert len(indice.shape) == 2
        feat = self.projector_1(self.ori_codebook(indice)) # B N C
        B, N, C = feat.shape
        feat = feat.view(B, N//4, 4 * C)
        feat = self.projector_3(self.projector_2(feat))
        feat = feat * self._weight + self._bias # 1 N C
        vae_latent_feat = self.vae_preprocess_layers(feat)
        vae_latent_feat = self.vae_head.get_latent_feat(vae_latent_feat, image_size) # B C H W  torch.Size([1, 16, 128, 128]) # patch size = 8
        image_numpy = self.vae_head.latent_feat_2_image(vae_latent_feat)
        return feat, vae_latent_feat, image_numpy
    
@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False

SigLIP_MODEL_CONFIG = {
    "siglip2_giant_patch16_384":{
        "image_size": 384,
        "patch_size": 16,
        "width": 1536,
        "layers": 40,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    }
}

def resize_evaclip_pos_embed(model: ViqEncoder, interpolation: str = 'bicubic'):
    # interpolate position embedding
    orig_size = 24
    new_size = 128
    pos_tokens = model.pos_embed
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, model.embed_dim).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode=interpolation, align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    model.pos_embed = nn.Parameter(pos_tokens, requires_grad=True)
    return model 

def create_siglip_vit(
    model_name: str = "siglip2_giant_patch16_384",
    image_size: int = 384,
    select_layer: int = -1,
    ckpt_path: str = "",
    teacher: bool = False,
    gradient_checkpointing: bool = False,
    fsq_levels: list = [8, 8, 8, 5, 5, 5],
    **kwargs,
):
    print('kwargs:', kwargs)
    assert (
        model_name in SigLIP_MODEL_CONFIG.keys()
    ), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    model = ViqEncoder(
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        dynamic_img_pad=False,
        strict_img_size=teacher,
        ignore_head=kwargs.get("ignore_head", False),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
        drop_path_rate=0.0 if not teacher else 0.0,
        fsq_levels=fsq_levels
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # if ckpt_path.endswith(".pth"):
        #     new_state_dict = {}
        #     for k in state_dict.keys():
        #         if 'perceptual_loss' in k:
        #             continue
        #         if k.startswith('base_model.model.model.vision_tower.vision_tower.'):
        #             new_k = k.replace('base_model.model.model.vision_tower.vision_tower.', '')

        #             if 'dualvq_head.' in new_k:
        #                 new_k = new_k.replace('dualvq_head.', 'fusion_block.')
        #             if 'movq' in new_k:
        #                 new_k = new_k.replace('movq', 'vae')
        #             # if 'vq_low.' in new_k:
        #             #     new_k = new_k.replace('vq_low.', 'bn_')
                    
        #             if 'vae_cpu' in new_k:
        #                 continue

        #             new_state_dict[new_k] = state_dict[k]

        #     new_state_dict = {}
        #     for key in state_dict.keys():
        #         if key.startswith("visual.trunk."):
        #             new_state_dict[key[13:]] = state_dict[key]

        # state_dict = new_state_dict

        if not teacher:
            model = resize_evaclip_pos_embed(model, interpolation='bilinear')
            patch_embed = state_dict['patch_embed.proj.weight']
            if patch_embed.shape[-1] != model.patch_embed.proj.weight.shape[-1]:
                patch_embed = torch.nn.functional.interpolate(
                    patch_embed.float(), size=(vision_cfg.patch_size, vision_cfg.patch_size), mode='bicubic', align_corners=False)
                print(f'interpolate model patch size to {vision_cfg.patch_size}...')
                state_dict['patch_embed.proj.weight'] = patch_embed
            pos_embed = state_dict['pos_embed']

            if pos_embed.shape[1] != model.pos_embed.shape[1]:
                pos_embed = pos_embed.reshape(1, 24, 24, 1536).permute(0, 3, 1, 2)
                pos_embed = torch.nn.functional.interpolate(pos_embed, size=(128, 128), mode='bicubic', align_corners=False)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, 1536)
                print(f'interpolate model pos embed size to 128...')
            state_dict['pos_embed'] = pos_embed

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"SigLIP-ViT restores from {ckpt_path}, Act as Teacher? {teacher}\n"
            f"\tincompatible_keys:', {incompatible_keys}."
        )
    if gradient_checkpointing:
        model.set_grad_checkpointing(True)

    return model

def create_image_preprocess():
    # image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_path = _os.path.abspath(_os.path.join(
        _VIQ_ROOT, 'llava_viq', 'model', 'multimodal_encoder', 'assets', 'default_processor'))
    print(f'Loading CLIPImageProcessor from {clip_path}')
    image_processor = CLIPImageProcessor.from_pretrained(clip_path)
    image_processor.image_mean = [0.5, 0.5, 0.5]
    image_processor.image_std = [0.5, 0.5, 0.5]
    image_processor.do_resize = False
    image_processor.do_center_crop = False
    return image_processor

def resize_and_center_crop(image, target_size=512):
    width, height = image.size
    
    if width < height:
        new_width = target_size
        new_height = int((target_size / width) * height)
    else:
        new_height = target_size
        new_width = int((target_size / height) * width)
    
    image = image.resize((new_width, new_height))
    width, height = image.size
    
    left = (width - target_size) / 2
    top = (height - target_size) / 2
    right = (width + target_size) / 2
    bottom = (height + target_size) / 2
    
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def process_image(image, image_processor, target_size):
    image = resize_and_center_crop(image, target_size=target_size)
    image_size = image.size
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image, image_size

def get_image_data_input():
    url_list = [
        'https://images.pexels.com/photos/792381/pexels-photo-792381.jpeg',
        'https://images.pexels.com/photos/1570264/pexels-photo-1570264.jpeg',
        'https://images.pexels.com/photos/1051073/pexels-photo-1051073.jpeg',
        'https://images.pexels.com/photos/831430/pexels-photo-831430.jpeg',
        "https://images.pexels.com/photos/3782131/pexels-photo-3782131.jpeg",
        "https://images.pexels.com/photos/6044974/pexels-photo-6044974.jpeg"
    ]
    image_processor = create_image_preprocess()
    images = []
    image_sizes = []
    for i, url in enumerate(url_list):
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        image, image_size = process_image(image=img, image_processor=image_processor, target_size=1536)
        images.append(image.unsqueeze(0).cuda().bfloat16())
        image_sizes.append(image_size)
    
    for i, url in enumerate(url_list):
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        image, image_size = process_image(image=img, image_processor=image_processor, target_size=512)
        images.append(image.unsqueeze(0).cuda().bfloat16())
        image_sizes.append(image_size)

    return images, image_sizes

class AnyResViqVQWrapper(nn.Module):
    def __init__(self, config, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.select_layer = -1
        if self.select_layer < -1: self.select_layer += 1

        self.path = config['weight_root']
        self.codebook_size = config['codebook_size']
        self.levels = config['levels']

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):

        ckpt_path = self.path
        embedder_path = os.path.join(os.path.dirname(ckpt_path), 'embedder.pth')
        drawer_path = os.path.join(os.path.dirname(ckpt_path), 'index_drawer.pth')

        self.vision_tower = create_siglip_vit(ckpt_path=ckpt_path, fsq_levels=self.levels)
        self.embedder = IndexEmbeder(weight=embedder_path, codedim=len(self.levels), codesize=self.codebook_size)
        self.drawer = IndexDrawer(weight=drawer_path, codedim=len(self.levels), codesize=self.codebook_size)

        self.image_processor = create_image_preprocess()
        print("Loading vision model...")

        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()

        for p in self.embedder.parameters():
            p.requires_grad = False
        self.embedder.eval()

        for p in self.drawer.parameters():
            p.requires_grad = False
        self.drawer.eval()

        self.is_loaded = True

    def train(self, mode = True):
        self.training = mode
        self.vision_tower.eval()
        self.embedder.eval()
        self.drawer.eval()

    def forward_func(self, images):
        if type(images) is list:
            xs = [x.to(self.dtype) for x in images]
            indices, image_sizes = self.vision_tower.get_image_indices(xs)
            # indices = self.vision_tower.get_image_indices(xs)
            feats = self.embedder(indices) # list forward List of [ (1 H*W//16//16 C) ]
            indices = [ind.squeeze(-1) for ind in indices]
        else:
            raise
        return indices, feats, image_sizes
    
    def forward(self, images, cal_attn_pool=False):
        with torch.no_grad():
            indices, feats, image_sizes = self.forward_func(images, cal_attn_pool=cal_attn_pool)
            return indices, feats, image_sizes

    def forward_indices(self, images, skip_fusion=False):
        if type(images) is list:
            xs = [x.to(self.dtype) for x in images]
            indices, image_sizes = self.vision_tower.get_image_indices(xs)
            return indices, image_sizes
        else:
            raise
        return indices, image_sizes
    
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.vision_tower.set_grad_checkpointing(enable)

    @property
    def dummy_feature(self):
        dim = self.vision_tower.mllm_out_dims
        return torch.zeros(1, dim, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.pos_embed.dtype

    @property
    def device(self):
        return self.vision_tower.pos_embed.device

    @property
    def hidden_size(self):
        dim = self.vision_tower.mllm_out_dims
        _hidden_size = dim
        return _hidden_size

    @property
    def config(self):
        return type('LLaVAConfigWrapper', (), {
            'patch_size': 16,
        })()



