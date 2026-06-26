import math
import warnings
from dataclasses import dataclass
from functools import partial
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
import numpy as np
import random
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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

from flash_attn import flash_attn_func, flash_attn_varlen_func
import requests
from tqdm import tqdm
from typing import Optional
import hashlib
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import os

from ..envir_defines import *



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
        return theta_HWhF


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=(2, 3), keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma.view(1, self.dim, 1, 1) * x_norm

class ConvStem(nn.Module):
    # two stems
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=5,
        norm_layer=None,
        checkpointing=False,
    ):
        super().__init__()

        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.depth = depth

        # stem 1
        stem1 = []
        input_dim, _ = in_chans, embed_dim // (2 ** (depth - 1))

        self.depth1 = depth1 = 1
        output_dims = [2048]
        kernal_sizes = [32]
        strides = [32]
        paddings = [0]


        assert len(kernal_sizes) == self.depth1
        output_dim = output_dims[-1]
        for idx in range(depth1):
            stage1_list = [
                nn.Conv2d(
                    input_dim,
                    output_dims[idx],
                    kernel_size=kernal_sizes[idx],
                    stride=strides[idx],
                    padding=paddings[idx],
                    bias=False,
                ),
            ]
            if idx == depth1 - 1 and output_dims[idx] != embed_dim:
                stage1_list.append(nn.Conv2d(output_dims[idx], embed_dim, kernel_size=1))
            stage1 = nn.Sequential(*stage1_list)
            input_dim = output_dims[idx]
            stem1.append(stage1)
        self.proj1 = nn.ModuleList(stem1)


        # stem 2
        stem2 = []
        input_dim, _ = in_chans, embed_dim // (2 ** (depth - 1))

        self.depth2 = depth2 = 4
        output_dims = [64, 64, 128, 512]
        kernal_sizes = [4, 3, 3, 3]
        strides = [4, 2, 2, 2, 1, 1]
        paddings = [0, 1, 1, 1, 1, 1]

        assert len(kernal_sizes) == self.depth2
        output_dim = output_dims[-1]
        for idx in range(depth2):
            if idx == 4 or idx == 5:
                stage2_list = [
                    depthwise_separable_conv(input_dim, output_dims[idx]),
                    nn.GroupNorm(1, output_dims[idx], eps=1e-6),
                    nn.GELU(),
                ]
            else:
                stage2_list = [
                    nn.Conv2d(
                        input_dim,
                        output_dims[idx],
                        kernel_size=kernal_sizes[idx],
                        stride=strides[idx],
                        padding=paddings[idx],
                        bias=False,
                    ),
                    RMSNorm(output_dims[idx]),
                    nn.GELU(),
                ]
            if idx == depth2 - 1:
                stage2_list.append(nn.Conv2d(output_dims[idx], embed_dim, kernel_size=1))
            stage2 = nn.Sequential(*stage2_list)
            input_dim = output_dims[idx]
            stem2.append(stage2)
        self.proj2 = nn.ModuleList(stem2)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        # stem 1
        x1 = x
        x2 = x 
        for i, stage in enumerate(self.proj1):
            x1 = stage(x1)
            if i == (len(self.proj1) - 1):
                x1 = x1.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x1 = self.norm(x1)
        for i, stage in enumerate(self.proj2):
            x2 = stage(x2)
            if i == (len(self.proj2) - 1):
                x2 = x2.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x2 = self.norm(x2)
        x = x1 + x2
        return x

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

class CasualAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:
        # Apply a causal mask to the attention mechanism
        B, N, C = x.shape
        qkv_bias = False
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
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
                causal=True,
                )

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        else:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True) # -> b, n, h, c

            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x

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

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:
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
        assert cu_slens[-1] == N // self.num_heads, f"error cu_slens for x. {cu_slens[-1]} vs {N}//{self.num_heads}"
        qkv_bias = False
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        # qkv B 4N 3C 
        # because we use token shuffle to expand the sequence length, here we use a different reshape pattern
        qkv = qkv.reshape(B, N//self.num_heads, self.num_heads, 3, self.head_dim).permute(3, 0, 2, 1, 4)  # 3, B, n_head=4, N, C
        
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


class RoPECasualAttentionHeadless(nn.Module):
    '''
        we directly input the multihead feature produced by ts,
        so we do not need split head here
    '''
    fused_attn: Final[bool]

    def __init__(
        self,
        head_dim: int,
        num_heads: int = 8,
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
        self.scale = self.head_dim**-0.5

        if self.head_dim > max_head_dim:
            self.decrease_dim = nn.Linear(self.head_dim, max_head_dim)
            self.increase_dim = nn.Linear(max_head_dim, self.head_dim)
            self.head_dim = head_dim = max_head_dim
        else:
            self.decrease_dim = self.increase_dim = nn.Identity()

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
        qkv_bias = False
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        # qkv B 4N 3C 
        # because we use token shuffle to expand the sequence length, here we use a different reshape pattern
        qkv = qkv.reshape(B, N//self.num_heads, self.num_heads, 3, self.head_dim).permute(3, 0, 2, 1, 4)  # 3, B, n_head=4, N, C
        
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope_theta is not None:
            rope_theta = rope_theta.to(q.dtype) # B, N, num_heads, C
            
            # rope_theta from B, N, num_heads, C to B, num_heads, N, C
            rope_theta = rope_theta.permute(0, 2, 1, 3)

            cos_theta = torch.cos(rope_theta).to(q.device)
            sin_theta = torch.sin(rope_theta).to(q.device)

            # calculate rope2d for q
            x_q, y_q = q.chunk(2, dim=-1)
            x_out_quantized = x_q * cos_theta - y_q * sin_theta
            y_out_quantized = x_q * sin_theta + y_q * cos_theta
            q = torch.cat((x_out_quantized, y_out_quantized), dim=-1)
            
            # calculate rope2d for k
            x_k, y_k = k.chunk(2, dim=-1)
            x_out_quantized = x_k * cos_theta - y_k * sin_theta
            y_out_quantized = x_k * sin_theta + y_k * cos_theta
            k = torch.cat((x_out_quantized, y_out_quantized), dim=-1)


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
                causal=True,
                ) # -> b, n, h, c

            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

        else:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True) # -> b, n, h, c

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

class RoPECausalBlockHeadless(nn.Module):
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
        self.attn = RoPECasualAttentionHeadless(
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


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

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

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens, rope_theta=rope_theta)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CausalBlock(nn.Module):
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
        self.attn = CasualAttention(
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

    def forward(self, x: torch.Tensor, cu_slens=None, rope_theta=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens, rope_theta=rope_theta)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def get_target_size(h, w, target_area=512*512):
    aspect_ratios = [(1, 1), (3, 4), (4, 3), (2, 3), (3, 2), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1)]
    ratio = h / w
    min_diff = float('inf')
    best_ratio = None
    for aspect in aspect_ratios:
        predefined_ratio = aspect[0] / aspect[1]
        diff = abs(predefined_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_ratio = aspect
    # get best ratio
    aspect_h, aspect_w = best_ratio
    aspect_ratio_value = aspect_h / aspect_w

    target_w = int(np.sqrt(target_area / aspect_ratio_value))
    target_h = int(aspect_ratio_value * target_w)

    target_h = max(target_h, 32) // 32 * 32
    target_w = max(target_w, 32) // 32 * 32
    return (target_h, target_w)

def crop_images_and_featuresv2(images, feature_maps_list, patch_size=16, crop_size=(512, 512), max_seq_len=2048):
    """
    对多个图像和其对应的多个特征图进行裁剪，裁剪大小为 crop_size。
    
    images: list of image tensors，形状为 [(1, C, H, W), ...]
    feature_maps_list: list of list of feature maps，形状为 [[(1, H//16, W//16, C), ...], ...]
    crop_size: (crop_h, crop_w)
    
    返回:
        batched_images: Tensor, shape (B, C, crop_h, crop_w)
        batched_feature_maps_per_level: List[Tensor]，每个元素形状为 (B, h, w, C)
        valid_flags_tensor: BoolTensor, shape (B,)
    """
    # raise NotImplementedError(f"We now do not support crop image and its corresponding feature map for diffusion training since the feature is full-attention produced.")
    _crop_h, _crop_w = crop_size
    cropped_images = []
    cropped_feature_maps_per_level = [[] for _ in range(len(feature_maps_list))]
    valid_flags = []
    crop_image_sizes = []
    kept_original_images = []

    num_images = len(images)

    for i in range(num_images):
        image = images[i]
        feature_maps_per_image = [feature_maps_list[level][i] for level in range(len(feature_maps_list))]

        _, _, H, W = image.shape
        _, H_f, W_f, C = feature_maps_per_image[0].shape
        assert H // H_f == patch_size

        crop_h, crop_w = get_target_size(H, W, target_area=_crop_h * _crop_w)

        if (H == W and H < 256):
            # placeholder image and empty image
            continue
        if H < crop_h or W < crop_w:
            # continue
            crop_h = H
            crop_w = W

        # 随机裁剪位置
        x_min_f = random.randint(0, W_f - crop_w // patch_size)
        y_min_f = random.randint(0, H_f - crop_h // patch_size)
        x_max_f = x_min_f + crop_w // patch_size
        y_max_f = y_min_f + crop_h // patch_size

        x_min = x_min_f * patch_size
        y_min = y_min_f * patch_size
        x_max = x_max_f * patch_size
        y_max = y_max_f * patch_size

        # 裁剪图像
        image_crop = image[:, :, y_min:y_max, x_min:x_max]
        cropped_images.append(image_crop)

        # 裁剪每一层的特征图
        for level, fmap in enumerate(feature_maps_per_image):
            fmap_crop = fmap[:, y_min_f:y_max_f, x_min_f:x_max_f]
            fmap_crop = fmap_crop.flatten(1, 2)
            cropped_feature_maps_per_level[level].append(fmap_crop)

        valid_flags.append(1)
        crop_image_sizes.append((crop_h//patch_size, crop_w//patch_size))
        kept_original_images.append(image)


    now_seq_len = 0
    seq_len = []
    final_images = []
    final_image_sizes = []
    final_cropped_feature_maps_per_level = [[] for _ in range(len(feature_maps_list))]
    final_kept_original_images = []
    for i in range(len(cropped_images)):
        feat_seq_len = cropped_feature_maps_per_level[0][i].shape[1]
        if now_seq_len + feat_seq_len < max_seq_len:
            now_seq_len = now_seq_len + feat_seq_len
        else:
            break

        for level in range(len(final_cropped_feature_maps_per_level)):
            final_cropped_feature_maps_per_level[level].append(cropped_feature_maps_per_level[level][i])
        
        final_images.append(cropped_images[i])
        final_image_sizes.append(crop_image_sizes[i])
        final_kept_original_images.append(kept_original_images[i])
        seq_len.append(feat_seq_len)

    try:
        final_cropped_feature_maps_per_level = [
            torch.cat(feature_list, dim=1) for feature_list in final_cropped_feature_maps_per_level
        ]
    except Exception as e:
        print(f"last feat_seq_len {feat_seq_len}, seq_len is {seq_len}, now_seq_len is {now_seq_len}, max_seq_len is {max_seq_len}")
        raise e

    return final_images, final_cropped_feature_maps_per_level, final_image_sizes, seq_len, final_kept_original_images



# LPIPS

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


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