
from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast
import torch.nn.functional as F

import einx
from einops import rearrange, pack, unpack

import random
import numpy as np
import os 
from typing import Union, Optional
try:
    from llava_viq.model.multimodal_encoder.quantizers import FSQ, SimVQ, IBQ
except ImportError:
    from llava_viq._paths import ensure_on_sys_path
    ensure_on_sys_path()
    from llava_viq.model.multimodal_encoder.quantizers import FSQ, SimVQ, IBQ

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def linfnorm(t, dim = -1, eps = 1e-6):
    # L-infinity normalization: project features onto the surface of the unit
    # hypercube so that max_i |t_i| == 1 (||t||_inf == 1). This is the proximal
    # representation used in ViQ stage 2-1 to keep features close to the
    # quantization anchors before discretization.
    return t / t.abs().amax(dim=dim, keepdim=True).clamp_min(eps)

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding='same')
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


class VQPacker(nn.Module):
    def __init__(self, dim, codebook_dim, codebook_size, symmetry_vq=False, limit='none', ts_factor=1):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim
        has_projections = (dim != codebook_dim)
        if limit == 'escape':
            self.project_in = self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(dim, codebook_dim) if has_projections else nn.Identity()
            if symmetry_vq:
                self.project_out = nn.Linear(codebook_dim, dim) if has_projections else nn.Identity()
            else:
                self.project_out = None

        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        self.limit = limit
        self.ts_factor = ts_factor

        if 'fsq' in self.limit:
            self.project_in = self.project_out = nn.Identity()
            levels = [8, 8, 8, 5, 5, 5]
            codebook_size = np.prod(levels)
            self.fsq = FSQ(
                dim = self.codebook_dim, 
                levels = levels,
                symmetry_vq=True,
                ts_factor=self.ts_factor
            )
            self.post_conv = nn.Sequential(*[ResidualBlock(self.dim) for _ in range(2)])
            print(f'build FSQ with levels: {levels}, with codebook_size: {codebook_size}')
        
        if 'simvq' in self.limit:
            # limit should be in format as simvq-<feat_dim>-<other_options>
            self.project_in = self.project_out = nn.Identity()
            quantize_feat_dim = int(self.limit.split('-')[1])
            self.simvq = SimVQ(
                dim = self.dim,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                symmetry_vq=True,
                ts_factor=self.ts_factor
            )
            self.post_conv = nn.Sequential(*[ResidualBlock(self.dim) for _ in range(2)])
            print(f'build SimVQ with feat_dim: {quantize_feat_dim}, codebook_size: {codebook_size}')

        if 'ibq' in self.limit:
            self.project_in = self.project_out = nn.Identity()
            quantize_feat_dim = int(self.limit.split('-')[1])
            self.ibq = IBQ(
                dim = self.dim,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                symmetry_vq=True,
                ts_factor=self.ts_factor,
                skip_quant_prob=0.1
            )   
            self.post_conv = nn.Sequential(*[ResidualBlock(self.dim) for _ in range(2)])       
            print(f'build IBQ with feat_dim: {quantize_feat_dim}, codebook_size: {codebook_size}')  


    def quantize(self, x, slen=None, image_sizes=None):
        # if we need to normalize the vector here?
        if self.limit == 'none' or self.limit == 'escape':
            x = x
            loss = [self.zero] * len(slen) if slen else self.zero
            info = {}
        elif self.limit == 'l2':
            x = l2norm(x)
            loss = [self.zero] * len(slen) if slen else self.zero
            info = {}
        elif self.limit == 'l_infinite':
            x = linfnorm(x)
            loss = [self.zero] * len(slen) if slen else self.zero
            info = {}
        elif self.limit == 'tanh':
            tanh = nn.Tanh()
            x = tanh(x)
            loss = [self.zero] * len(slen) if slen else self.zero
            info = {}
        elif 'fsq' in self.limit:
            x_hat, _, info = self.fsq(x, slen=slen, image_sizes=image_sizes) # fsq has no loss to opti
            if slen is not None:
                loss = [self.zero] * len(slen)
            else:
                loss = self.zero
            x = x_hat                      
        elif 'simvq' in self.limit:
            x_hat, loss, info = self.simvq(x, slen=slen, image_sizes=image_sizes)
            x = x_hat
        elif 'ibq' in self.limit:
            x_hat, loss, info = self.ibq(x, slen=slen, image_sizes=image_sizes)
            x = x_hat
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')
        return x, loss, info

    def forward(self, x, slen=None, image_sizes=None):
        x = self.project_in(x)
        
        x, loss, info = self.quantize(x, slen=slen, image_sizes=image_sizes)

        x_list = x.split(slen, dim=1)
        x_new_list = []
        # forward conv in round 
        for _x, image_size in zip(x_list, image_sizes):
            _x = rearrange(_x, 'b (h w) c -> b c h w', h=image_size[0], w=image_size[1])
            _x = self.post_conv(_x)
            _x = rearrange(_x, 'b c h w -> b (h w) c')
            x_new_list.append(_x)
        x = torch.cat(x_new_list, dim=1)

        if self.project_out is not None:
            x = self.project_out(x)
        return x, loss, info