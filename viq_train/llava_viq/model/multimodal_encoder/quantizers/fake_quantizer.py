
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
    from llava_viq.model.multimodal_encoder.quantizers import FSQ
except ImportError:
    from llava_viq._paths import ensure_on_sys_path
    ensure_on_sys_path()
    from llava_viq.model.multimodal_encoder.quantizers import FSQ

if 'DISABLE_BN_FSQ_LOSS' in os.environ:
    print(f"DISABLE_BN_FSQ_LOSS is set")
    DISABLE_BN_FSQ_LOSS = True
else:
    DISABLE_BN_FSQ_LOSS = False

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slightly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )


    def kl_slen(self, other: "DiagonalGaussianDistribution" = None, slen=None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                if slen:
                    mean_list = self.mean.split(slen, dim=-1) # B C N
                    var_list = self.var.split(slen, dim=-1) # B C N
                    logvar_list = self.logvar.split(slen, dim=-1) # B C N
                    loss_list = []
                    for mean, var, logvar in zip(mean_list, var_list, logvar_list):
                        _loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)
                        loss_list.append(_loss * 1e-6)
                    return loss_list
                else:
                    return 0.5 * torch.sum(
                        torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                        dim=[1, 2, 3],
                    )
            else:
                raise NotImplementedError()
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def linfnorm(t, dim = -1, eps = 1e-6):
    # L-infinity normalization: project features onto the surface of the unit
    # hypercube so that max_i |t_i| == 1 (||t||_inf == 1). This is the proximal
    # representation used in ViQ stage 2-1 to keep features close to the
    # quantization anchors before discretization.
    return t / t.abs().amax(dim=dim, keepdim=True).clamp_min(eps)

class FakeQuantizer(nn.Module):
    def __init__(self, dim, codebook_dim, symmetry_vq=False, limit='none', ts_factor=1):
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
            levels = [8, 8, 8, 5, 5, 5]
            codebook_size = np.prod(levels)
            self.fsq = FSQ(
                dim = self.codebook_dim, 
                levels = levels,
                symmetry_vq=True,
                ts_factor=self.ts_factor
            )
        if 'vae' in self.limit:
            latent_width = int(self.limit.split('-')[-1])
            
            self.vae_project_in = nn.Linear(codebook_dim, 2 * latent_width)
            self.vae_project_out = nn.Linear(latent_width, codebook_dim)


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
                if DISABLE_BN_FSQ_LOSS:
                    loss = [self.zero] * len(slen)
                else:
                    xs = x.split(slen, dim=1)
                    x_hats = x_hat.split(slen, dim=1)
                    loss_list = []
                    for _x, _q in zip(xs, x_hats):
                        _loss = F.mse_loss(_x.detach(), _q)
                        loss_list.append(_loss)
                    loss = loss_list
            else:
                loss = self.zero
            x = x_hat

            if 'l2' in self.limit:
                x = l2norm(x)
                      
        elif 'vae' in self.limit:
            vae_latent = self.vae_project_in(x).permute(0, 2, 1) # B C N 
            posterior = DiagonalGaussianDistribution(vae_latent)
            if self.training:
                vae_latent = posterior.sample()
            else:
                vae_latent = posterior.mode()
            x = self.vae_project_out(vae_latent.permute(0, 2, 1)) # B N C
            loss = posterior.kl_slen(slen=slen)
            info = {}
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')
        return x, loss, info

    def forward(self, x, slen=None, image_sizes=None):
        x = self.project_in(x)
        x, loss, info = self.quantize(x, slen=slen, image_sizes=image_sizes)
        if self.project_out is not None:
            x = self.project_out(x)
        return x, loss, info