from __future__ import annotations
from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import rearrange, pack, unpack


def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def linfnorm(t, dim = -1, eps = 1e-6):
    # L-infinity normalization: project features onto the surface of the unit
    # hypercube so that max_i |t_i| == 1 (||t||_inf == 1). This is the proximal
    # representation used in ViQ stage 2-1 to keep features close to the
    # quantization anchors before discretization.
    return t / t.abs().amax(dim=dim, keepdim=True).clamp_min(eps)

def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )

def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)


def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse

# class

class SimVQ(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim,
        codebook_transform: Module | None = None,
        init_fn: Callable = identity,
        channel_first = False,
        rotation_trick = True,  # works even better with rotation trick turned on, with no straight through and the commit loss from input to quantize
        input_to_quantize_commit_loss_weight = 0.25,
        commitment_weight = 1.,
        frozen_codebook_dim = None, # frozen codebook dim could have different dimensions than projection
        symmetry_vq=False,
        limit = 'none',
        ts_factor=1
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first
        self.limit = limit

        frozen_codebook_dim = default(frozen_codebook_dim, codebook_dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (frozen_codebook_dim ** -0.5)
        codebook = init_fn(codebook)

        # the codebook is actually implicit from a linear layer from frozen gaussian or uniform


        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, codebook_dim, bias = False)

        has_projections = (dim != codebook_dim)
        self.project_in = nn.Linear(dim, codebook_dim) if has_projections else nn.Identity()
        self.dim = dim
        if symmetry_vq:
            self.project_out = nn.Linear(codebook_dim, dim) if has_projections else nn.Identity()
        else:
            self.project_out = None


        self.code_transform = codebook_transform

        self.register_buffer('frozen_codebook', codebook, persistent = True)
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        # self.frozen_codebook = nn.Parameter(codebook, requires_grad=False)

        # whether to use rotation trick from Fifty et al. 
        # https://arxiv.org/abs/2410.06424

        self.rotation_trick = rotation_trick

        # commit loss weighting - weighing input to quantize a bit less is crucial for it to work

        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight

        # total commitment loss weight

        self.commitment_weight = commitment_weight
        
        self.forward_times = 0
        self.analysis_code_collection = []
        self.history_code_collection = []

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def get_codebook_entry(
        self,
        indices
    ):
        implicit_codebook = self.codebook

        frozen_codes = get_at('[c] d, b ... -> b ... d', self.frozen_codebook, indices)
        quantized = self.code_transform(frozen_codes)

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')

        # to align with fake quantizer
        if self.limit == 'none':
            quantized = quantized
        elif self.limit == 'l2':
            quantized = l2norm(quantized)
        elif self.limit == 'l_infinite':
            quantized = linfnorm(quantized)
        elif self.limit == 'tanh':
            tanh = nn.Tanh()
            quantized = tanh(quantized)
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')

        return quantized

    def forward(
        self,
        x,
        slen=None,
        image_sizes=None
    ):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        x, inverse_pack = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        implicit_codebook = self.codebook

        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)

        # select codes

        quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)

        # analysis the usage precent.
        global_rank = torch.distributed.get_rank()
        if global_rank == 0:
            tensor_in = torch.unique(indices.flatten())
            self.analysis_code_collection.append(tensor_in)
            self.forward_times += 1
            freq = 20 if slen is not None else 20*64

            if self.forward_times % freq == 0:
                used_code = torch.unique(torch.cat(self.analysis_code_collection, dim=0))
                num_unique = used_code.numel()
                self.analysis_code_collection = []
                print(f'\n\nRank {global_rank} - Round{self.forward_times}: {freq} times forward Usage is {num_unique}/{self.codebook_size} = {num_unique / self.codebook_size:.2%}')
                
                self.history_code_collection.append(used_code)
                if len(self.history_code_collection) > 20:
                    self.history_code_collectio = self.history_code_collection[-20:]
                used_code = torch.unique(torch.cat(self.history_code_collection, dim=0))
                num_unique = used_code.numel()
                print(f'Rank {global_rank} - Round{self.forward_times}: History {20 * freq } Usage is {num_unique}/{self.codebook_size} = {num_unique / self.codebook_size:.2%} \n\n ')

        if self.training:
            # commit loss and straight through, as was done in the paper
            if slen is not None:
                xs = x.split(slen, dim=1)
                quantizeds = quantized.split(slen, dim=1)
                commit_loss_list = []
                for _x, _q in zip(xs, quantizeds):
                    _commit_loss = (
                        F.mse_loss(_x.detach(), _q) +
                        F.mse_loss(_x, _q.detach()) * self.input_to_quantize_commit_loss_weight
                    ) * self.commitment_weight
                    commit_loss_list.append(_commit_loss)
                commit_loss = commit_loss_list
            else:
                commit_loss = (
                    F.mse_loss(x.detach(), quantized) +
                    F.mse_loss(x, quantized.detach()) * self.input_to_quantize_commit_loss_weight
                ) * self.commitment_weight

            if self.rotation_trick:
                # rotation trick from @cfifty
                prev_quantized = quantized
                quantized = rotate_to(x, quantized)
            else:
                quantized = (quantized - x).detach() + x
        else:
            print('use eval mode for simvq')
            commit_loss = self.zero
            quantized = quantized.contiguous()

        # to align with fake quantizer
        if self.limit == 'none':
            quantized = quantized
        elif self.limit == 'l2':
            quantized = l2norm(quantized)
        elif self.limit == 'l_infinite':
            quantized = linfnorm(quantized)
        elif self.limit == 'tanh':
            tanh = nn.Tanh()
            quantized = tanh(quantized)
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')

        if self.project_out is not None:
            quantized = self.project_out(quantized)
            assert quantized.size()[-1] == self.dim

        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d-> b d ...')

        return quantized, commit_loss , {"indices": indices}
