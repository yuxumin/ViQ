"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack

import os
import random
from timm.layers import Mlp

try:
    from llava_viq.model.multimodal_encoder.layers.model_utils import GoldenGateRoPE2d, RoPEBlockHeadless, RoPECausalBlockHeadless, Block, CausalBlock
except ImportError:
    from llava_viq._paths import ensure_on_sys_path
    ensure_on_sys_path()
    from llava_viq.model.multimodal_encoder.layers.model_utils import GoldenGateRoPE2d, RoPEBlockHeadless, RoPECausalBlockHeadless, Block, CausalBlock



if 'ADD_PRE_ATTN' in os.environ:
    print(f"ADD_PRE_ATTN is set")
    ADD_PRE_ATTN = int(os.environ['ADD_PRE_ATTN'])
else:
    ADD_PRE_ATTN = 0

if 'ADD_POST_CAUSAL_ATTN' in os.environ:
    print(f"ADD_POST_CAUSAL_ATTN is set")
    ADD_POST_CAUSAL_ATTN = int(os.environ['ADD_POST_CAUSAL_ATTN'])
else:
    ADD_POST_CAUSAL_ATTN = 0

if 'ENABLE_ROPE' in os.environ:
    print(f"ENABLE_ROPE is set")
    ENABLE_ROPE = True
else:
    ENABLE_ROPE = False

if 'RETURN_FSQ_INDEX' in os.environ:
    RETURN_FSQ_INDEX = True
else:
    RETURN_FSQ_INDEX = False

if 'FORCE_HEADLESSATTN' in os.environ:
    FORCE_HEADLESSATTN = True
else:
    FORCE_HEADLESSATTN = False
# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

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

# tensor helpers

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    zhat = z.floor()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
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

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
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
        # for adaption to a freeze bn
        if ADD_PRE_ATTN:
            if self.ts_factor != 1 or FORCE_HEADLESSATTN:
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
                    ) for _ in range(ADD_PRE_ATTN)],
                )
            else:
                self.fsq_pre_attn = nn.Sequential(
                    *[Block(
                        dim=self.dim,
                        num_heads=self.dim // 128,
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
                    ) for _ in range(ADD_PRE_ATTN)],
                )

        else:
            self.fsq_pre_attn = None

        if ADD_POST_CAUSAL_ATTN:
            if self.ts_factor != 1:
                self.fsq_post_attn = nn.Sequential(
                    *[RoPECausalBlockHeadless(
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
                    ) for _ in range(ADD_POST_CAUSAL_ATTN)],
                )
            else:
                self.fsq_post_attn = nn.Sequential(
                    *[CausalBlock(
                        dim=self.dim,
                        num_heads=self.dim // 128,
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
                    ) for _ in range(ADD_POST_CAUSAL_ATTN)],
                )
        else:
            self.fsq_post_attn = None 

        if ENABLE_ROPE:
            assert ADD_PRE_ATTN or ADD_POST_CAUSAL_ATTN, "ROPE requires attention"
            if self.ts_factor != 1 or FORCE_HEADLESSATTN:
                rope_dim = self.dim
                if self.dim > max_head_dim:
                    rope_dim = max_head_dim
            else:
                rope_dim = 128 # head dim
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
        return (zhat * self._basis).sum(dim=-1).to(int32)

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
        slen=None,
        image_sizes=None
        ):

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        if slen:
            assert sum(slen) == z.shape[1], f"sum(slen) = {sum(slen)}, but z.shape[1] = {z.shape[1]}"

        if ENABLE_ROPE:
            # prepare rope theta for this sequence
            theta_list = []
            for i in range(len(image_sizes)):
                theta_HWhF = self.rope2d.get_cos_sin_theta(image_sizes[i])
                theta_flattened = theta_HWhF.reshape(-1, theta_HWhF.shape[-2], theta_HWhF.shape[-1]).unsqueeze(0)    # shape (1, H*W, h, F)
                theta_list.append(theta_flattened)
            theta = torch.cat(theta_list, dim=1)
            assert theta.size(1) == z.size(1) // self.ts_factor, f"{theta.size()} vs {z.size()}"
        else:
            theta = None

        if self.fsq_pre_attn:
            slen_for_headless_attn = [length // self.ts_factor for length in slen] 
            cu_indices = [0, ]
            for i in slen_for_headless_attn:
                cu_indices.append(cu_indices[-1] + i)
            cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(z.device)
            for idx, blk in enumerate(self.fsq_pre_attn):
                z = blk(z, cu_slens=cu_slens, rope_theta=theta)

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

            indices = self.codes_to_indices(codes) 
            assert indices.max() < self.codebook_size, f"expected max index to be < {self.codebook_size} but found {indices.max()}"
            
            return indices, theta
    
    def translate_indices_to_feats(self, indices, orig_dtype, slen=None, theta=None): 
        codes = self.implicit_codebook[indices]
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        codes = codes.to(orig_dtype)

        try:
            # analysis the usage precent.
            global_rank = torch.distributed.get_rank()
            if global_rank == 0:
                tensor_in = torch.unique(indices.flatten())
                self.analysis_code_collection.append(tensor_in)
                self.forward_times += 1
                freq = 20
                
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
        except:
            if not RETURN_FSQ_INDEX:
                print(f"get error when try to get codebook usage")

        # project out
        out = self.project_out(codes)

        if self.fsq_post_attn:
            slen_for_headless_attn = [length // self.ts_factor for length in slen] 
            cu_indices = [0, ]
            for i in slen_for_headless_attn:
                cu_indices.append(cu_indices[-1] + i)
            cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(indices.device)
            for idx, blk in enumerate(self.fsq_post_attn):
                out = blk(out, cu_slens=cu_slens, rope_theta=theta)

        # reconstitute image or video dimensions

        return out

    def forward_eval(
            self, 
            z,
            slen=None,
            image_sizes=None
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
            indices, theta = self.get_indices(z, slen, image_sizes)  
            out = self.translate_indices_to_feats(indices, orig_dtype, slen=slen, theta=theta)


        if should_pack:
            out = unpack_one(out, ps, 'b * d')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if self.channel_first:
            out = rearrange(out, 'b ... d -> b d ...')


        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        if self.symmetry_vq:
            assert out.size()[-1] == self.dim

        # return quantized output and indices
        if slen is None:
            return out, self.zero, {"indices": indices}
        else:
            return out, [self.zero] * len(slen), {"indices": indices}

    def forward(
            self, 
            z,
            slen=None,
            image_sizes=None
        ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        if not self.training:
            return self.forward_eval(z, slen, image_sizes)
        else:
            is_img_or_video = z.ndim >= 4
            should_pack = is_img_or_video #default(self.channel_first, is_img_or_video)

            # standardize image or video into (batch, seq, dimension)
            if self.channel_first:
                z = rearrange(z, 'b d ... -> b ... d')
            if should_pack:
                z, ps = pack_one(z, 'b * d')

            assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
            if slen:
                assert sum(slen) == z.shape[1], f"sum(slen) = {sum(slen)}, but z.shape[1] = {z.shape[1]}"

            if ENABLE_ROPE:
                # prepare rope theta for this sequence
                theta_list = []
                for i in range(len(image_sizes)):
                    theta_HWhF = self.rope2d.get_cos_sin_theta(image_sizes[i])
                    theta_flattened = theta_HWhF.reshape(-1, theta_HWhF.shape[-2], theta_HWhF.shape[-1]).unsqueeze(0)    # shape (1, H*W, h, F)
                    theta_list.append(theta_flattened)
                theta = torch.cat(theta_list, dim=1)
                assert theta.size(1) == z.size(1) // self.ts_factor, f"{theta.size()} vs {z.size()}"
            else:
                theta = None

            if self.fsq_pre_attn:
                slen_for_headless_attn = [length // self.ts_factor for length in slen] 
                cu_indices = [0, ]
                for i in slen_for_headless_attn:
                    cu_indices.append(cu_indices[-1] + i)
                cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(z.device)
                for idx, blk in enumerate(self.fsq_pre_attn):
                    z = blk(z, cu_slens=cu_slens, rope_theta=theta)

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

                if self.return_indices:
                    indices = self.codes_to_indices(codes) 
                    assert indices.max() < self.codebook_size, f"expected max index to be < {self.codebook_size} but found {indices.max()}"


                codes = rearrange(codes, 'b n c d -> b n (c d)')

                codes = codes.to(orig_dtype)

            # for eval
            if not self.training:
                assert (self.implicit_codebook[indices] == codes).all()

            try:
                # analysis the usage precent.
                global_rank = torch.distributed.get_rank()
                if global_rank == 0:
                    tensor_in = torch.unique(indices.flatten())
                    self.analysis_code_collection.append(tensor_in)
                    self.forward_times += 1
                    freq = 20
                    
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
            except:
                print(f"get error when try to get codebook usage")

            # project out
            out = self.project_out(codes)

            if self.fsq_post_attn:
                slen_for_headless_attn = [length // self.ts_factor for length in slen] 
                cu_indices = [0, ]
                for i in slen_for_headless_attn:
                    cu_indices.append(cu_indices[-1] + i)
                cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(z.device)
                for idx, blk in enumerate(self.fsq_post_attn):
                    out = blk(out, cu_slens=cu_slens, rope_theta=theta)

            # reconstitute image or video dimensions

            if should_pack:
                out = unpack_one(out, ps, 'b * d')
                indices = maybe(unpack_one)(indices, ps, 'b * c')

            if self.channel_first:
                out = rearrange(out, 'b ... d -> b d ...')


            if not self.keep_num_codebooks_dim and self.return_indices:
                indices = maybe(rearrange)(indices, '... 1 -> ...')

            if self.symmetry_vq:
                assert out.size()[-1] == self.dim

            # return quantized output and indices
            if slen is None:
                return out, self.zero, {"z_in": z_in, "codes": codes, "indices": indices}
            else:
                return out, [self.zero] * len(slen), {"z_in": z_in, "codes": codes, "indices": indices}