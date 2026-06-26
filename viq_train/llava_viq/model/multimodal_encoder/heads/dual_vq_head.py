"""DualVQHead: the dual-branch vector-quantization head for the viq encoder.

Split out of ``siglip_vit_anyres_viq`` to keep the encoder file manageable.
"""
import math
from functools import partial
from typing import Optional, Type
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from timm.layers import Mlp, LayerType
except Exception:
    Mlp = None
    LayerType = None

from ..layers._common import ResualMLP
from ..layers.model_utils import Block, CausalBlock
from ..envir_defines import *

try:
    from llava_viq.model.multimodal_encoder.quantizers import SimVQ, FakeQuantizer, IBQ, FSQ, VQPacker
except ImportError:
    from llava_viq._paths import ensure_on_sys_path
    ensure_on_sys_path()
    from llava_viq.model.multimodal_encoder.quantizers import SimVQ, FakeQuantizer, IBQ, FSQ, VQPacker


class DualVQHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            vq_low_type, 
            vq_low_size, 
            vq_low_limit, 
            vq_high_type, 
            vq_high_size, 
            vq_high_limit,
            vq_high_enable_token_shuffle,
            vq_high_enable_layernorm_trick,
            vq_low_enable_token_shuffle,
            vq_low_enable_layernorm_trick,
            vq_low_preprocess_type=None,
            vq_high_preprocess_type=None,
            vq_low_postprocess_type=None,
            vq_high_postprocess_type=None,
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
            use_first_code_for_rec: bool = False,
        ):
        super().__init__()

        def _blocks(dim, depth, block_cls=Block):
            # Shared Block/CausalBlock factory: all calls use identical kwargs
            # except dim / depth / block class. Returns a list for nn.Sequential(*...).
            return [block_cls(
                dim=dim,
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
            ) for _ in range(depth)]

        def _mlp(in_f, hidden_f, out_f, act_layer=nn.GELU):
            return Mlp(in_features=in_f, hidden_features=hidden_f,
                       out_features=out_f, act_layer=act_layer)

        self.num_features = input_dim
        if 'attn-large' in vq_low_preprocess_type:
            self.low_branch_dim_factor = 2
        else:
            self.low_branch_dim_factor = 1

        self.use_first_code_for_rec = use_first_code_for_rec

        self.ts_factor = 4 if vq_low_enable_token_shuffle else 1
        self.vq_low, self.vq_low_features = self.build_quantizer(vq_low_type, vq_low_size, vq_low_limit, channel_factor=self.low_branch_dim_factor)
        self.vq_high, self.vq_high_features = self.build_quantizer(vq_high_type, vq_high_size, vq_high_limit, channel_factor=1)
        print(f'Build DualVQHead, with {vq_low_type}:{vq_low_size}:{vq_low_limit} + {vq_high_type}:{vq_high_size}:{vq_high_limit}')

        self.grad_checkpointing = False
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        self.vq_high_enable_token_shuffle = vq_high_enable_token_shuffle
        if self.vq_high_enable_token_shuffle:
            self.vq_high_pre_token_shuffle_head = _mlp(self.num_features, 4 * self.num_features, 4 * self.num_features)
            self.vq_high_post_token_shuffle_head = _mlp(self.num_features, self.num_features, self.num_features)
            if vq_high_postprocess_type != 'none':
                self.vq_high_inverse_token_shuffle_head = _mlp(4 * self.vq_high_features, 4 * self.vq_high_features, 4 * self.vq_high_features)
            else:
                self.vq_high_inverse_token_shuffle_head = None

        vq_low_postprocess_dim = self.vq_low_features

        self.vq_low_enable_token_shuffle = vq_low_enable_token_shuffle
        if self.vq_low_enable_token_shuffle:
            self.vq_low_pre_token_shuffle_head = _mlp(self.low_branch_dim_factor * self.num_features, 4 * self.low_branch_dim_factor * self.num_features, 4 * self.low_branch_dim_factor * self.num_features)
            self.vq_low_post_token_shuffle_head = _mlp(self.low_branch_dim_factor * self.num_features, self.low_branch_dim_factor * self.num_features, self.low_branch_dim_factor * self.num_features)
            if vq_low_postprocess_type != 'none' and 'nomerge' not in vq_low_preprocess_type:
                # vq_low_postprocess_type = attn-downsample2x-nomerge
                # we do not merge the code
                if SYMMETRY_VQ:
                    self.vq_low_inverse_token_shuffle_head = _mlp(4 * self.vq_low_features, 4 * self.vq_low_features, self.vq_low_features)
                    assert self.vq_low_features == self.low_branch_dim_factor * self.num_features, f'The design for vq system is not symmetric'
                    vq_low_postprocess_dim = self.vq_low_features
                else:
                    self.vq_low_inverse_token_shuffle_head = _mlp(4 * self.vq_low_features, 4 * self.vq_low_features, 4 * self.vq_low_features)
                    vq_low_postprocess_dim = 4 * self.vq_low_features
                self.low_expand_method = '1d'
            else:
                # 不用考虑de shuffle，因为有deshuffle的时候只能是1d
                self.vq_low_inverse_token_shuffle_head = None
                self.low_expand_method = '2d'


        if self.use_first_code_for_rec:
            self.first_code_for_rec_fc = _mlp(self.vq_low_features, self.vq_low_features, self.vq_low_features)

        self.vq_low_enable_layernorm_trick = vq_low_enable_layernorm_trick
        if self.vq_low_enable_layernorm_trick:
            self.vq_low_norm_trick_pre_quantize_norm_wo_affine = nn.LayerNorm(self.low_branch_dim_factor * self.num_features, elementwise_affine=False)
            self.vq_low_norm_trick_post_quantize_affine_weight = nn.Parameter(torch.ones(2 * self.num_features if 'attn-large' in vq_low_postprocess_type else self.num_features))
            self.vq_low_norm_trick_post_quantize_affine_bias = nn.Parameter(torch.zeros(2 * self.num_features if 'attn-large' in vq_low_postprocess_type else self.num_features))

        self.vq_high_enable_layernorm_trick = vq_high_enable_layernorm_trick
        if self.vq_high_enable_layernorm_trick:
            self.vq_high_norm_trick_pre_quantize_norm_wo_affine = nn.LayerNorm(self.num_features, elementwise_affine=False)
            self.vq_high_norm_trick_post_quantize_affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.vq_high_norm_trick_post_quantize_affine_bias = nn.Parameter(torch.zeros(self.num_features))


        if vq_low_preprocess_type == 'attn':
            self.vq_low_preprocess_layers = nn.Sequential(
                    *_blocks(self.num_features, 3),
                )
        elif 'attn-downsample2x' in vq_low_preprocess_type:
            # attn-downsample2x
            # attn-downsample2x-nomerge
            self.vq_low_preprocess_layers = nn.Sequential(
                    *_blocks(self.num_features, 3),
                )
        elif vq_low_preprocess_type == 'attn-large-shallow':
            self.vq_low_preprocess_layers = nn.Sequential(
                *_blocks(2*self.num_features, 3),
            )
        elif vq_low_preprocess_type == 'attn-large':
            self.vq_low_preprocess_layers = nn.Sequential(
                *_blocks(2*self.num_features, 6),
            )
        else:
            self.vq_low_preprocess_layers = nn.Identity()


        if 'downsample2x' in vq_low_preprocess_type:
            self.downsample_attn = nn.Sequential(
                *_blocks(self.low_branch_dim_factor*self.num_features, 1)
            )
            self.predictor = nn.Sequential(
                nn.Linear(self.low_branch_dim_factor*self.num_features*2, self.low_branch_dim_factor*self.num_features),
                nn.GELU(),
                nn.Linear(self.low_branch_dim_factor*self.num_features, self.low_branch_dim_factor*self.num_features),
            )
            self.downsample_ratio = 2
        else:
            self.downsample_attn = None
            self.predictor = None
            self.downsample_ratio = None

        if vq_high_preprocess_type == 'attn':
            self.vq_high_preprocess_layers = nn.Sequential(
                    *_blocks(self.num_features, 3),
                )
        else:
            self.vq_high_preprocess_layers = nn.Identity()
        

        self.vq_low_aux_preprocess_layers = None
        self.vq_low_aux_postprocess_layers = None


        if vq_low_postprocess_type == 'mlp':
            self.vq_low_postprocess_layers = _mlp(vq_low_postprocess_dim, self.num_features, self.num_features)
        elif vq_low_postprocess_type in ('resualmlp', 'resualmlp-light'):
            depth = 3 if vq_low_postprocess_type == 'resualmlp-light' else 9
            self.vq_low_postprocess_layers = nn.Sequential(
                    _mlp(vq_low_postprocess_dim, self.num_features, self.num_features),
                    *[ResualMLP(in_features=self.num_features, hidden_features=2 * self.num_features, out_features=self.num_features) for _ in range(depth)]
                )
        elif vq_low_postprocess_type == 'casual_attn':
            self.vq_low_postprocess_layers = nn.Sequential(
                _mlp(vq_low_postprocess_dim, self.num_features, self.num_features),
                norm_layer(self.num_features),
                *_blocks(self.num_features, 3, CausalBlock),
            )
        elif vq_low_postprocess_type == 'casual_attn-wide':
            self.vq_low_postprocess_layers = nn.Sequential(
                _mlp(vq_low_postprocess_dim, 2 * self.num_features, 2 * self.num_features),
                norm_layer(2 * self.num_features),
                *_blocks(2 * self.num_features, 3, CausalBlock),
                norm_layer(2 * self.num_features),
                _mlp(2 * self.num_features, self.num_features, self.num_features)
            )
        elif vq_low_postprocess_type == 'attn':
            self.vq_low_postprocess_layers = nn.Sequential(
                _mlp(vq_low_postprocess_dim, self.num_features, self.num_features),
                norm_layer(self.num_features),
                *_blocks(self.num_features, 3)
            )
        elif vq_low_postprocess_type == 'attn-large':
            self.vq_low_postprocess_layers = nn.Sequential(
                _mlp(vq_low_postprocess_dim, 2 * self.num_features, 2 * self.num_features),
                norm_layer(2 * self.num_features),
                *_blocks(2*self.num_features, 6)
            )
        elif vq_low_postprocess_type == 'attn-deep':
            self.vq_low_postprocess_layers = nn.Sequential(
                _mlp(vq_low_postprocess_dim, self.num_features, self.num_features),
                norm_layer( self.num_features),
                *_blocks(self.num_features, 6)
            )
        else:
            self.vq_low_postprocess_layers = nn.Identity()

        if vq_high_postprocess_type == 'mlp':
            self.vq_high_postprocess_layers = _mlp(4 * self.vq_high_features if vq_high_enable_token_shuffle else self.vq_high_features, self.num_features, self.num_features)
        elif vq_high_postprocess_type == 'casual_attn':
            self.vq_high_postprocess_layers = nn.Sequential(
                _mlp(4 * self.vq_high_features if vq_high_enable_token_shuffle else self.vq_high_features, self.num_features, self.num_features),
                norm_layer(self.num_features),
                *_blocks(self.num_features, 3, CausalBlock)
            )
        elif vq_high_postprocess_type == 'attn':
            self.vq_high_postprocess_layers = nn.Sequential(
                _mlp(4 * self.vq_high_features if vq_high_enable_token_shuffle else self.vq_high_features, self.num_features, self.num_features),
                norm_layer(self.num_features),
                *_blocks(self.num_features, 3)
            )
        else:
            self.vq_high_postprocess_layers = nn.Identity()

        self.one_time_debug_info = True
        self.close_low_branch = False
        self.close_high_branch = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def build_quantizer(self, vq_type, vq_size, vq_limit, channel_factor=1):
        if vq_type == 'simvq':
            assert vq_limit != 'escape', f'simvq does not support escape limit'
            print(f"use SimVQ quantizer")
            quantize_feat_dim = 32
            codebook_size = vq_size
            quantize = SimVQ(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor
            )
        elif vq_type == 'simvq-wide':
            assert vq_limit != 'escape', f'simvq does not support escape limit'
            print(f"use SimVQ-WIDE quantizer")
            quantize_feat_dim = 512
            codebook_size = vq_size
            quantize = SimVQ(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'simvq-middle':
            assert vq_limit != 'escape', f'simvq does not support escape limit'
            print(f"use SimVQ-MIDDLE quantizer")
            quantize_feat_dim = 128
            codebook_size = vq_size
            quantize = SimVQ(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'simvq-normal':
            assert vq_limit != 'escape', f'simvq does not support escape limit'
            print(f"use SimVQ-NORMAL quantizer")
            quantize_feat_dim = 256
            codebook_size = vq_size
            quantize = SimVQ(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                codebook_size = codebook_size,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'fsq':
            print(f"use FSQ quantizer")
            if FSQ16K:
                levels = [8, 8, 8, 6, 5]
            elif FSQ16KV2:
                levels = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            elif FSQ8K:
                levels = [8, 8, 8, 4, 4]
            elif FSQ4K:
                levels = [8, 8, 4, 4, 4]
            elif FSQ2K:
                levels = [8, 8, 4, 3, 3]
            else:
                levels = [8, 8, 8, 5, 5, 5]
            print(f"FSQ levels: {levels}")
            quantize_feat_dim = len(levels)
            codebook_size = np.prod(levels)
            quantize = FSQ(
                dim = self.num_features * channel_factor, 
                levels = levels,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'packer':
            print(f"use Packer quantizer with limit {vq_limit}")
            quantize_feat_dim = 32
            codebook_size = vq_size
            quantize = VQPacker(
                dim = self.num_features * channel_factor,
                codebook_dim = self.num_features * channel_factor, # no projections
                codebook_size = codebook_size,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'fake':
            print(f"use Fake quantizer")
            quantize_feat_dim = 32
            quantize = FakeQuantizer(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'fake-wide':
            print(f"use Fake-WIDE quantizer")
            quantize_feat_dim = 512
            quantize = FakeQuantizer(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        elif vq_type == 'fake-normal':
            print(f"use Fake-NORMAL quantizer")
            quantize_feat_dim = 256
            quantize = FakeQuantizer(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor
            )
        elif vq_type == 'fake-middle':
            print(f"use Fake-MIDDLE quantizer")
            quantize_feat_dim = 128
            quantize = FakeQuantizer(
                dim = self.num_features * channel_factor,
                codebook_dim = quantize_feat_dim,
                limit = vq_limit,
                symmetry_vq=SYMMETRY_VQ,
                ts_factor=self.ts_factor 
            )
        else:
            raise NotImplementedError(f"unsupported vq method {vq_type}")
    
        if vq_limit == 'escape' or SYMMETRY_VQ:
            vq_dim = self.num_features * channel_factor
        else:
            vq_dim = quantize_feat_dim

        return quantize, vq_dim
    
    def _forward_low_branc_infer(self, x_low, oryx_x, image_sizes, num_prefix_tokens, slen=None, cu_slens=None, return_feat_rec_loss=False):
        ####################################################################
        # --------     Branch reconstruction: vq_low, x_low     --------   # 
        ####################################################################
        if slen is None and cu_slens is None:
            batchify = True
            raise
        else:
            batchify = False
        
        if self.vq_low_enable_layernorm_trick:
            x_low = self.vq_low_norm_trick_pre_quantize_norm_wo_affine(x_low)

        if isinstance(self.vq_low_preprocess_layers, torch.nn.Sequential):
            if batchify:
                # x is batchify
                for idx, blk in enumerate(self.vq_low_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_low = checkpoint(blk, x_low, use_reentrant=True)
                    else:
                        x_low = blk(x_low)
            else:
                for idx, blk in enumerate(self.vq_low_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                    else:
                        x_low = blk(x_low, cu_slens=cu_slens)
        else:
            x_low = self.vq_low_preprocess_layers(x_low)

        # aux layer
        if self.vq_low_aux_preprocess_layers is not None:
            if isinstance(self.vq_low_aux_preprocess_layers, torch.nn.Sequential):
                if batchify:
                    # x is batchify
                    for idx, blk in enumerate(self.vq_low_aux_preprocess_layers):
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_low = checkpoint(blk, x_low, use_reentrant=True)
                        else:
                            x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.vq_low_aux_preprocess_layers):
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                        else:
                            x_low = blk(x_low, cu_slens=cu_slens)
            else:
                x_low = self.vq_low_aux_preprocess_layers(x_low)


        # Token Shuffle
        if self.vq_low_enable_token_shuffle:
            if num_prefix_tokens == 0:
                B, N, C = x_low.shape
                x_low = self.vq_low_pre_token_shuffle_head(x_low) # 1 N C  -> 1 N 4C
                # for casual infer use
                x_low = x_low.view(B, 4 * N, C) # 1 N 4C -> 1 4N C
                x_low = self.vq_low_post_token_shuffle_head(x_low) #  1 4N C ->  1 4N C
            else:
                raise NotImplementedError(f"I do not if it is correct to process cls token as below.")

        # VQ One
        if batchify:
            xs = x_low.split(1, dim=0)
            vq_x = torch.cat(xs, dim=1)
            vq_slen = [xs.shape[1]] * len(xs)
        else:
            xs = x_low.split([length * 4 for length in slen] if self.vq_low_enable_token_shuffle else slen, dim=1)
            vq_x = torch.cat(xs, dim=1)
            vq_slen = [length * 4 for length in slen] if self.vq_low_enable_token_shuffle else slen

        ######### no for circle
        x_low, _, _ = self.vq_low(vq_x, slen=vq_slen, image_sizes=image_sizes)
        if batchify:
            xs = x_low.split(vq_slen, dim=1)
            x_low = torch.cat(xs, dim=0)

        #########  for circle
        #     _x, commit_loss, _ = self.vq_low(_x)

            
        return x_low

    def _forward_low_branch(self, x_low, oryx_x, image_sizes, num_prefix_tokens, slen=None, cu_slens=None, return_feat_rec_loss=False):
        ####################################################################
        # --------     Branch reconstruction: vq_low, x_low     --------   # 
        ####################################################################
        if slen is None and cu_slens is None:
            batchify = True
            raise
        else:
            batchify = False

        common_factor = 1
        
        feature_lookup = {
            'oryx': {'seq_feature': oryx_x, 'shape_factor': 1 * common_factor}
        }


        if self.vq_low_enable_layernorm_trick:
            x_low = self.vq_low_norm_trick_pre_quantize_norm_wo_affine(x_low)

        anchor_x = x_low

        if isinstance(self.vq_low_preprocess_layers, torch.nn.Sequential):
            if batchify:
                # x is batchify
                for idx, blk in enumerate(self.vq_low_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_low = checkpoint(blk, x_low, use_reentrant=True)
                    else:
                        x_low = blk(x_low)
            else:
                for idx, blk in enumerate(self.vq_low_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                    else:
                        x_low = blk(x_low, cu_slens=cu_slens)
        else:
            x_low = self.vq_low_preprocess_layers(x_low)

        escape_feature = x_low

        # aux layer
        if self.vq_low_aux_preprocess_layers is not None:
            raise NotImplementedError("Aux layers for VQ low not implemented yet")
            if isinstance(self.vq_low_aux_preprocess_layers, torch.nn.Sequential):
                if batchify:
                    # x is batchify
                    for idx, blk in enumerate(self.vq_low_aux_preprocess_layers):
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_low = checkpoint(blk, x_low, use_reentrant=True)
                        else:
                            x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.vq_low_aux_preprocess_layers):
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                        else:
                            x_low = blk(x_low, cu_slens=cu_slens)
            else:
                x_low = self.vq_low_aux_preprocess_layers(x_low)


        if self.downsample_attn is not None:
            x_low_list = x_low.split(slen, dim=1)
            x_low_list_new = []
            new_image_sizes = []
            for _x, _size in zip(x_low_list, image_sizes):
                H, W = _size
                B, C = _x.shape[0], _x.shape[-1]
                assert B == 1
                assert H % self.downsample_ratio == 0
                assert W % self.downsample_ratio == 0
                _x = _x.reshape(
                    B, 
                    H//self.downsample_ratio, 
                    self.downsample_ratio,
                    W//self.downsample_ratio, 
                    self.downsample_ratio, 
                    C
                ).permute(0, 1, 3, 2, 4, 5).reshape(
                    B, 
                    H//self.downsample_ratio, 
                    W//self.downsample_ratio, 
                    self.downsample_ratio * self.downsample_ratio, 
                    C
                )
                _x = _x.reshape(B * H//self.downsample_ratio * W//self.downsample_ratio, self.downsample_ratio * self.downsample_ratio, C)
                x_low_list_new.append(_x)
                new_image_sizes.append((H//self.downsample_ratio, W//self.downsample_ratio))
            new_batchify_slen = [xi.size(0) for xi in x_low_list_new]
            new_x_low = torch.cat(x_low_list_new, dim=0)

            new_x_low = self.downsample_attn(new_x_low) # B' 4 C -> B' 4 C

            pooled_new_x_low = new_x_low.mean(-2, keepdim=True).expand(-1, 4, -1) # B' 4 C -> B' 1 C -> B' 4 C
            fused_new_x = torch.cat([new_x_low, pooled_new_x_low], dim=-1) # B' 4 C -> B' 4 2C 
            score = self.predictor(fused_new_x) # B' 4 2C -> B' 4 C 
            normalized_score = F.softmax(score, dim=-2)

            new_x_low = (new_x_low * normalized_score).sum(dim=-2)  # B' 4 C -> B' 4 C -> B' C 

            x_low = new_x_low.unsqueeze(0) # B' C -> 1 B' C
            slen = new_batchify_slen
            d = torch.nn.functional.pad(torch.cumsum(torch.tensor(slen), dim=0), (1, 0)).to(torch.int32).to(x_low.device)
            image_sizes = new_image_sizes
            common_factor *= 0.5


        # Token Shuffle
        if self.vq_low_enable_token_shuffle:
            if num_prefix_tokens == 0:
                B, N, C = x_low.shape
                x_low = self.vq_low_pre_token_shuffle_head(x_low) # 1 N C  -> 1 N 4C
                # for casual infer use
                if self.low_expand_method == '1d':
                    x_low = x_low.reshape(B, N, 4, C).view(B, 4 * N, C) # 1 N 4C -> 1 4N C
                elif self.low_expand_method == '2d':
                    x_low_list = x_low.split(slen, dim=1)
                    x_low_list_new = []
                    for _x, _size in zip(x_low_list, image_sizes):
                        _x = _x.reshape(_x.shape[0], _size[0], _size[1], 2, 2, -1) # 1 h w 4 C
                        _x = _x.permute(0, 1, 3, 2, 4, 5) # 1 h 2 w 2 c
                        _x = _x.reshape(_x.shape[0], _size[0] * _size[1] * 2 * 2, -1) # 1 4N C
                        x_low_list_new.append(_x)
                    x_low = torch.cat(x_low_list_new, dim=1) # 1 sum_4N C
                else:
                    raise
                x_low = self.vq_low_post_token_shuffle_head(x_low) #  1 4N C ->  1 4N C
            else:
                raise NotImplementedError(f"I do not if it is correct to process cls token as below.")
            common_factor *= 2


        feature_lookup.update({'after_ts':{'seq_feature': x_low, 'shape_factor': 1 * common_factor}})

        # VQ One
        if batchify:
            xs = x_low.split(1, dim=0)
            vq_x = torch.cat(xs, dim=1)
            vq_slen = [xs[0].shape[1]] * len(xs)
        else:
            xs = x_low.split([length * 4 for length in slen] if self.vq_low_enable_token_shuffle else slen, dim=1)
            vq_x = torch.cat(xs, dim=1)
            vq_slen = [length * 4 for length in slen] if self.vq_low_enable_token_shuffle else slen

        ######### no for circle
        x_low, commit_loss_low_list, info = self.vq_low(vq_x, slen=vq_slen, image_sizes=image_sizes)
        if batchify:
            xs = x_low.split(vq_slen, dim=1)
            x_low = torch.cat(xs, dim=0)

        #########  for circle
        #     _x, commit_loss, _ = self.vq_low(_x)

            
        if self.use_first_code_for_rec:
            pass
            assert self.low_expand_method == '1d' and num_prefix_tokens == 0
            B, N, C = x_low.shape
            x_low_first_code = self.first_code_for_rec_fc(x_low.view(B, N//4, 4, C)[:, :, 0]) # B N/4 C

        else:
            x_low_first_code = None

        
        feature_lookup.update({'after_quant_8x':{'seq_feature': x_low, 'shape_factor': 1 * common_factor}})

        # Token Shuffle
        if self.vq_low_enable_token_shuffle and self.vq_low_inverse_token_shuffle_head is not None:
            if num_prefix_tokens == 0:
                assert self.low_expand_method == '1d'
                B, N, C = x_low.shape
                x_low = x_low.view(B, N//4, 4 * C) 
                x_low = self.vq_low_inverse_token_shuffle_head(x_low)
            else:
                raise NotImplementedError(f"I do not if it is correct to process cls token as below.")
            common_factor *= 0.5


        feature_lookup.update({'after_quant_16x':{'seq_feature': x_low, 'shape_factor': 1 * common_factor}})

        # aux layer
        if self.vq_low_aux_postprocess_layers is not None:
            raise NotImplementedError("Aux layers for VQ low not implemented yet")
            if isinstance(self.vq_low_aux_postprocess_layers, torch.nn.Sequential):
                if batchify:
                    # x is batchify
                    for idx, blk in enumerate(self.vq_low_aux_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, use_reentrant=True)
                            else:
                                x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.vq_low_aux_postprocess_layers):                        
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                            else:
                                x_low = blk(x_low, cu_slens=cu_slens)
            else:
                x_low = self.vq_low_aux_postprocess_layers(x_low)

        x_ae = x_low
        if ADD_AUX_LAYERS:
            raise NotImplementedError("Aux layers for VQ low not implemented yet")
            xs_target = x_ae.split(slen, dim=1)
            xs_oryx = escape_feature.split(slen, dim=1)

            ae_loss_list = []
            for _x, _y in zip(xs_target, xs_oryx):
                ae_loss = F.mse_loss(_y.detach(), _x)
                ae_loss_list.append(ae_loss)

        else:
            ae_loss_list = [self.zero] * len(slen)


        # original layer
        if self.vq_low_postprocess_layers is not None:
            if isinstance(self.vq_low_postprocess_layers, torch.nn.Sequential):
                if batchify:
                    raise NotImplementedError("Batchify mode for VQ low postprocess layers not implemented yet")
                    for idx, blk in enumerate(self.vq_low_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, use_reentrant=True)
                            else:
                                x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.vq_low_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            if self.use_first_code_for_rec:
                                _temp_x = torch.cat([x_low, x_low_first_code], dim=1)
                                _temp_x = blk(_temp_x)
                                x_low, x_low_first_code = _temp_x.split([x_low.shape[1], x_low_first_code.shape[1]], dim=1)
                            else:
                                x_low = blk(x_low)
                        else:
                            raise NotImplementedError("Non-batchify mode for VQ low postprocess layers with non-MLP/LayerNorm blocks not implemented yet")
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                            else:
                                x_low = blk(x_low, cu_slens=cu_slens)
            else:
                raise NotImplementedError("VQ low postprocess layers other than Sequential not implemented yet")
                assert isinstance(self.vq_low_postprocess_layers, Mlp) or isinstance(self.vq_low_postprocess_layers, nn.Identity)
                x_low = self.vq_low_postprocess_layers(x_low)


        x_low_norm = x_low
        if not return_feat_rec_loss:
            feat_rec_loss_list = [self.zero] * len(slen)
        else:
            if x_low_norm.shape[-1] != anchor_x.shape[-1]:
                assert anchor_x.shape[-1] == 2 * x_low_norm.shape[-1]
                anchor_x = anchor_x[..., :x_low_norm.shape[-1]]
            xs_target = x_low_norm.split(slen, dim=1)
            xs_oryx = anchor_x.split(slen, dim=1)

            feat_rec_loss_list = []
            for _x, _y in zip(xs_target, xs_oryx):
                feat_rec_loss = F.mse_loss(_y.detach(), _x)
                feat_rec_loss_list.append(feat_rec_loss)

        if self.vq_low_enable_layernorm_trick:
            x_low = x_low * self.vq_low_norm_trick_post_quantize_affine_weight + self.vq_low_norm_trick_post_quantize_affine_bias
            if self.use_first_code_for_rec:
                x_low_first_code = x_low_first_code * self.vq_low_norm_trick_post_quantize_affine_weight + self.vq_low_norm_trick_post_quantize_affine_bias

        if self.use_first_code_for_rec:
            feature_lookup.update({'after_postprocess':{'seq_feature': x_low_first_code, 'shape_factor': 1 * common_factor}})
        else:
            feature_lookup.update({'after_postprocess':{'seq_feature': x_low, 'shape_factor': 1 * common_factor}})

        return x_low, commit_loss_low_list, feature_lookup, feat_rec_loss_list, ae_loss_list, info

    def _forward_high_branch(self, x_high, oryx_x, image_sizes, num_prefix_tokens, slen=None, cu_slens=None, return_feat_rec_loss=False):
        ####################################################################
        # --------     Branch semantic: vq_high, x_high         --------   # 
        ####################################################################
        
        if slen is None and cu_slens is None:
            batchify = True
        else:
            batchify = False

        if self.vq_high_enable_layernorm_trick:
            x_high = self.vq_high_norm_trick_pre_quantize_norm_wo_affine(x_high)

        anchor_x = x_high

        if isinstance(self.vq_high_preprocess_layers, torch.nn.Sequential):
            if batchify:
                # x is batchify
                for idx, blk in enumerate(self.vq_high_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_high = checkpoint(blk, x_high, use_reentrant=True)
                    else:
                        x_high = blk(x_high)
            else:
                for idx, blk in enumerate(self.vq_high_preprocess_layers):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x_high = checkpoint(blk, x_high, cu_slens, use_reentrant=True)
                    else:
                        x_high = blk(x_high, cu_slens=cu_slens)
        else:
            x_high = self.vq_high_preprocess_layers(x_high)


        # Token Shuffle
        if self.vq_high_enable_token_shuffle:
            if num_prefix_tokens == 0:
                B, N, C = x_high.shape
                x_high = self.vq_high_pre_token_shuffle_head(x_high) # 1 N C  -> 1 N 4C
                x_high = x_high.view(B, 4 * N, C) # 1 N 4C -> 1 4N C
                x_high = self.vq_high_post_token_shuffle_head(x_high) #  1 4N C ->  1 4N C
            else:
                raise NotImplementedError(f"I do not if it is correct to process cls token as below.")


        if batchify:
            xs = x_high.split(1, dim=0)            
        else:
            xs = x_high.split([length * 4 for length in slen] if self.vq_high_enable_token_shuffle else slen, dim=1)
        
        
        quantized_x_high_list = []
        commit_loss_high_list = []
        for _x in xs:
            _x, commit_loss, _ = self.vq_high(_x)
            quantized_x_high_list.append(_x)
            commit_loss_high_list.append(commit_loss)

        if batchify:
            x_high = torch.cat(quantized_x_high_list, dim=0)
        else:
            x_high = torch.cat(quantized_x_high_list, dim=1)

        # Token Shuffle
        if self.vq_high_enable_token_shuffle and self.vq_high_inverse_token_shuffle_head is not None:
            if num_prefix_tokens == 0:
                B, N, C = x_high.shape
                x_high = x_high.view(B, N//4, 4 * C) 
                x_high = self.vq_high_inverse_token_shuffle_head(x_high)
            else:
                raise NotImplementedError(f"I do not if it is correct to process cls token as below.")

        if isinstance(self.vq_high_postprocess_layers, torch.nn.Sequential):
            if batchify:
                for idx, blk in enumerate(self.vq_high_postprocess_layers):
                    if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                        x_high = blk(x_high)
                    else:
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_high = checkpoint(blk, x_high, use_reentrant=True)
                        else:
                            x_high = blk(x_high)
            else:
                for idx, blk in enumerate(self.vq_high_postprocess_layers):
                    if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                        x_high = blk(x_high)
                    else:
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            x_high = checkpoint(blk, x_high, cu_slens, use_reentrant=True)
                        else:
                            x_high = blk(x_high, cu_slens=cu_slens)
        else:
            assert isinstance(self.vq_high_postprocess_layers, Mlp) or isinstance(self.vq_high_postprocess_layers, nn.Identity)
            x_high = self.vq_high_postprocess_layers(x_high)

        x_high_norm = x_high
        if not return_feat_rec_loss:
            feat_rec_loss_list = [self.zero] * len(slen)
        else:
            xs_target = x_high_norm.split(slen, dim=1)
            xs_oryx = anchor_x.split(slen, dim=1)

            feat_rec_loss_list = []
            for _x, _y in zip(xs_target, xs_oryx):
                feat_rec_loss = F.mse_loss(_y.detach(), _x)
                feat_rec_loss_list.append(feat_rec_loss)

        if self.vq_high_enable_layernorm_trick:
            x_high = x_high * self.vq_high_norm_trick_post_quantize_affine_weight + self.vq_high_norm_trick_post_quantize_affine_bias

        return x_high, commit_loss_high_list, feat_rec_loss_list


    def forward_infer(
            self, 
            x, 
            num_prefix_tokens, 
            image_sizes, 
            slen=None, 
            cu_slens=None, 
            movq_plugin_position=None, 
            return_feat_rec_loss=False,
            cls_distill_feature_type='low',
        ):


        if slen is None and cu_slens is None:
            batchify = True
        else:
            batchify = False

        # x : [1, sum_i (N_i), C]
        oryx_x = x
        x_low = x_high = x

        prev_x_low = x_low
        prev_x_high = x_high

        if self.close_low_branch:
            raise RuntimeError(f'cant close low branch')
        else:
            x_low = self._forward_low_branc_infer(x_low, oryx_x, image_sizes, num_prefix_tokens, slen=slen, cu_slens=cu_slens, return_feat_rec_loss=return_feat_rec_loss)
        
        return x_low

    def forward(
            self, 
            x, 
            num_prefix_tokens, 
            image_sizes, 
            slen=None, 
            cu_slens=None, 
            movq_plugin_position=None, 
            return_feat_rec_loss=False,
            cls_distill_feature_type='low',
            infer=False
        ):

        if infer:
            return self.forward_infer(
                x=x,
                num_prefix_tokens=num_prefix_tokens,
                image_sizes=image_sizes,
                slen=slen,
                cu_slens=cu_slens,
                movq_plugin_position=movq_plugin_position,
                return_feat_rec_loss=return_feat_rec_loss,
                cls_distill_feature_type=cls_distill_feature_type,
            )


        if slen is None and cu_slens is None:
            batchify = True
        else:
            batchify = False

        # x : [1, sum_i (N_i), C]
        oryx_x = x
        x_low = x_high = x

        prev_x_low = x_low
        prev_x_high = x_high

        if self.close_low_branch:
            # raise RuntimeError(f'cant close low branch')
            x_low = x_low
            commit_loss_low_list = [self.zero] * len(slen)
            feat_rec_loss_list_low = [self.zero] * len(slen)
            ae_loss_list_low = [self.zero] * len(slen)
            feature_lookup = {}
        else:
            x_low, commit_loss_low_list, feature_lookup, feat_rec_loss_list_low, ae_loss_list_low, info = self._forward_low_branch(x_low, oryx_x, image_sizes, num_prefix_tokens, slen=slen, cu_slens=cu_slens, return_feat_rec_loss=return_feat_rec_loss)
        
        if self.close_high_branch:
            x_high = x_high
            commit_loss_high_list = [self.zero] * len(slen)
            feat_rec_loss_list_high = [self.zero] * len(slen)
        else:
            x_high, commit_loss_high_list, feat_rec_loss_list_high = self._forward_high_branch(x_high, oryx_x, image_sizes, num_prefix_tokens, slen=slen, cu_slens=cu_slens, return_feat_rec_loss=return_feat_rec_loss)

        # for quick implement
        commit_loss_list = [loss1 + loss2 for loss1, loss2 in zip(commit_loss_low_list, commit_loss_high_list)]
        # for quick implement
        if cls_distill_feature_type == 'low':
            feat_rec_loss_list = feat_rec_loss_list_low
        else:
            feat_rec_loss_list = feat_rec_loss_list_high

        ae_loss_list = ae_loss_list_low


        if self.one_time_debug_info:
            try:
                prev_x_low_and_x_low_allclose = torch.allclose(prev_x_low.float(), x_low.float(), atol=1e-5)
                if prev_x_low_and_x_low_allclose:
                    self.close_low_branch = True
            except:
                pass

            try:
                prev_x_high_and_x_high_allclose = torch.allclose(prev_x_high.float(), x_high.float(), atol=1e-5)
                if prev_x_high_and_x_high_allclose:
                    self.close_high_branch = True
            except:
                pass
            self.one_time_debug_info = False


        if self.use_first_code_for_rec:
            assert movq_plugin_position == 'after_postprocess'
        if self.close_low_branch and self.close_high_branch:
            movq_input_feature = {'seq_feature': oryx_x, 'shape_factor': 1}
            assert (x_high == oryx_x).all() and (x_low == oryx_x).all()
        else:
            if movq_plugin_position is not None:
                movq_input_feature = feature_lookup[movq_plugin_position]
            else:
                movq_input_feature = None

        return movq_input_feature, x_high, x_low, commit_loss_list, feat_rec_loss_list, ae_loss_list, info

