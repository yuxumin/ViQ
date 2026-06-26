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

import torch.distributed as dist
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

from typing import Optional

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import os

try:
    from llava_viq._paths import PROJECT_ROOT as _PROJECT_ROOT
except ImportError:
    _PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    )

from ..heads.movq_modules import decoder_head_MoVQ
from ..heads.vitvq_modules import ViTDecoder
from ..envir_defines import *
from ..layers.model_utils import depthwise_separable_conv, RMSNorm, ConvStem, \
        _no_grad_trunc_normal_, trunc_normal_, init_weights, init_weights_vit_timm, \
        CasualAttention, Attention, SwiGLU, LayerScale, Block, CausalBlock, \
        get_target_size, crop_images_and_featuresv2
from ..layers._common import ResualMLP
from ..heads.dual_vq_head import DualVQHead
from ..heads.vae_heads import FixVAEHead, FixVAEHead_F16, QwenImageVAEHead, QwenImageVAEHead_trainable



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
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
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
        vq_low_type: Literal["vq", "bsq", "fsq", "packer", "lfq", "simvq", "simvq-wide", "fake", "fake-wide", "fake-middle", "simvq-middle", "fake-normal", "simvq-normal", 'ibq', 'ibq-wide', 'ibq-normal', 'ibq-middle'] = "simvq",
        vq_low_size: int = 2 ** 15, 
        vq_low_limit: Literal["none", "l2", "tanh", "escape"] = "none",
        vq_high_type: Literal["vq", "bsq", "fsq", "packer", "lfq", "simvq", "simvq-wide", "fake", "fake-wide", "fake-middle", "simvq-middle", "fake-normal", "simvq-normal", 'ibq', 'ibq-wide', 'ibq-normal', 'ibq-middle'] = "simvq",
        vq_high_size: int = 2 ** 15, 
        vq_high_limit: Literal["none", "l2", "tanh", "escape"] = "none",
        vq_high_enable_token_shuffle: bool = True,
        vq_high_enable_layernorm_trick: bool = True,
        vq_low_enable_token_shuffle: bool = True,
        vq_low_enable_layernorm_trick: bool = True,
        return_feat_rec_loss: bool = False,
        return_cls_distill_loss: bool = True,
        enable_movq_decoder: bool = True,
        movq_type: Literal["movq", "vqvit", "fixed_vae", "fixed_vae_f16", 'qwen_vae'] = "movq",
        movq_preprocess_embed_dim: int = 64,
        movq_preprocess_type: Literal["none", "mlp", "attn", "attn-shallow"] = "attn",
        movq_plugin_position: Literal["oryx", "after_sa", "after_ts", 'after_quant_8x', 'after_quant_16x', 'after_postprocess', 'after_concat'] = "oryx",
        movq_disable_perceptual_loss: bool = False,
        vq_low_preprocess_type: Literal["none", "attn", "attn-large", "attn-large-shallow"] = "attn",
        vq_high_preprocess_type: Literal["none", "attn"] = "attn",
        vq_low_postprocess_type: Literal["none", "mlp", "attn", "attn-large", "attn-deep", "casual_attn", "casual_attn-wide", "resualmlp", "resualmlp-light"] = "attn",
        vq_high_postprocess_type: Literal["none", "mlp", "attn", "casual_attn"] = "attn",
        mllm_feature_type: Literal["high", "concat", "low"] = "high",
        cls_distill_feature_type: Literal["high", "low"] = "high",
        use_first_code_for_rec: bool = False,
        extra_sa_before_map: bool = False
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
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        def _blocks(dim, depth):
            # Shared Block factory; all uses share identical kwargs except dim/depth.
            # (dpr is defined later but resolved lazily at call time.)
            return [Block(
                dim=dim,
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
            ) for _ in range(depth)]

        def _mlp(in_f, hidden_f, out_f, act_layer=nn.GELU):
            return Mlp(in_features=in_f, hidden_features=hidden_f,
                       out_features=out_f, act_layer=act_layer)

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

        # Classifier Head
        self.return_cls_distill_loss = return_cls_distill_loss
        if return_cls_distill_loss:
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


            self.extra_sa_before_map = extra_sa_before_map
            if self.extra_sa_before_map:
                self.extra_sa_before_attn_pool = nn.Sequential(
                        *_blocks(self.embed_dim, 1)
                    )

        # additional components
        # Vector-Quantizer
        self.dualvq_head = DualVQHead(
                        self.embed_dim, 
                        vq_low_type, 
                        vq_low_size, 
                        vq_low_limit, 
                        vq_high_type, 
                        vq_high_size, 
                        vq_high_limit,
                        vq_high_enable_token_shuffle=vq_high_enable_token_shuffle,
                        vq_high_enable_layernorm_trick=vq_high_enable_layernorm_trick,
                        vq_low_enable_token_shuffle=vq_low_enable_token_shuffle,
                        vq_low_enable_layernorm_trick=vq_low_enable_layernorm_trick,
                        vq_low_preprocess_type=vq_low_preprocess_type,
                        vq_high_preprocess_type=vq_high_preprocess_type,
                        vq_low_postprocess_type=vq_low_postprocess_type,
                        vq_high_postprocess_type=vq_high_postprocess_type,
                        use_first_code_for_rec=use_first_code_for_rec
                    )
        
        # TODO: deal with the situation if the 'downsample2x' in vq_low_preprocess_type
        # attn-downsample2x True
        # attn-downsample2x-nomerge False
        self.low_downsample2x = ('downsample2x' in vq_low_preprocess_type ) and ('nomerge' not in vq_low_preprocess_type)

        # vq one for recon (8x with token shuffle)
        self.vq_low_feat_dim = self.dualvq_head.vq_low_features
        # vq high for high-level semantic (16x always)
        self.vq_high_feat_dim = self.dualvq_head.vq_high_features

        self.cls_distill_feature_type = cls_distill_feature_type
        self.mllm_feature_type = mllm_feature_type
        if self.mllm_feature_type == 'high' or self.mllm_feature_type == 'low':
            self.mllm_out_dims = embed_dim
        else:
            self.mllm_out_dims = 2 * embed_dim

        self.return_feat_rec_loss = return_feat_rec_loss 

        self.enable_movq_decoder = enable_movq_decoder
        self.movq_type = movq_type
        self.movq_plugin_position = movq_plugin_position
        self.movq_disable_perceptual_loss = movq_disable_perceptual_loss

        if self.enable_movq_decoder:
            if self.movq_plugin_position == 'oryx':
                _feat_dim = embed_dim
                patch_size = 16
                ch_mult = [1, 1, 2, 2, 4] # 16x feat

            elif self.movq_plugin_position == 'after_sa':
                _feat_dim = 2 * embed_dim if 'attn-large' in vq_low_preprocess_type else embed_dim
                patch_size = 16
                ch_mult = [1, 1, 2, 2, 4] # 16x feat

            elif self.movq_plugin_position == 'after_ts':
                assert vq_low_enable_token_shuffle, f"after_ts position is meaningless if vq_low_enable_token_shuffle is False"
                _feat_dim = 2 * embed_dim if 'attn-large' in vq_low_preprocess_type  else embed_dim
                patch_size = 8
                ch_mult = [1, 2, 2, 4] # 8x feat

            elif self.movq_plugin_position == 'after_quant_8x':
                raise NotImplementedError(f"do not support 8x feat. (TS will distort the space relation).")

            elif self.movq_plugin_position == 'after_quant_16x':
                _feat_dim = self.vq_low_feat_dim * 4
                patch_size = 16
                ch_mult = [1, 1, 2, 2, 4] # 16x feat
            
            elif self.movq_plugin_position == 'after_postprocess':
                _feat_dim = 2 * embed_dim if 'attn-large' in vq_low_postprocess_type  else embed_dim
                patch_size = 16
                ch_mult = [1, 1, 2, 2, 4] # 16x feat
            
            elif self.movq_plugin_position == 'after_concat':
                _feat_dim = 3 * embed_dim if 'attn-large' in vq_low_postprocess_type else 2 * embed_dim
                patch_size = 16
                ch_mult = [1, 1, 2, 2, 4] # 16x feat
            
            else:
                raise NotImplementedError(f"do not support the movq_plugin_position {movq_plugin_position}")

            if movq_preprocess_type == 'mlp':
                self.movq_preprocess_layers = _mlp(_feat_dim, movq_preprocess_embed_dim, movq_preprocess_embed_dim)
                movq_input_dim = movq_preprocess_embed_dim
            elif movq_preprocess_type in ('attn', 'attn-shallow'):
                num_blocks = 2 if movq_preprocess_type == 'attn-shallow' else 3
                self.movq_preprocess_layers = nn.Sequential(
                    _mlp(_feat_dim, movq_preprocess_embed_dim, movq_preprocess_embed_dim),
                    norm_layer(movq_preprocess_embed_dim),
                    *_blocks(movq_preprocess_embed_dim, num_blocks)
                )
                movq_input_dim = movq_preprocess_embed_dim
            else:
                self.movq_preprocess_layers = nn.Identity()
                movq_input_dim = _feat_dim


        if weight_init != "skip":
            self.init_weights(weight_init)

        # define decoder after init (perceptual loss)
        if self.enable_movq_decoder:
            if movq_type == 'vqvit':
                self.movq_head = ViTDecoder(
                                in_dims = movq_input_dim,
                                image_size=512, # should be align with fixed size setting
                                patch_size=patch_size,
                                dim=768,
                                disable_perceptual_loss=movq_disable_perceptual_loss
                            )
            elif movq_type == 'movq':
                self.movq_head = decoder_head_MoVQ(
                            quant_dim=movq_input_dim, 
                            ch_mult=ch_mult,
                            disable_perceptual_loss=movq_disable_perceptual_loss,
                        )
            elif movq_type == 'fixed_vae':
                assert VAE_PATH is not None
                self.movq_head = FixVAEHead(
                            in_dims=movq_input_dim,
                            vae_path=VAE_PATH,
                            patch_size=patch_size
                        )
            elif movq_type == 'fixed_vae_f16':
                self.movq_head = FixVAEHead_F16(
                        in_dims=movq_input_dim
                    )
            elif movq_type == 'qwen_vae':
                assert VAE_PATH is not None
                self.movq_head = QwenImageVAEHead(
                        in_dims=movq_input_dim,
                        vae_path=VAE_PATH,
                        factor=4 if self.low_downsample2x else 2
                    )
            elif movq_type == 'qwen_vae_trainable':
                assert VAE_PATH is not None
                self.movq_head = QwenImageVAEHead_trainable(
                        in_dims=movq_input_dim,
                        vae_path=VAE_PATH,
                    )
            else:
                raise

        self.one_time_debug_info = True

        
    @torch.no_grad()
    def _update_module_by_module(self, source_module, target_module):
        for (name_s, param_s), (name_t, param_t) in zip(source_module.named_parameters(), target_module.named_parameters()):
            param_t.data = param_s.data
            print(f'copy the weight from {name_s} to {name_t}')

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
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
        self.dualvq_head.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

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

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n
        )

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features_list_infer(self, x_list):
        x_all = []
        image_sizes = []
        padded_input_images = []
        for x in x_list:
            bs, _, h, w = x.shape

            # fix patch size=14 in datasets 
            pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
            pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

            bs, _, h, w = x.shape
            padded_input_images.append(x)

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

        # --------------------------------- Oryx Above ------------------------------- #

        # x : [1, sum_i (N_i), C]
        x_return_ori = x.split(slen, dim=1)

        x = self.dualvq_head(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            slen=slen,
            cu_slens=cu_slens,
            image_sizes=image_sizes,
            movq_plugin_position=self.movq_plugin_position,
            return_feat_rec_loss=self.return_feat_rec_loss,
            cls_distill_feature_type=self.cls_distill_feature_type,
            infer=True
        )

        return x

    def original_forward_list(self, x_list):
        input_images = x_list
        x_all = []
        image_sizes = []
        padded_input_images = []
        for x in x_list:
            bs, _, h, w = x.shape

            # fix patch size=14 in datasets 
            pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
            pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

            bs, _, h, w = x.shape
            padded_input_images.append(x)

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
        return x, slen, cu_slens, padded_input_images, image_sizes

    def forward_features_list(self, x_list):
        # with torch.no_grad():
        x, slen, cu_slens, padded_input_images, image_sizes = self.original_forward_list(x_list)

        # --------------------------------- Oryx Above ------------------------------- #

        # x : [1, sum_i (N_i), C]
        oryx_x = x
        x_return_ori = x.split(slen, dim=1)

        movq_input_feature, x_high, x_low, commit_loss_list, feat_rec_loss_list, ae_loss_list, info = self.dualvq_head(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            slen=slen,
            cu_slens=cu_slens,
            image_sizes=image_sizes,
            movq_plugin_position=self.movq_plugin_position,
            return_feat_rec_loss=self.return_feat_rec_loss,
            cls_distill_feature_type=self.cls_distill_feature_type,
        )

        if self.enable_movq_decoder and not SKIP_RECON_VAE:
            seq_feature = movq_input_feature['seq_feature']
            shape_factor = movq_input_feature['shape_factor']

            ###### 1.  crop







            ###### 2.  any res
            unshuffle_slen = [int(length * shape_factor * shape_factor) for length in slen]
            unshuffle_image_sizes = [(int(h * shape_factor), int(w * shape_factor)) for (h, w) in image_sizes]
            unshuffle_cu_indices = [0, ]
            for i in unshuffle_slen:
                unshuffle_cu_indices.append(unshuffle_cu_indices[-1] + i)
            unshuffle_cu_slens = torch.tensor(unshuffle_cu_indices, dtype=torch.int32).to(x.device)

            if isinstance(self.movq_preprocess_layers, torch.nn.Sequential):
                for idx, blk in enumerate(self.movq_preprocess_layers):
                    if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                        seq_feature = blk(seq_feature)
                    else:
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            seq_feature = checkpoint(blk, seq_feature, unshuffle_cu_slens if shape_factor != 1 else cu_slens, use_reentrant=True)
                        else:
                            seq_feature = blk(seq_feature, cu_slens=unshuffle_cu_slens if shape_factor != 1 else cu_slens)
            else:
                seq_feature = self.movq_preprocess_layers(seq_feature)

            if self.movq_type == 'movq':
                rec_loss = self.movq_head.forward_loss_varlen(
                    seq_feature,
                    padded_input_images,
                    unshuffle_slen if shape_factor != 1 else slen,
                    unshuffle_cu_slens if shape_factor != 1  else cu_slens,
                    unshuffle_image_sizes if shape_factor != 1  else image_sizes,
                    num_prefix_tokens=self.num_prefix_tokens,
                    image_limit=32
                )

            elif self.movq_type == 'vqvit':
                rec_loss = self.movq_head.forward(
                    seq_feature,
                    padded_input_images,
                    unshuffle_slen if shape_factor != 1  else slen,
                    unshuffle_cu_slens if shape_factor != 1 else cu_slens,
                    unshuffle_image_sizes if shape_factor != 1  else image_sizes,
                    num_prefix_tokens=self.num_prefix_tokens,
                    only_return_loss=True,
                    image_limit=32
                )
            elif self.movq_type == 'fixed_vae' or self.movq_type == 'fixed_vae_f16' or self.movq_type == 'qwen_vae'  or self.movq_type == 'qwen_vae_trainable':
                # FIXME: if shape factor == 0.5/. use  shape_factor!=1 instead.
                rec_loss = self.movq_head.forward_loss_varlen(
                    seq_feature,
                    padded_input_images,
                    unshuffle_slen if shape_factor != 1 else slen,
                    unshuffle_cu_slens if shape_factor != 1 else cu_slens,
                    unshuffle_image_sizes if shape_factor != 1 else image_sizes,
                    num_prefix_tokens=self.num_prefix_tokens
                )
            
            else:
                raise
        else:
            rec_loss = self.zero

        if self.mllm_feature_type == 'high':
            out_x = x_high
        elif self.mllm_feature_type == 'low':
            out_x = x_low
        elif self.mllm_feature_type == 'concat':
            out_x = torch.cat([x_high, x_low], dim=-1)
        else:
            raise NotImplementedError(f"mllm_feature_type {self.mllm_feature_type} is not supported.")
        
        if self.low_downsample2x and self.mllm_feature_type == 'low':
            xs_out = out_x.split([s // 4 for s in slen], dim=1)
            image_sizes = [(h//2, w//2) for (h, w) in image_sizes]
        else:
            xs_out = out_x.split(slen, dim=1)

        xs_ori = x_return_ori
        xs_high = x_high.split(slen, dim=1)

        if self.low_downsample2x:
            xs_low = x_low.split([s // 4 for s in slen], dim=1)
        else:
            xs_low = x_low.split(slen, dim=1)

        return xs_ori, xs_out, xs_high, xs_low, image_sizes, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, info

    def forward_codes4recon_list(self, codes, image_sizes, slen, cu_slens):
        batchify = False
        x_low = codes # 1, N, 3072
        # Token Shuffle
        if self.dualvq_head.vq_low_enable_token_shuffle and self.dualvq_head.vq_low_inverse_token_shuffle_head is not None:
            B, N, C = x_low.shape
            x_low = x_low.view(B, N//4, 4 * C) 
            x_low = self.dualvq_head.vq_low_inverse_token_shuffle_head(x_low)
            
        # aux layer
        if self.dualvq_head.vq_low_aux_postprocess_layers is not None:
            if isinstance(self.dualvq_head.vq_low_aux_postprocess_layers, torch.nn.Sequential):
                if batchify:
                    # x is batchify
                    for idx, blk in enumerate(self.dualvq_head.vq_low_aux_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, use_reentrant=True)
                            else:
                                x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.dualvq_head.vq_low_aux_postprocess_layers):                        
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                            else:
                                x_low = blk(x_low, cu_slens=cu_slens)
            else:
                x_low = self.dualvq_head.vq_low_aux_postprocess_layers(x_low)

        # original layer
        if self.dualvq_head.vq_low_postprocess_layers is not None:
            if isinstance(self.dualvq_head.vq_low_postprocess_layers, torch.nn.Sequential):
                if batchify:
                    for idx, blk in enumerate(self.dualvq_head.vq_low_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, use_reentrant=True)
                            else:
                                x_low = blk(x_low)
                else:
                    for idx, blk in enumerate(self.dualvq_head.vq_low_postprocess_layers):
                        if isinstance(blk, Mlp) or isinstance(blk, ResualMLP) or isinstance(blk, nn.LayerNorm):
                            x_low = blk(x_low)
                        else:
                            if self.grad_checkpointing and not torch.jit.is_scripting():
                                x_low = checkpoint(blk, x_low, cu_slens, use_reentrant=True)
                            else:
                                x_low = blk(x_low, cu_slens=cu_slens)
            else:
                assert isinstance(self.dualvq_head.vq_low_postprocess_layers, Mlp) or isinstance(self.dualvq_head.vq_low_postprocess_layers, nn.Identity)
                x_low = self.dualvq_head.vq_low_postprocess_layers(x_low)

        if self.dualvq_head.vq_low_enable_layernorm_trick:
            x_low = x_low * self.dualvq_head.vq_low_norm_trick_post_quantize_affine_weight + self.dualvq_head.vq_low_norm_trick_post_quantize_affine_bias

        shape_factor = 1
        unshuffle_slen = [length * shape_factor * shape_factor for length in slen]
        unshuffle_image_sizes = [(h * shape_factor, w * shape_factor) for (h, w) in image_sizes]
        unshuffle_cu_indices = [0, ]
        for i in unshuffle_slen:
            unshuffle_cu_indices.append(unshuffle_cu_indices[-1] + i)
        unshuffle_cu_slens = torch.tensor(unshuffle_cu_indices, dtype=torch.int32).to(x_low.device)

        seq_feature = x_low

        if isinstance(self.movq_preprocess_layers, torch.nn.Sequential):
            for idx, blk in enumerate(self.movq_preprocess_layers):
                if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                    seq_feature = blk(seq_feature)
                else:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        seq_feature = checkpoint(blk, seq_feature, unshuffle_cu_slens if shape_factor == 2 else cu_slens, use_reentrant=True)
                    else:
                        seq_feature = blk(seq_feature, cu_slens=unshuffle_cu_slens if shape_factor == 2 else cu_slens)
        else:
            seq_feature = self.movq_preprocess_layers(seq_feature)

        image_numpy_list = self.movq_head.forward_recon_list(
            seq_feature,
            slen,
            cu_slens,
            image_sizes,
            self.num_prefix_tokens
        )

        x_low = x_low.split(slen, dim=1)

        return x_low, image_sizes, image_numpy_list

    def forward_features(self, x):
        raise
        input_images = x
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]

        x = self.patch_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        # --------------------------------- Oryx Above ------------------------------- #
        oryx_x = x
        x_return_ori = x
        
        movq_input_feature, x_high, x_low, commit_loss_high_list, commit_loss_low_list = self.dualvq_head(
            x=x,
            num_prefix_tokens=self.num_prefix_tokens,
            slen=None,
            cu_slens=None,
            movq_plugin_position=self.movq_plugin_position
        )
        
        if not self.return_feat_rec_loss:
            feat_rec_loss_list = [self.zero] * len(x_high)
        else:
            if self.cls_distill_feature_type == 'low':
                target_x = x_low
            else:
                target_x = x_high

            xs_target = target_x.split(1, dim=0)
            xs_oryx = oryx_x.split(1, dim=0)

            feat_rec_loss_list = []
            for _x, _y in zip(xs_target, xs_oryx):
                feat_rec_loss = F.mse_loss(_y.detach(), _x)
                feat_rec_loss_list.append(feat_rec_loss)


        if self.one_time_debug_info:
            try:
                x_and_x_low_allclose = torch.allclose(x.float(), x_low.float(), atol=1e-5)
            except:
                pass
            self.one_time_debug_info = False


        if self.enable_movq_decoder:
            seq_feature = movq_input_feature['seq_feature']
            shape_factor = movq_input_feature['shape_factor']


            if isinstance(self.movq_preprocess_layers, torch.nn.Sequential):
                for idx, blk in enumerate(self.movq_preprocess_layers):
                    if isinstance(blk, Mlp) or isinstance(blk, nn.LayerNorm):
                        seq_feature = blk(seq_feature)
                    else:
                        if self.grad_checkpointing and not torch.jit.is_scripting():
                            seq_feature = checkpoint(blk, seq_feature, use_reentrant=True)
                        else:
                            seq_feature = blk(seq_feature)
            else:
                seq_feature = self.movq_preprocess_layers(seq_feature)


            shaped_feature = seq_feature[:, self.num_prefix_tokens:].reshape(len(seq_feature), h * shape_factor, w * shape_factor, -1)
            
            if self.movq_type == 'vqvit':
                rec_loss = self.movq_head.forward_batch(
                    shaped_feature,
                    input_images,
                    batch_limit=32
                )
            elif self.movq_type == 'movq':
                rec_loss = self.movq_head.forward_loss(
                    shaped_feature,
                    input_images,
                    batch_limit=32
                )
            elif self.movq_type == 'fixed_vae' or self.movq_type == 'fixed_vae_f16':
                pass

            else:
                raise

        else:
            rec_loss = self.zero

        if self.mllm_feature_type == 'high':
            out_x = x_high
        elif self.mllm_feature_type == 'low':
            out_x = x_low
        elif self.mllm_feature_type == 'concat':
            out_x = torch.cat([x_high, x_low], dim=-1)
        else:
            raise NotImplementedError(f"mllm_feature_type {self.mllm_feature_type} is not supported.")

        # for quick implement
        commit_loss_list = [loss1 + loss2 for loss1, loss2 in zip(commit_loss_low_list, commit_loss_high_list)]

        return  x_return_ori, out_x, x_high, x_low, (h, w), rec_loss, commit_loss_list, feat_rec_loss_list

    def forward_features_teacher(self, x: torch.Tensor) -> torch.Tensor:
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]

        x = self.patch_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        # fake the batchify input as a list tensor for convenience
        xs = x.split(1, dim=0)
        x_list = torch.cat(xs, dim=1)
        slen = [xs[0].shape[1]] * len(xs)

        cu_indices = [0, ]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)
        cu_slens = torch.tensor(cu_indices, dtype=torch.int32).to(x.device)
        image_sizes = [(h, w)] * len(xs)

        _, x_high, x_low, _, _, _ = self.dualvq_head(
            x=x_list,
            num_prefix_tokens=self.num_prefix_tokens,
            slen=slen,
            cu_slens=cu_slens,
            image_sizes=image_sizes,
        )


        if self.mllm_feature_type == 'high':
            out_x = x_high
        elif self.mllm_feature_type == 'low':
            out_x = x_low
        elif self.mllm_feature_type == 'concat':
            out_x = torch.cat([x_high, x_low], dim=-1)
        else:
            raise NotImplementedError(f"mllm_feature_type {self.mllm_feature_type} is not supported.")
        
        xs_out = out_x.split(slen, dim=1)
        xs_high = x_high.split(slen, dim=1)
        xs_low = x_low.split(slen, dim=1)

        out_x = torch.cat(xs_out, dim=0)
        x_high = torch.cat(xs_high, dim=0)
        x_low = torch.cat(xs_low, dim=0)

        return out_x, x_high, x_low, (h, w)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        if self.extra_sa_before_map:
            x = self.extra_sa_before_attn_pool(x)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    @torch.no_grad()
    def forward_teacher(self, input):
        assert type(input) is not list
        _, _, cls_token, _, _, _, _ = self.forward(list(input.split(1, dim=0)), cal_attn_pool=True)
        return cls_token

    def forward(self, input, cal_attn_pool=False, infer_mode=False):
        if infer_mode:
            assert type(input) is list
            x = self.forward_features_list_infer(input)
            return x

        if type(input) is list:
            xs_ori, xs_out, xs_high, xs_low, image_sizes, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, info = self.forward_features_list(input)
            if not cal_attn_pool:
                return xs_out, image_sizes, None, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, info
            else:
                assert self.return_cls_distill_loss, f"self.return_cls_distill_loss should be True if cal_attn_pool is True"
                if CLS_WITH_ORIGIN_LAYER_OUTPUT:
                    cls_tokens = []
                    for cur_x in xs_ori:
                        cls_tokens.append(self.forward_head(cur_x))
                    cls_tokens = torch.cat(cls_tokens, dim=0)
                else:
                    cls_tokens = []
                    if self.cls_distill_feature_type == 'low':
                        target_xs = xs_low
                    else:
                        target_xs = xs_high
                    for cur_x in target_xs:
                        cls_tokens.append(self.forward_head(cur_x))
                    cls_tokens = torch.cat(cls_tokens, dim=0)
                return xs_out, image_sizes, cls_tokens, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, info
                
        else:
            raise
            x_ori, x_out, x_high, x_low, image_sizes, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list = self.forward_features(input)
            
            if not cal_attn_pool:
                return x_out, image_sizes, None, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list
            else:
                assert self.return_cls_distill_loss, f"self.return_cls_distill_loss should be True if cal_attn_pool is True"
                if CLS_WITH_ORIGIN_LAYER_OUTPUT:
                    cls_token = self.forward_head(x_ori)
                else:
                    if self.cls_distill_feature_type == 'low':
                        cls_token = self.forward_head(x_low)
                    else:
                        cls_token = self.forward_head(x_high)
                return x_out, image_sizes, cls_token

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
    },
    "siglip2_so400m_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
}

 # TODO: Why need it.

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
    old_ckpt: bool = False,
    teacher: bool = False,
    gradient_checkpointing: bool = False,
    vq_low_type: Literal["vq", "bsq", "fsq", "packer", "lfq", "simvq", "simvq-wide", "fake", "fake-wide", "fake-middle", "simvq-middle", "fake-normal", "simvq-normal", 'ibq', 'ibq-wide', 'ibq-normal', 'ibq-middle'] = "simvq",
    vq_low_size: int = 2 ** 15, 
    vq_low_limit: Literal["none", "l2", "tanh", "escape"] = "none",
    vq_high_type: Literal["vq", "bsq", "fsq", "packer", "lfq", "simvq", "simvq-wide", "fake", "fake-wide", "fake-middle", "simvq-middle", "fake-normal", "simvq-normal", 'ibq', 'ibq-wide', 'ibq-normal', 'ibq-middle'] = "simvq",
    vq_high_size: int = 2 ** 15, 
    vq_high_limit: Literal["none", "l2", "tanh", "escape"] = "none",
    vq_low_enable_token_shuffle: bool = True,
    vq_low_enable_layernorm_trick: bool = True,
    vq_high_enable_token_shuffle: bool = True,
    vq_high_enable_layernorm_trick: bool = True,
    return_feat_rec_loss: bool = False,
    return_cls_distill_loss: bool = True,
    enable_movq_decoder: bool = True,
    movq_type: Literal["movq", "vqvit", 'fixed_vae', 'fixed_vae_f16', 'qwen_vae'] = "movq",
    movq_preprocess_embed_dim: int = 64,
    movq_preprocess_type: Literal["none", "mlp", "attn", "attn-shallow"] = "attn",
    movq_plugin_position: Literal["oryx", "after_sa", "after_ts", 'after_quant_8x', 'after_quant_16x', 'after_postprocess', 'after_concat'] = "oryx",
    movq_disable_perceptual_loss: bool = False,
    vq_low_preprocess_type: Literal["none", "attn", "attn-large", "attn-large-shallow"] = "attn",
    vq_high_preprocess_type: Literal["none", "attn"] = "attn",
    vq_low_postprocess_type: Literal["none", "mlp", "attn", "attn-large", "attn-deep", "casual_attn", "casual_attn-wide", "resualmlp", "resualmlp-light"] = "attn",
    vq_high_postprocess_type: Literal["none", "mlp", "attn", "casual_attn"] = "attn",
    mllm_feature_type: Literal["high", "concat", "low"] = "high",
    cls_distill_feature_type: Literal["high", "low"] = "high",
    use_first_code_for_rec: bool = False,
    extra_sa_before_map: bool = False,
    **kwargs,
):
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
        # -- new -- #
        vq_low_type=vq_low_type,
        vq_low_size=vq_low_size,
        vq_low_limit=vq_low_limit,
        vq_high_type=vq_high_type,
        vq_high_size=vq_high_size, 
        vq_high_limit=vq_high_limit,
        vq_low_enable_token_shuffle=vq_low_enable_token_shuffle,
        vq_low_enable_layernorm_trick=vq_low_enable_layernorm_trick,
        vq_high_enable_token_shuffle=vq_high_enable_token_shuffle,
        vq_high_enable_layernorm_trick=vq_high_enable_layernorm_trick,
        enable_movq_decoder=enable_movq_decoder,
        movq_type=movq_type,
        movq_preprocess_embed_dim=movq_preprocess_embed_dim,
        movq_preprocess_type=movq_preprocess_type,
        movq_plugin_position=movq_plugin_position,
        movq_disable_perceptual_loss=movq_disable_perceptual_loss,
        return_feat_rec_loss=return_feat_rec_loss,
        return_cls_distill_loss=return_cls_distill_loss,
        vq_low_preprocess_type=vq_low_preprocess_type,
        vq_high_preprocess_type=vq_high_preprocess_type,
        vq_low_postprocess_type=vq_low_postprocess_type,
        vq_high_postprocess_type=vq_high_postprocess_type,
        mllm_feature_type=mllm_feature_type,
        cls_distill_feature_type=cls_distill_feature_type,
        use_first_code_for_rec=use_first_code_for_rec,
        extra_sa_before_map=extra_sa_before_map
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if ckpt_path.endswith(".pth"):
            new_state_dict = {}
            for k in state_dict.keys():
                if 'perceptual_loss' in k:
                    continue
                if k.startswith('base_model.model.model.vision_tower.vision_tower.'):
                    new_k = k.replace('base_model.model.model.vision_tower.vision_tower.', '')
                    new_state_dict[new_k] = state_dict[k]

        else:
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("visual.trunk."):
                    new_state_dict[key[13:]] = state_dict[key]

        if not teacher:
            model = resize_evaclip_pos_embed(model, interpolation='bilinear')
            patch_embed = new_state_dict['patch_embed.proj.weight']
            if patch_embed.shape[-1] != model.patch_embed.proj.weight.shape[-1]:
                patch_embed = torch.nn.functional.interpolate(
                    patch_embed.float(), size=(vision_cfg.patch_size, vision_cfg.patch_size), mode='bicubic', align_corners=False)
                print(f'interpolate model patch size to {vision_cfg.patch_size}...')
                new_state_dict['patch_embed.proj.weight'] = patch_embed
            pos_embed = new_state_dict['pos_embed']


            if pos_embed.shape[1] != model.pos_embed.shape[1]:
                pos_embed = pos_embed.reshape(1, 24, 24, vision_cfg.width).permute(0, 3, 1, 2)
                pos_embed = torch.nn.functional.interpolate(pos_embed, size=(128, 128), mode='bicubic', align_corners=False)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, vision_cfg.width)
                print(f'interpolate model pos embed size to 128...')
            new_state_dict['pos_embed'] = pos_embed
        
        incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
        # The plain SigLIP checkpoint only contains the ViT backbone; the ViQ
        # head modules (dualvq_head / movq_preprocess_layers / movq_head / the
        # teacher tower) are built fresh and are *expected* to be missing here.
        # Only report backbone keys that genuinely failed to load.
        _head_prefixes = ('dualvq_head.', 'movq_preprocess_layers.', 'movq_head.',
                          'vision_tower_teacher.')
        backbone_missing = [k for k in incompatible_keys.missing_keys
                            if not k.startswith(_head_prefixes)]
        print(f"SigLIP-ViT restores from {ckpt_path}, Act as Teacher? {teacher}")
        if backbone_missing or incompatible_keys.unexpected_keys:
            print(f"\tbackbone missing_keys: {backbone_missing}")
            print(f"\tunexpected_keys: {incompatible_keys.unexpected_keys}")
        else:
            print("\tbackbone loaded OK (ViQ head modules initialized fresh).")
    if gradient_checkpointing:
        model.set_grad_checkpointing(True)
    return model


from transformers import CLIPImageProcessor
from .siglip_vit_anyres import create_siglip_vit as create_siglip_vit_oryx2

class AnyResViqWrapper(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args

        self.select_layer = -1
        if self.select_layer < -1: self.select_layer += 1
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.forward_times = 0

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        clip_path = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'model', 'multimodal_encoder', 'assets', 'default_processor')
        print(f'Loading CLIPImageProcessor from {clip_path}')
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_path)
        self.image_processor.image_mean = [0.5, 0.5, 0.5]
        self.image_processor.image_std = [0.5, 0.5, 0.5]
        print("Loading vision model...")

        if VIQ_SO400M:
            model_name = 'siglip2_so400m_patch16_384'
            if os.path.exists('/home/models/siglip2_so400m_oryx.pth'):
                model_path = '/home/models/siglip2_so400m_oryx.pth'
                print("Loading vision model from /home/models/siglip2_so400m_oryx.pth")
            else:
                raise NotImplementedError(f'weights are not found in /home/models/siglip2_so400m_oryx.pth')

        else:
            model_name = 'siglip2_giant_patch16_384'
            if os.path.exists('/home/models/siglip2_g_anyres_s4.pth'):
                model_path = '/home/models/siglip2_g_anyres_s4.pth'
                print("Loading vision model from /home/models/siglip2_g_anyres_s4.pth")
            else:
                raise NotImplementedError(f'weights are not found in /home/models/siglip2_g_anyres_s4.pth')


        viq_args = {
            'ckpt_path': model_path,
            'model_name': model_name,
            'vq_low_type': VQ_LOW_TYPE,
            'vq_low_size': VQ_LOW_SIZE,
            'vq_low_limit': VQ_LOW_LIMIT,
            'vq_high_type': VQ_HIGH_TYPE,
            'vq_high_size': VQ_HIGH_SIZE,
            'vq_high_limit': VQ_HIGH_LIMIT,
            'vq_low_enable_token_shuffle': VQ_LOW_ENABLE_TOKEN_SHUFFLE,
            'vq_low_enable_layernorm_trick': VQ_LOW_ENABLE_LAYERNORM_TRICK,
            'vq_high_enable_token_shuffle': VQ_HIGH_ENABLE_TOKEN_SHUFFLE,
            'vq_high_enable_layernorm_trick': VQ_HIGH_ENABLE_LAYERNORM_TRICK,
            'enable_movq_decoder': ENABLE_MOVQ_DECODER,
            'movq_type': MOVQ_TYPE,
            'movq_preprocess_embed_dim': MOVQ_PREPROCESS_EMBED_DIM,
            'movq_preprocess_type': MOVQ_PREPROCESS_TYPE,
            'movq_plugin_position': MOVQ_PLUGIN_POSITION,
            'movq_disable_perceptual_loss': MOVQ_DISABLE_PERCEPTUAL_LOSS,
            'return_feat_rec_loss': RETURN_FEAT_REC_LOSS,
            'return_cls_distill_loss': TRAIN_CLS_TOKEN,
            'cls_distill_feature_type': CLS_DISTILL_FEATURE_TYPE,
            'vq_low_preprocess_type': VQ_LOW_PREPROCESS_TYPE,
            'vq_high_preprocess_type': VQ_HIGH_PREPROCESS_TYPE,
            'vq_low_postprocess_type': VQ_LOW_POSTPROCESS_TYPE,
            'vq_high_postprocess_type': VQ_HIGH_POSTPROCESS_TYPE,
            'mllm_feature_type': MLLM_FEATURE_TYPE,
            'use_first_code_for_rec': USE_FIRST_CODE_FOR_REC,
            'extra_sa_before_map': EXTRA_SA_BEFORE_MAP
        }

        print('Creating SigLIP Vision Transformer with additional args:')
        for _k, _v in viq_args.items():
            print(f'    {_k}: {_v}')


        if VIT_WITH_GRAD:
            # siglip2_giant_patch16_384
            self.vision_tower = create_siglip_vit(**viq_args, gradient_checkpointing=True) # gradient_checkpointing=False)
            self.vision_tower.train()
        else:
            self.vision_tower = create_siglip_vit(**viq_args, gradient_checkpointing=False)
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            self.vision_tower.eval()

        print("Loading teacher model...")
        if USE_STAGE1_AS_TEACHER:
            teacher_args = {
                'ckpt_path': TEACHER_CKPT,
                'model_name': 'siglip2_giant_patch16_384',
                'vq_low_type': TEACHER_VQ_LOW_TYPE,
                'vq_low_size': TEACHER_VQ_LOW_SIZE,
                'vq_low_limit': TEACHER_VQ_LOW_LIMIT,
                'vq_high_type': TEACHER_VQ_HIGH_TYPE,
                'vq_high_size': TEACHER_VQ_HIGH_SIZE,
                'vq_high_limit': TEACHER_VQ_HIGH_LIMIT,
                'vq_low_enable_token_shuffle': TEACHER_VQ_LOW_ENABLE_TOKEN_SHUFFLE,
                'vq_low_enable_layernorm_trick': TEACHER_VQ_LOW_ENABLE_LAYERNORM_TRICK,
                'vq_high_enable_token_shuffle': TEACHER_VQ_HIGH_ENABLE_TOKEN_SHUFFLE,
                'vq_high_enable_layernorm_trick': TEACHER_VQ_HIGH_ENABLE_LAYERNORM_TRICK,
                'enable_movq_decoder': TEACHER_ENABLE_MOVQ_DECODER,
                'return_feat_rec_loss': TEACHER_RETURN_FEAT_REC_LOSS,
                'return_cls_distill_loss': TEACHER_RETURN_CLS_DISTILL_LOSS,
                'vq_low_preprocess_type': TEACHER_VQ_LOW_PREPROCESS_TYPE,
                'vq_high_preprocess_type': TEACHER_VQ_HIGH_PREPROCESS_TYPE,
                'vq_low_postprocess_type': TEACHER_VQ_LOW_POSTPROCESS_TYPE,
                'vq_high_postprocess_type': TEACHER_VQ_HIGH_POSTPROCESS_TYPE,
                'mllm_feature_type': TEACHER_MLLM_FEATURE_TYPE,
                'cls_distill_feature_type': TEACHER_CLS_DISTILL_FEATURE_TYPE,
                'use_first_code_for_rec': TEACHER_USE_FIRST_CODE_FOR_REC,
                'extra_sa_before_map': TEACHER_EXTRA_SA_BEFORE_MAP
            }
            print('Creating SigLIP Vision Transformer Teacher with additional args:')
            for _k, _v in teacher_args.items():
                print(f'    {_k}: {_v}')
            self.vision_tower_teacher = create_siglip_vit(**teacher_args, gradient_checkpointing=False, teacher=False)
        else:
            # SigLIP2 open_clip weights, e.g.:
            #   SO400M:  <weights>/siglip2_so400m_16_384/open_clip_pytorch_model.bin
            #   g:       <weights>/siglip2_g_16_384/open_clip_pytorch_model.bin
            self.vision_tower_teacher = create_siglip_vit_oryx2(
                ckpt_path='/home/models/open_clip_pytorch_model.bin',
                model_name=model_name,
                gradient_checkpointing=False, 
                teacher=True
            )
        for p in self.vision_tower_teacher.parameters():
            p.requires_grad = False
        self.vision_tower_teacher.eval()
        self.is_loaded = True

    def train(self, mode = True):
        self.training = mode

        if self.is_loaded and not VIT_WITH_GRAD:
            self.vision_tower.eval()

        if hasattr(self.vision_tower, 'movq_head'):
            if hasattr(self.vision_tower.movq_head, 'perceptual_loss'):
                self.vision_tower.movq_head.perceptual_loss.eval()
            if hasattr(self.vision_tower.movq_head, 'vae'):
                self.vision_tower.movq_head.vae.eval()


    def forward_func(self, images, force_fix_size=False, cal_attn_pool=False):
        kept_ratios = 0.0
        kept_ratios_first_code = 0.0
        self.forward_times += 1
        if type(images) is list:
            if force_fix_size:
                xs = [x.to(self.dtype) for x in images]
                xs = torch.cat([F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False) for x in xs], dim=0)
                image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)

                image_features = torch.split(image_features, 1, dim=0)
                img_size = [img_size] * len(images)
            else:
                xs = [x.to(self.dtype) for x in images]
                image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, info = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)
                if self.forward_times % 10 == 0:
                    # no need to see it here. or
                    codes1 = info.get('indices', None)
                    if codes1 is not None:
                        with torch.no_grad():
                            xs_noised = [x + torch.randn_like(x) * 0.01 for x in xs]
                            _, _, _, _, _, _, _, info2 = self.vision_tower(xs_noised, cal_attn_pool=cal_attn_pool)
                        codes2 = info2.get('indices', None)
                    else:
                        codes2 = None
                    
                    if codes1 is not None and codes2 is not None:
                        kept_ratios = torch.tensor((codes1.detach().clone() == codes2.detach().clone()).sum() / torch.ones_like(codes2.detach().clone()).sum()).to(codes1.device)
                    else:
                        kept_ratios = torch.tensor(0.0).to(image_features[0].device)

                    if codes1 is not None and codes2 is not None:
                        kept_ratios_first_code = torch.tensor((codes1[:, ::4].detach().clone() == codes2[:, ::4].detach().clone()).sum() / torch.ones_like(codes2[:, ::4].detach().clone()).sum()).to(codes1.device)
                    else:
                        kept_ratios_first_code = torch.tensor(0.0).to(image_features[0].device)


                else:
                    kept_ratios = torch.tensor(0.0).to(image_features[0].device)
                    kept_ratios_first_code = torch.tensor(0.0).to(image_features[0].device)
        else:
            raise

        return image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code
    
    def forward(self, images, cal_attn_pool=False):
        if VIT_WITH_GRAD:
            image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code  = self.forward_func(images, cal_attn_pool=cal_attn_pool)
            if cls_token is not None:
                return image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code
            else:
                return image_features, img_size, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code
            
        else:
            with torch.no_grad():
                image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code  = self.forward_func(images, cal_attn_pool=cal_attn_pool)
                if cls_token is not None:
                    return image_features, img_size, cls_token, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code
                else:
                    return image_features, img_size, rec_loss_list, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code
    
    def forward_attn_pool(self, images_features):
        images_features = self.vision_tower.forward_head(images_features)
        return images_features

    def forward_teacher(self, images):
        with torch.no_grad():
            xs = [x.to(self.dtype) for x in images]
            dtype = xs[0].dtype
            xs = torch.cat([F.interpolate(x.float(), size=(384, 384), mode='bilinear', align_corners=False).to(dtype) for x in xs], dim=0)
            if USE_STAGE1_AS_TEACHER:
                cls_token = self.vision_tower_teacher.forward_teacher(xs)
                image_features = None
                img_size = None
            else:
                image_features, img_size, cls_token = self.vision_tower_teacher(xs, cal_attn_pool=True)
            image_features = torch.split(image_features, 1, dim=0) if image_features is not None else image_features
            img_size = [img_size] * len(images)
            return image_features, img_size, cls_token

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
            # 'image_size': 224,
            'patch_size': 16,
        })()
