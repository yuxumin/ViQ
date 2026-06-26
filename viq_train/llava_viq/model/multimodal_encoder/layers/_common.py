"""Shared building blocks for the viq vision encoder.

Kept separate to avoid circular imports between the encoder
(``siglip_vit_anyres_viq``), the dual-VQ head (``dual_vq_head``) and the
VAE heads (``vae_heads``), all of which reuse ``ResualMLP``.
"""
import torch
import torch.nn as nn

try:
    from timm.layers import Mlp, DropPath
except Exception:
    Mlp = None
    DropPath = None


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
