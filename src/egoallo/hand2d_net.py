from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal, assert_never

import numpy as np
import torch
import copy
from einops import rearrange
from jaxtyping import Bool, Float
from loguru import logger
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn
from manopth.manolayer import ManoLayer
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3

class PatchFeatToLatent(nn.Module):
    """(B, 768, 16, 16) → (B, 8, 16, 16)  纯 1×1 通道压缩"""

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 8,
        activation: str = "gelu",
    ):
        super().__init__()
        Activation = nn.GELU if activation == "gelu" else nn.ReLU
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            Activation(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: Float[Tensor, "batch_time 768 16 16"],
    ) -> Float[Tensor, "batch_time 8 16 16"]:
        return self.net(x)


class LatentCNNEncoder(nn.Module):
    """PatchFeat 768x16x16 -> 64 spatial tokens + global feature."""

    def __init__(
        self,
        patch_feat_channels: int = 768,
        latent_channels: int = 8,
        d_out: int = 256,
        global_feat_dim: int = 128,
        activation: str = "gelu",
    ):
        super().__init__()
        Activation = nn.GELU if activation == "gelu" else nn.ReLU

        # 768×16×16 → 8×16×16
        self.patch_to_latent = PatchFeatToLatent(
            in_channels=patch_feat_channels,
            out_channels=latent_channels,
            activation=activation,
        )

        # 8×16×16 → d_out×8×8
        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.GroupNorm(8, 64),
            Activation(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),              # 8→8
            nn.GroupNorm(8, 128),
            Activation(),
            nn.Conv2d(128, d_out, kernel_size=3, stride=1, padding=1),           # 8→8
            nn.GroupNorm(8, d_out),
            Activation(),
        )

        self.num_tokens = 8 * 8  # = 64
        self.d_out = d_out
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, d_out) * 0.02
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(d_out, d_out),
            Activation(),
            nn.Linear(d_out, global_feat_dim),
        )

    def forward(
        self,
        patch_feat: Float[Tensor, "batch_time 768 16 16"],
    ) -> tuple[
        Float[Tensor, "batch_time num_tokens d_out"],
        Float[Tensor, "batch_time global_feat_dim"],
    ]:
        latent = self.patch_to_latent(patch_feat)     # (B*T, 8, 16, 16)
        feat = self.conv_layers(latent)               # (B*T, d_out, 8, 8)

        global_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)  # (B*T, d_out)
        global_feat = self.global_proj(global_feat)                   # (B*T, global_feat_dim)
        tokens = feat.flatten(2).transpose(1, 2)      # (B*T, 64, d_out)
        tokens = tokens + self.spatial_pos_embed

        return tokens, global_feat


class Pose2DTransformerDecoder(nn.Module):
    def __init__(
        self,
        patch_feat_channels: int = 768,
        latent_channels: int = 8,
        d_latent: int = 256,
        num_joints: int = 21,
        num_layers: int = 4,
        num_heads: int = 8,
        d_feedforward: int = 1024,
        dropout_p: float = 0.1,
        global_feat_dim: int = 128,
        activation: str = "gelu",
        d_cond: int = 7,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.d_latent = d_latent
        self.global_feat_dim = global_feat_dim

        # 替换原来的 image_proj + spatial_pos_embed
        self.latent_cnn = LatentCNNEncoder(
            patch_feat_channels=patch_feat_channels,
            latent_channels=latent_channels,
            d_out=d_latent,
            global_feat_dim=global_feat_dim,
            activation=activation,
        )

        self.joint_queries = nn.Parameter(
            torch.randn(1, num_joints, d_latent) * 0.02
        )

        Activation = nn.GELU if activation == "gelu" else nn.ReLU
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_latent,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout_p,
            activation=(nn.functional.gelu if activation == "gelu" else nn.functional.relu),
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_latent)

        self.pose_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            Activation(),
            nn.Linear(d_latent, 1),
            nn.Sigmoid(),
        )

        # Project noisy 2D joints into query space for flow matching denoising.
        self.noisy_joint_proj = nn.Linear(2, d_latent)

        # Project conditioning vector into query space.
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_latent),
            Activation(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        img_patch_feat: Float[Tensor, "batch time 768 16 16"],
        noisy_joint_2d: Float[Tensor, "batch time num_joints 2"] | None = None,
        cond: Float[Tensor, "batch_time d_cond"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch time num_joints 2"],
        Float[Tensor, "batch time num_joints 1"],
        Float[Tensor, "batch time num_joints d_latent"],
        Float[Tensor, "batch time global_feat_dim"],
    ]:
        batch, time = img_patch_feat.shape[:2]
        x = img_patch_feat.reshape(batch * time, *img_patch_feat.shape[2:])

        tokens, global_feat = self.latent_cnn(x)
        # tokens: (B*T, 64, d_latent),  global_feat: (B*T, global_feat_dim)
        global_feat = global_feat.reshape(batch, time, self.global_feat_dim)
        queries = self.joint_queries.expand(batch * time, -1, -1)

        # Condition queries on noisy 2D joints (zeros when not in flow matching mode
        # to keep noisy_joint_proj parameters in the computation graph for DDP).
        if noisy_joint_2d is None:
            noisy_joint_2d = torch.zeros(batch, time, self.num_joints, 2, device=queries.device)
        noisy_flat = noisy_joint_2d.reshape(batch * time, self.num_joints, 2)
        queries = queries + self.noisy_joint_proj(noisy_flat)

        # Condition on noise level / timestep embedding.
        if cond is not None:
            # cond: (B*T, d_cond) -> broadcast to all joints as additive bias
            cond_bias = self.cond_proj(cond)  # (B*T, d_latent)
            queries = queries + cond_bias[:, None, :]  # (B*T, num_joints, d_latent)

        for layer in self.layers:
            queries = layer(queries, tokens)

        queries = self.final_norm(queries)

        pose_2d = self.pose_head(queries)
        confidence = self.confidence_head(queries)

        pose_2d = pose_2d.reshape(batch, time, self.num_joints, 2)
        confidence = confidence.reshape(batch, time, self.num_joints, 1)
        joint_feat = queries.reshape(batch, time, self.num_joints, self.d_latent)
        

        return pose_2d, confidence, joint_feat, global_feat