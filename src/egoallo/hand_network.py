from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import Literal, assert_never

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float
from loguru import logger
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn

from manopth.manolayer import ManoLayer
from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


def project_rotmats_via_svd(
    rotmats: Float[Tensor, "*batch 3 3"],
) -> Float[Tensor, "*batch 3 3"]:
    u, s, vh = torch.linalg.svd(rotmats)
    del s
    return torch.einsum("...ij,...jk->...ik", u, vh)


class HandDenoiseTraj(TensorDataclass):
    """Data structure for denoising. Contains tensors that we are denoising, as
    well as utilities for packing + unpacking them."""

    mano_betas: Float[Tensor, "*#batch timesteps 10"]
    """Hand shape parameters. We don't really need the timesteps axis here,
    it's just for convenience."""

    mano_poses: Float[Tensor, "*#batch timesteps 15 3"]
    """Relative poses for each hand joint."""

    global_orientation: Float[Tensor, "*#batch timesteps 3"]
    """Global Orientation for hand."""

    camera_pose: Float[Tensor, "*#batch timesteps 3"]
    """Global Camera translation"""

    mano_side: Float[Tensor, "*#batch timesteps 1"]
    """right or left hand"""
    @staticmethod
    def get_packed_dim() -> int:
        packed_dim = 10 + 15 * 3 + 3 + 3 + 1
        return packed_dim

    def apply_to_hand(self, mano_model: ManoLayer) -> tuple[torch.Tensor, torch.Tensor]:
        mano_layer = ManoLayer(
            flat_hand_mean=False,
            ncomps=45,
            side=mano_side_str[(int)(self.mano_side.numpy())],
            mano_root='/public/home/group_ucb/yunqili/code/dex-ycb-toolkit/manopth/mano/models',
        )
        vertices, joints = mano_layer(
            torch.cat((self.mano_poses,self.global_orientation),dim=-1),
            self.mano_betas.unsqueeze(0).repeat(self.mano_poses.shape[0], 1),
            self.camera_pose
        )
        vertices = vertices/ 1000  # Convert to meters
        faces_m = mano_layer.th_faces
        return vertices, faces_m

    def pack(self) -> Float[Tensor, "*#batch timesteps d_state"]:
        """Pack trajectory into a single flattened vector."""
        (*batch, time, num_joints, _) = self.mano_poses.shape
        assert num_joints == 15
        return torch.cat(
            [
                x.reshape((*batch, time, -1))
                for x in vars(self).values()
                if x is not None
            ],
            dim=-1,
        )

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*#batch timesteps d_state"],
        project_rotmats: bool = False,
    ) -> HandDenoiseTraj:
        """Unpack trajectory from a single flattened vector.

        Args:
            x: Packed trajectory.
            project_rotmats: If True, project the rotation matrices to SO(3) via SVD.
        """
        (*batch, time, d_state) = x.shape
        assert d_state == cls.get_packed_dim()

        mano_betas, mano_poses, global_orientation,camera_pose,mano_side = torch.split(
            x, [10, 15 * 3, 3, 3, 1], dim=-1
        )
        mano_poses = mano_poses.reshape((*batch, time, 15, 3))
        hand_rotmats = None
        assert mano_betas.shape == (*batch, time, 10)

        # if project_rotmats:
        #     # We might want to handle the -1 determinant case as well.
        #     hand = project_rotmats_via_svd(body_rotmats)

        return HandDenoiseTraj(
            mano_betas=mano_betas,
            mano_poses=mano_poses,
            global_orientation=global_orientation,
            camera_pose=camera_pose,
            mano_side=mano_side
        )


@dataclass(frozen=True)
class HandDenoiserConfig:
    max_t: int = 1000
    fourier_enc_freqs: int = 3
    d_latent: int = 512
    d_feedforward: int = 2048
    d_noise_emb: int = 1024
    num_heads: int = 4
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout_p: float = 0.0
    activation: Literal["gelu", "relu"] = "gelu"

    positional_encoding: Literal["transformer", "rope"] = "rope"
    noise_conditioning: Literal["token", "film"] = "token"

    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"] = (
        "kv_from_cond_q_from_x"
    )

    include_canonicalized_cpf_rotation_in_cond: bool = True

    cond_param: Literal[
        "ours", "canonicalized", "absolute", "absrel", "absrel_global_deltas"
    ] = "ours"
    """Which conditioning parameterization to use.

    "ours" is the default, we try to be clever and design something with nice
        equivariance properties.
    "canonicalized" contains a transformation that's canonicalized to aligned
        to the first frame.
    "absolute" is the naive case, where we just pass in transformations
        directly.
    """

    @cached_property
    def d_cond(self) -> int:
        """Dimensionality of conditioning vector."""

        if self.cond_param == "ours":
            d_cond = 0
            d_cond += 3 ## only palm position
        elif self.cond_param == "canonicalized":
            d_cond = 12
        elif self.cond_param == "absolute":
            d_cond = 12
        elif self.cond_param == "absrel":
            # Both absolute and relative!
            d_cond = 24
        elif self.cond_param == "absrel_global_deltas":
            # Both absolute and relative!
            d_cond = 24
        else:
            assert_never(self.cond_param)

        # Add two 3D positions to the conditioning dimension if we're including
        # hand conditioning.

        d_cond = d_cond + d_cond * self.fourier_enc_freqs * 2  # Fourier encoding.
        return d_cond

    def make_cond(
        self,
        rel_palm_pose: Float[Tensor, "batch time 3"], # relative palm pose to the camera
    ) -> Float[Tensor, "batch time d_cond"]:
        """Construct conditioning information from CPF pose."""
        (batch, time, _) = rel_palm_pose.shape

        # Construct device pose conditioning.
        if self.cond_param == "ours":
            cond = rel_palm_pose
        else:
            assert_never(self.cond_param)
        cond = fourier_encode(cond, freqs=self.fourier_enc_freqs)
        return cond


class HandDenoiser(nn.Module):
    """Denoising network for human motion.

    Inputs are noisy trajectory, conditioning information, and timestep.
    Output is denoised trajectory.
    """

    def __init__(self, config: HandDenoiserConfig):
        super().__init__()

        self.config = config
        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        # MLP encoders and decoders for each modality we want to denoise.
        modality_dims: dict[str, int] = {
            "mano_betas": 10,
            "mano_poses": 15 * 3,
            "global_orientation": 3,
            "camera_pose":3,
            "mano_side":1
        }

        assert sum(modality_dims.values()) == self.get_d_state()
        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(modality_dim, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                )
                for k, modality_dim in modality_dims.items()
            }
        )
        self.decoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(config.d_latent, config.d_latent),
                    nn.LayerNorm(normalized_shape=config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, config.d_latent),
                    Activation(),
                    nn.Linear(config.d_latent, modality_dim),
                )
                for k, modality_dim in modality_dims.items()
            }
        )

        # Helpers for converting between input dimensionality and latent dimensionality.
        self.latent_from_cond = nn.Linear(config.d_cond, config.d_latent)

        # Noise embedder.
        self.noise_emb = nn.Embedding(
            # index 0 will be t=1
            # index 999 will be t=1000
            num_embeddings=config.max_t,
            embedding_dim=config.d_noise_emb,
        )
        self.noise_emb_token_proj = (
            nn.Linear(config.d_noise_emb, config.d_latent, bias=False)
            if config.noise_conditioning == "token"
            else None
        )

        # Encoder / decoder layers.
        # Inputs are conditioning (current noise level, observations); output
        # is encoded conditioning information.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=False,  # No conditioning for encoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        d_latent=config.d_latent,
                        d_noise_emb=config.d_noise_emb,
                        d_feedforward=config.d_feedforward,
                        n_heads=config.num_heads,
                        dropout_p=config.dropout_p,
                        activation=config.activation,
                        include_xattn=True,  # Include conditioning for the decoder.
                        use_rope_embedding=config.positional_encoding == "rope",
                        use_film_noise_conditioning=config.noise_conditioning == "film",
                        xattn_mode=config.xattn_mode,
                    )
                )
                for _ in range(config.decoder_layers)
            ]
        )

    def get_d_state(self) -> int:
        return HandDenoiseTraj.get_packed_dim()

    def forward(
        self,
        x_t_packed: Float[Tensor, "batch time state_dim"],
        t: Float[Tensor, "batch"],
        *,
        rel_palm_pose: Float[Tensor, "batch time 3"],
        project_output_rotmats: bool,
        # Observed hand positions, relative to the CPF.
        mask: Bool[Tensor, "batch time"] | None,
        # Mask for when to drop out / keep conditioning information.
        cond_dropout_keep_mask: Bool[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time state_dim"]:
        """Predict a denoised trajectory. Note that `t` refers to a noise
        level, not a timestep."""
        config = self.config

        x_t = HandDenoiseTraj.unpack(x_t_packed)
        (batch, time, num_hand_joints, _) = x_t.mano_poses.shape
        assert num_hand_joints == 15

        # Encode the trajectory into a single vector per timestep.
        x_t_encoded = (
            self.encoders["mano_betas"](x_t.mano_betas.reshape((batch, time, -1)))
            + self.encoders["mano_poses"](x_t.mano_poses.reshape((batch, time, -1)))
            + self.encoders["global_orientation"](x_t.global_orientation)
            + self.encoders["camera_pose"](x_t.camera_pose)
            + self.encoders["mano_side"](x_t.mano_side)
        )
        assert x_t_encoded.shape == (batch, time, config.d_latent)

        # Embed the diffusion noise level.
        assert t.shape == (batch,)
        noise_emb = self.noise_emb(t - 1)
        assert noise_emb.shape == (batch, config.d_noise_emb)

        # Prepare conditioning information.
        cond = config.make_cond(
            rel_palm_pose = rel_palm_pose
        )

        # Randomly drop out conditioning information; this serves as a
        # regularizer that aims to improve sample diversity.
        if cond_dropout_keep_mask is not None:
            assert cond_dropout_keep_mask.shape == (batch,)
            cond = cond * cond_dropout_keep_mask[:, None, None]

        # Prepare encoder and decoder inputs.
        if config.positional_encoding == "rope":
            pos_enc = 0
        elif config.positional_encoding == "transformer":
            pos_enc = make_positional_encoding(
                d_latent=config.d_latent,
                length=time,
                dtype=cond.dtype,
            )[None, ...].to(x_t_encoded.device)
            assert pos_enc.shape == (1, time, config.d_latent)
        else:
            assert_never(config.positional_encoding)

        encoder_out = self.latent_from_cond(cond) + pos_enc
        decoder_out = x_t_encoded + pos_enc

        # Append the noise embedding to the encoder and decoder inputs.
        # This is weird if we're using rotary embeddings!
        if self.noise_emb_token_proj is not None:
            noise_emb_token = self.noise_emb_token_proj(noise_emb)
            assert noise_emb_token.shape == (batch, config.d_latent)
            encoder_out = torch.cat([noise_emb_token[:, None, :], encoder_out], dim=1)
            decoder_out = torch.cat([noise_emb_token[:, None, :], decoder_out], dim=1)
            assert (
                encoder_out.shape
                == decoder_out.shape
                == (batch, time + 1, config.d_latent)
            )
            num_tokens = time + 1
        else:
            num_tokens = time
        # print("mask is",mask,"and mask type is",mask.shape,batch,time)
        # Compute attention mask. This needs to be a fl
        if mask is None:
            attn_mask = None
        else:
            assert mask.shape == (batch, time)
            assert mask.dtype == torch.bool
            if self.noise_emb_token_proj is not None:  # Account for noise token.
                mask = torch.cat([mask.new_ones((batch, 1)), mask], dim=1)
            # Last two dimensions of mask are (query, key). We're masking out only keys;
            # it's annoying for the softmax to mask out entire rows without getting NaNs.
            attn_mask = mask[:, None, None, :].repeat(1, 1, num_tokens, 1)
            assert attn_mask.shape == (batch, 1, num_tokens, num_tokens)
            assert attn_mask.dtype == torch.bool

        # Forward pass through transformer.
        for layer in self.encoder_layers:
            encoder_out = layer(encoder_out, attn_mask, noise_emb=noise_emb)
        for layer in self.decoder_layers:
            decoder_out = layer(
                decoder_out, attn_mask, noise_emb=noise_emb, cond=encoder_out
            )

        # Remove the extra token corresponding to the noise embedding.
        if self.noise_emb_token_proj is not None:
            decoder_out = decoder_out[:, 1:, :]
        assert isinstance(decoder_out, Tensor)
        assert decoder_out.shape == (batch, time, config.d_latent)

        packed_output = torch.cat(
            [
                # Project rotation matrices for body_rotmats via SVD,
                (
                    modality_decoder(decoder_out).reshape((batch,time,15 * 3))
                    # if enabled,
                    if key in ("mano_poses")
                    # otherwise, just decode normally.
                    else modality_decoder(decoder_out)
                )
                for key, modality_decoder in self.decoders.items()
            ],
            dim=-1,
        )
        assert packed_output.shape == (batch, time, self.get_d_state())

        # Return packed output.
        return packed_output


@cache
def make_positional_encoding(
    d_latent: int, length: int, dtype: torch.dtype
) -> Float[Tensor, "length d_latent"]:
    """Computes standard Transformer positional encoding."""
    pe = torch.zeros(length, d_latent, dtype=dtype)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_latent, 2).float() * (-np.log(10000.0) / d_latent)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    assert pe.shape == (length, d_latent)
    return pe


def fourier_encode(
    x: Float[Tensor, "*#batch channels"], freqs: int
) -> Float[Tensor, "*#batch channels+2*freqs*channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs))
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class TransformerBlockConfig:
    d_latent: int
    d_noise_emb: int
    d_feedforward: int
    n_heads: int
    dropout_p: float
    activation: Literal["gelu", "relu"]
    include_xattn: bool
    use_rope_embedding: bool
    use_film_noise_conditioning: bool
    xattn_mode: Literal["kv_from_cond_q_from_x", "kv_from_x_q_from_cond"]


class TransformerBlock(nn.Module):
    """An even-tempered Transformer block."""

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.sattn_qkv_proj = nn.Linear(
            config.d_latent, config.d_latent * 3, bias=False
        )
        self.sattn_out_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)

        self.layernorm1 = nn.LayerNorm(config.d_latent)
        self.layernorm2 = nn.LayerNorm(config.d_latent)

        assert config.d_latent % config.n_heads == 0
        self.rotary_emb = (
            RotaryEmbedding(config.d_latent // config.n_heads)
            if config.use_rope_embedding
            else None
        )

        if config.include_xattn:
            self.xattn_kv_proj = nn.Linear(
                config.d_latent, config.d_latent * 2, bias=False
            )
            self.xattn_q_proj = nn.Linear(config.d_latent, config.d_latent, bias=False)
            self.xattn_layernorm = nn.LayerNorm(config.d_latent)
            self.xattn_out_proj = nn.Linear(
                config.d_latent, config.d_latent, bias=False
            )

        self.norm_no_learnable = nn.LayerNorm(
            config.d_feedforward, elementwise_affine=False, bias=False
        )
        self.activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]()
        self.dropout = nn.Dropout(config.dropout_p)

        self.mlp0 = nn.Linear(config.d_latent, config.d_feedforward)
        self.mlp_film_cond_proj = (
            zero_module(
                nn.Linear(config.d_noise_emb, config.d_feedforward * 2, bias=False)
            )
            if config.use_film_noise_conditioning
            else None
        )
        self.mlp1 = nn.Linear(config.d_feedforward, config.d_latent)
        self.config = config

    def forward(
        self,
        x: Float[Tensor, "batch tokens d_latent"],
        attn_mask: Bool[Tensor, "batch 1 tokens tokens"] | None,
        noise_emb: Float[Tensor, "batch d_noise_emb"],
        cond: Float[Tensor, "batch tokens d_latent"] | None = None,
    ) -> Float[Tensor, "batch tokens d_latent"]:
        config = self.config
        (batch, time, d_latent) = x.shape

        # Self-attention.
        # We put layer normalization after the residual connection.
        x = self.layernorm1(x + self._sattn(x, attn_mask))

        # Include conditioning.
        if config.include_xattn:
            assert cond is not None
            x = self.xattn_layernorm(x + self._xattn(x, attn_mask, cond=cond))

        mlp_out = x
        mlp_out = self.mlp0(mlp_out)
        mlp_out = self.activation(mlp_out)

        # FiLM-style conditioning.
        if self.mlp_film_cond_proj is not None:
            scale, shift = torch.chunk(
                self.mlp_film_cond_proj(noise_emb), chunks=2, dim=-1
            )
            assert scale.shape == shift.shape == (batch, config.d_feedforward)
            mlp_out = (
                self.norm_no_learnable(mlp_out) * (1.0 + scale[:, None, :])
                + shift[:, None, :]
            )

        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp1(mlp_out)

        x = self.layernorm2(x + mlp_out)
        assert x.shape == (batch, time, d_latent)
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        """Multi-head self-attention."""
        config = self.config
        q, k, v = rearrange(
            self.sattn_qkv_proj(x),
            "b t (qkv nh dh) -> qkv b nh t dh",
            qkv=3,
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p
        )
        x = self.dropout(x)
        x = rearrange(x, "b nh t dh -> b t (nh dh)", nh=config.n_heads)
        x = self.sattn_out_proj(x)
        return x

    def _xattn(self, x: Tensor, attn_mask: Tensor | None, cond: Tensor) -> Tensor:
        """Multi-head cross-attention."""
        config = self.config
        k, v = rearrange(
            self.xattn_kv_proj(
                {
                    "kv_from_cond_q_from_x": cond,
                    "kv_from_x_q_from_cond": x,
                }[self.config.xattn_mode]
            ),
            "b t (qk nh dh) -> qk b nh t dh",
            qk=2,
            nh=config.n_heads,
        )
        q = rearrange(
            self.xattn_q_proj(
                {
                    "kv_from_cond_q_from_x": x,
                    "kv_from_x_q_from_cond": cond,
                }[self.config.xattn_mode]
            ),
            "b t (nh dh) -> b nh t dh",
            nh=config.n_heads,
        )
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=config.dropout_p, attn_mask=attn_mask
        )
        x = rearrange(x, "b nh t dh -> b t (nh dh)")
        x = self.xattn_out_proj(x)

        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
