"""Official-configuration Primus-M encoder adapted for ABUS classification.

The encoder layout and parameter names follow the Apache-2.0 Primus-M code in
``constantinulrich/SSL3D_classification`` and the EVA building block in the
Apache-2.0 ``MIC-DKFZ/dynamic-network-architectures`` project. Keeping those names is important:
OpenMind checkpoints store the pretrained encoder below the ``encoder.``
prefix.

The OpenMind checkpoint is distributed separately under CC-BY-4.0.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.layers.pos_embed_sincos import build_rotary_pos_embed
from timm.models.eva import EvaBlock
from torch import nn
from torch.utils.checkpoint import checkpoint


OPENMIND_SIMCLR_SHA256 = (
    "39bb8cc0a5b806d19b97791d676beb450a9114ec6d8031bb412614f606ebee70"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load old and new PyTorch checkpoints without requiring torch>=2."""
    checkpoint_path = Path(path)
    try:
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except pickle.UnpicklingError as error:
        observed_sha256 = _sha256(checkpoint_path)
        if observed_sha256 != OPENMIND_SIMCLR_SHA256:
            raise RuntimeError(
                "Refusing unsafe Primus-M checkpoint fallback because SHA256 does not "
                f"match the pinned official artifact: {observed_sha256}"
            ) from error
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    except TypeError:  # torch 1.13 has no weights_only argument.
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_data, dict):
        raise RuntimeError("Primus-M checkpoint must be a dictionary.")
    return checkpoint_data


class _PatchEmbed3D(nn.Module):
    """Official Primus-M patch projection with matching state-dict keys."""

    def __init__(self, patch_size: tuple[int, int, int], in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _RotaryEmbeddingCat3DCompat(nn.Module):
    """Fixed-shape 3D RoPE without timm 1.0.26's 2D buffer assumption.

    ``RotaryEmbeddingCat`` allocates ``dim * 2`` values per token, which is
    correct for a 2D grid.  ``build_rotary_pos_embed`` correctly emits
    ``dim * 3`` values for a 3D grid, so timm 1.0.26 then fails while copying
    the generated tensor into its undersized buffer.  Primus-M uses a fixed
    token grid; directly registering the correctly generated tensor preserves
    the official RoPE calculation without changing the globally installed
    timm package.
    """

    def __init__(
        self,
        dim: int,
        *,
        feat_shape: tuple[int, int, int],
        ref_feat_shape: tuple[int, int, int],
        temperature: float = 10000.0,
        grid_offset: float = 0.0,
        grid_indexing: str = "ij",
    ):
        super().__init__()
        sin_embed, cos_embed = build_rotary_pos_embed(
            feat_shape=list(feat_shape),
            dim=dim,
            temperature=temperature,
            linear_bands=False,
            in_pixels=False,
            ref_feat_shape=list(ref_feat_shape),
            grid_offset=grid_offset,
            grid_indexing=grid_indexing,
            dtype=torch.float32,
        )
        self.register_buffer(
            "pos_embed",
            torch.cat((sin_embed, cos_embed), dim=-1),
            persistent=False,
        )

    def get_embed(self) -> torch.Tensor:
        return self.pos_embed


class _PrimusEva(nn.Module):
    """EVA02 encoder used by Primus-M, with a torch-1.13-safe checkpoint path."""

    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ref_feat_shape: tuple[int, int, int],
        mlp_ratio: float,
        use_abs_pos_emb: bool,
        use_rot_pos_emb: bool,
        drop_path_rate: float,
        proj_drop_rate: float,
        attn_drop_rate: float,
        grad_checkpointing: bool,
        block_fn: Callable[..., nn.Module] = EvaBlock,
    ):
        super().__init__()
        self.grad_checkpointing = bool(grad_checkpointing)
        self.num_prefix_tokens = 0
        num_patches = int(np.prod(ref_feat_shape))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            if use_abs_pos_emb
            else None
        )
        self.pos_drop = nn.Dropout(p=0.0)

        if use_rot_pos_emb:
            rope_dim = round(embed_dim // num_heads / 1.5)
            if rope_dim != embed_dim / num_heads / 1.5 or rope_dim % 4 != 0:
                raise ValueError(
                    "Primus-M rotary dimension must be integral and divisible by four."
                )
            self.rope = _RotaryEmbeddingCat3DCompat(
                rope_dim,
                feat_shape=ref_feat_shape,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        drop_path_rates = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=True,
                    qkv_fused=False,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=True,
                    scale_mlp=True,
                    scale_attn_inner=True,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=float(drop_path_rates[index]),
                    norm_layer=nn.LayerNorm,
                    init_values=0.1,
                    num_prefix_tokens=0,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self._rescale_projection_weights()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _rescale_projection_weights(self) -> None:
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.grad_checkpointing = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rotary = self.rope.get_embed() if self.rope is not None else None
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                # torch 1.13 checkpoint() cannot forward keyword arguments.
                def run_block(value: torch.Tensor, current_block=block) -> torch.Tensor:
                    return current_block(value, rope=rotary)

                x = checkpoint(run_block, x)
            else:
                x = block(x, rope=rotary)
        return self.norm(x)


class _PrimusEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        grad_checkpointing: bool,
    ):
        super().__init__()
        ref_feat_shape = tuple(size // patch for size, patch in zip(input_shape, patch_size))
        self.down_projection = _PatchEmbed3D(patch_size, in_chans, embed_dim)
        self.eva = _PrimusEva(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ref_feat_shape=ref_feat_shape,
            mlp_ratio=mlp_ratio,
            use_abs_pos_emb=True,
            use_rot_pos_emb=True,
            drop_path_rate=0.2,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            grad_checkpointing=grad_checkpointing,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_projection(x)
        # Match the official ``b c w h d -> b (h w d) c`` token order.
        x = x.permute(0, 3, 2, 4, 1).reshape(x.shape[0], -1, x.shape[1])
        return self.eva(x)


class PrimusMClassifier(nn.Module):
    """OpenMind-pretrained Primus-M with an average-token classification head."""

    DEFAULT_INPUT_SHAPE = (128, 128, 64)
    DEFAULT_PATCH_SIZE = (8, 8, 8)
    DEFAULT_EMBED_DIM = 864
    DEFAULT_DEPTH = 16
    DEFAULT_NUM_HEADS = 12
    DEFAULT_MLP_RATIO = 2.66666667
    DEFAULT_PRETRAINED_GRID = (8, 8, 8)

    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        *,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        depth: int = DEFAULT_DEPTH,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        dropout: float = 0.2,
        grad_checkpointing: bool = True,
        pretrained_path: str | Path | None = None,
        pretrained_grid: tuple[int, int, int] = DEFAULT_PRETRAINED_GRID,
    ):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.patch_size = tuple(int(value) for value in patch_size)
        if len(self.input_shape) != 3 or len(self.patch_size) != 3:
            raise ValueError("input_shape and patch_size must each contain three dimensions.")
        if any(size % patch != 0 for size, patch in zip(self.input_shape, self.patch_size)):
            raise ValueError("Every Primus-M input dimension must be divisible by its patch size.")
        self.token_grid = tuple(
            size // patch for size, patch in zip(self.input_shape, self.patch_size)
        )
        self.encoder = _PrimusEncoder(
            input_shape=self.input_shape,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            grad_checkpointing=grad_checkpointing,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        self.variant = "primus_m_openmind_simclr"
        self.pretrained_report: dict[str, Any] | None = None
        if pretrained_path is not None:
            self.pretrained_report = self.load_openmind_checkpoint(
                pretrained_path,
                pretrained_grid=pretrained_grid,
            )

    @staticmethod
    def _zscore_each_volume(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    @staticmethod
    def interpolate_position_embedding(
        position_embedding: torch.Tensor,
        source_grid: tuple[int, int, int],
        target_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        expected_tokens = int(np.prod(source_grid))
        if tuple(position_embedding.shape[:2]) != (1, expected_tokens):
            raise RuntimeError(
                "Unexpected Primus-M position embedding shape: "
                f"got {tuple(position_embedding.shape)}, expected [1, {expected_tokens}, C]."
            )
        channels = position_embedding.shape[-1]
        spatial = position_embedding.transpose(1, 2).reshape(1, channels, *source_grid)
        resized = F.interpolate(
            spatial,
            size=target_grid,
            mode="trilinear",
            align_corners=False,
        )
        return resized.flatten(2).transpose(1, 2)

    @staticmethod
    def _strip_checkpoint_prefix(key: str) -> str:
        # nnSSL checkpoints may compose DDP/torch.compile and encoder wrappers
        # in either order (for example ``_orig_mod.encoder.``).
        prefixes = ("module.", "_orig_mod.", "encoder.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    def load_openmind_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        pretrained_grid: tuple[int, int, int] = DEFAULT_PRETRAINED_GRID,
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_data = _load_torch_checkpoint(checkpoint_path)
        source_state = checkpoint_data.get("network_weights")
        if not isinstance(source_state, dict):
            raise RuntimeError("OpenMind checkpoint is missing dictionary key 'network_weights'.")

        target_state = self.encoder.state_dict()
        target_parameter_keys = set(dict(self.encoder.named_parameters()))
        loaded_state: dict[str, torch.Tensor] = {}
        discarded_keys: list[str] = []
        shape_mismatches: list[str] = []
        for original_key, value in source_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = self._strip_checkpoint_prefix(str(original_key))
            if key == "eva.pos_embed":
                value = self.interpolate_position_embedding(
                    value,
                    tuple(int(item) for item in pretrained_grid),
                    self.token_grid,
                )
            if key not in target_state:
                discarded_keys.append(str(original_key))
                continue
            if target_state[key].shape != value.shape:
                shape_mismatches.append(
                    f"{key}: checkpoint={tuple(value.shape)} model={tuple(target_state[key].shape)}"
                )
                continue
            loaded_state[key] = value

        missing_parameters = sorted(target_parameter_keys - set(loaded_state))
        if shape_mismatches or missing_parameters:
            details = []
            if missing_parameters:
                details.append(f"missing parameters={missing_parameters}")
            if shape_mismatches:
                details.append(f"shape mismatches={shape_mismatches}")
            raise RuntimeError("Incompatible Primus-M checkpoint: " + "; ".join(details))

        incompatible = self.encoder.load_state_dict(loaded_state, strict=False)
        missing_after_load = sorted(
            key for key in incompatible.missing_keys if key in target_parameter_keys
        )
        if missing_after_load:
            raise RuntimeError(
                f"Primus-M parameter coverage was incomplete after loading: {missing_after_load}"
            )
        coverage = len(target_parameter_keys & set(loaded_state)) / max(1, len(target_parameter_keys))
        report = {
            "checkpoint": str(checkpoint_path),
            "parameter_key_coverage": coverage,
            "loaded_parameter_keys": len(target_parameter_keys & set(loaded_state)),
            "total_parameter_keys": len(target_parameter_keys),
            "discarded_non_encoder_keys": len(discarded_keys),
            "unexpected_loaded_keys": list(incompatible.unexpected_keys),
            "source_grid": list(pretrained_grid),
            "target_grid": list(self.token_grid),
        }
        print(f"Primus-M OpenMind load report: {report}")
        return report

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = bool(trainable)
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        return [
            {
                "params": list(self.encoder.parameters()),
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.head.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Primus-M expects [B, 1, X, Y, Z], got {tuple(x.shape)}.")
        if tuple(x.shape[2:]) != self.input_shape:
            raise ValueError(
                f"Primus-M expects spatial shape {self.input_shape}, got {tuple(x.shape[2:])}."
            )
        tokens = self.encoder(self._zscore_each_volume(x))
        return self.head(tokens.mean(dim=1))


__all__ = ["PrimusMClassifier"]
