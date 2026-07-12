"""Shared model-config loading + backbone construction for task scripts.

This is the single source of truth for turning ``configs/model/*.yaml``
into a built backbone. Every training/eval script routes spix
construction through the optimized builder
(:func:`create_optimized_vision_rwkv7`, ``use_cpp=True`` by default);
conv/vq/gnn use their canonical builders.

Resolution is external: ``img_size`` from the config is the default, but a
caller may override it (e.g. a resolution sweep). ``-1`` means native
resolution — the model accepts whatever image size enters, without
rescaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from spixrwkv7.models.conv_spixrwkv7 import create_conv_vision_rwkv7
from spixrwkv7.models.vq_rwkv7 import create_vq_rwkv7
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision_rwkv7

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "model"

_VARIANT_PREFIX = {"spix": "", "conv": "conv_", "vq": "vq_", "gnn": "gnn_"}
_VALID_SIZES = ["tiny", "small", "medium", "large"]


def _optimized_spix_builder():
    """Lazy import — pulls in the C++ kernel only when spix is actually built."""
    from spixrwkv7.kernels.optimized_vision import create_optimized_vision_rwkv7

    return create_optimized_vision_rwkv7


def load_model_config(model_type: str, size: str) -> dict[str, Any]:
    """Load ``configs/model/{prefix}{size}.yaml`` and return the ``model`` section."""
    if model_type not in _VARIANT_PREFIX:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    if size not in _VALID_SIZES:
        raise ValueError(f"Unknown size: {size!r} (expected one of {_VALID_SIZES})")
    path = CONFIG_DIR / f"{_VARIANT_PREFIX[model_type]}{size}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)["model"]


def build_backbone(
    model_type: str,
    config: dict[str, Any],
    img_size: int | None = None,
) -> Any:
    """Build a backbone from a loaded config.

    Args:
        model_type: One of ``spix``, ``conv``, ``vq``, ``gnn``.
        config: The ``model`` section of a config YAML.
        img_size: Optional override of ``config["img_size"]`` (e.g. for a
            resolution sweep). ``-1`` keeps native resolution.
    """
    img = config["img_size"] if img_size is None else img_size

    if model_type == "spix":
        return _optimized_spix_builder()(
            img_size=img,
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            num_superpixels=config["num_superpixels"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            init_values=config.get("init_values", 1e-5),
            downsample_factor=config.get("downsample_factor", 16),
            norm_layer=config.get("norm_layer", "rmsnorm"),
            act_layer=config.get("act_layer", "swiglu"),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            use_cpp=config.get("use_cpp", True),
            use_attnres=config.get("use_attnres", False),
            attnres_mode=config.get("attnres_mode", "block"),
            attnres_gate_type=config.get("attnres_gate_type", "bias"),
            attnres_num_blocks=config.get("attnres_num_blocks", 8),
            attnres_recency_bias_init=config.get("attnres_recency_bias_init", 10.0),
            use_jit=config.get("use_jit", False),
        )

    if model_type == "conv":
        return create_conv_vision_rwkv7(
            img_size=img,
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            num_superpixels=config["num_superpixels"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            init_values=config.get("init_values", 1e-5),
            norm_layer=config.get("norm_layer", "rmsnorm"),
            act_layer=config.get("act_layer", "swiglu"),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            use_attnres=config.get("use_attnres", False),
            attnres_mode=config.get("attnres_mode", "block"),
            attnres_gate_type=config.get("attnres_gate_type", "bias"),
            attnres_num_blocks=config.get("attnres_num_blocks", 8),
            attnres_recency_bias_init=config.get("attnres_recency_bias_init", 10.0),
            use_cpp=config.get("use_cpp", False),
            use_jit=config.get("use_jit", False),
            conv_stem_channels=tuple(config.get("conv_stem_channels", [32, 64, 128])),
            conv_stem_kernel_sizes=tuple(config.get("conv_stem_kernel_sizes", [3, 5, 5])),
            conv_stem_strides=tuple(config.get("conv_stem_strides", [1, 2, 2])),
            conv_stem_norm=config.get("conv_stem_norm", "batchnorm2d"),
            conv_post_norm=config.get("conv_post_norm", "layernorm"),
        )

    if model_type == "vq":
        return create_vq_rwkv7(
            img_size=img,
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
            init_values=config.get("init_values", 0.0),
            final_norm=config.get("final_norm", True),
            out_indices=config.get("out_indices", [-1]),
            with_cls_token=config.get("with_cls_token", False),
            output_cls_token=config.get("output_cls_token", False),
            register_tokens=config.get("register_tokens", 0),
            scatter_output=config.get("scatter_output", False),
            codebook_size=config.get("codebook_size", 1024),
            downsample_factor=config.get("downsample_factor", 16),
            latent_dim=config.get("latent_dim", None),
            num_res_blocks=config.get("num_res_blocks", 2),
            use_ema=config.get("use_ema", False),
            beta=config.get("beta", 0.25),
            norm_layer=config.get("norm_layer", "rmsnorm"),
            act_layer=config.get("act_layer", "swiglu"),
            use_attnres=config.get("use_attnres", False),
            attnres_mode=config.get("attnres_mode", "block"),
            attnres_gate_type=config.get("attnres_gate_type", "bias"),
            attnres_num_blocks=config.get("attnres_num_blocks", 8),
            attnres_recency_bias_init=config.get("attnres_recency_bias_init", 10.0),
            use_jit=config.get("use_jit", False),
        )

    if model_type == "gnn":
        return create_gnn_vision_rwkv7(
            img_size=img,
            embed_dims=config["embed_dims"],
            num_heads=config["num_heads"],
            depth=config["depth"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
            scatter_output=config.get("scatter_output", True),
            diff_slic_iters=config.get("diff_slic_iters", 5),
            compactness=config.get("compactness", 0.5),
            init_values=config.get("init_values", 1e-5),
            norm_layer=config.get("norm_layer", "rmsnorm"),
            act_layer=config.get("act_layer", "swiglu"),
            spixel_backend=config.get("spixel_backend", "diff_slic"),
            register_tokens=config.get("register_tokens", 0),
            downsample_factor=config.get("downsample_factor", 16),
            gnn_conv=config.get("gnn_conv", "gatv2"),
            gnn_heads=config.get("gnn_heads", 4),
            gnn_aggr=config.get("gnn_aggr", "mean"),
            use_cpp=config.get("use_cpp", False),
            use_jit=config.get("use_jit", False),
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")
