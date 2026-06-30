"""Optimized Vision_RWKV7 using C++ kernels."""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from spixrwkv7.models.spixrwkv7 import Vision_RWKV7


class OptimizedVision_RWKV7(Vision_RWKV7):
    """Optimized Vision-RWKV-7 backbone using C++ kernels.

    This is a drop-in replacement for Vision_RWKV7 that uses optimized
    kernels when available. The optimization is applied at the block level.

    Usage:
        # Use optimized kernel for inference
        model = OptimizedVision_RWKV7(..., use_cpp=True)

        # Use parallel variant (Householder reflections)
        model = OptimizedVision_RWKV7(..., use_parallel=True)

        # Use PyTorch (default)
        model = OptimizedVision_RWKV7(..., use_cpp=False, use_parallel=False)
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 6,
        embed_dims: int = 192,
        num_heads: Optional[int] = None,
        depth: int = 12,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 0.0,
        final_norm: bool = True,
        out_indices: Sequence[int] = (-1,),
        with_cls_token: bool = False,
        output_cls_token: bool = False,
        scatter_output: bool = False,
        num_superpixels: int = 196,
        spixel_size: Optional[int] = None,
        diff_slic_iters: int = 5,
        compactness: float = 0.5,
        register_tokens: int = 0,
        use_cpp: bool = True,
        use_parallel: bool = False,
        norm_layer: str = "layernorm",
        act_layer: str = "relu2",
        spixel_backend: str = "diff_slic",
        use_attnres: bool = False,
        attnres_mode: str = "block",
        attnres_gate_type: str = "bias",
        attnres_num_blocks: int = 8,
        attnres_recency_bias_init: float = 10.0,
    ):
        super().__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depth=depth,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            final_norm=final_norm,
            out_indices=out_indices,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            scatter_output=scatter_output,
            num_superpixels=num_superpixels,
            spixel_size=spixel_size,
            diff_slic_iters=diff_slic_iters,
            compactness=compactness,
            register_tokens=register_tokens,
            use_cpp=use_cpp,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_parallel=use_parallel,
            spixel_backend=spixel_backend,
            use_attnres=use_attnres,
            attnres_mode=attnres_mode,
            attnres_gate_type=attnres_gate_type,
            attnres_num_blocks=attnres_num_blocks,
            attnres_recency_bias_init=attnres_recency_bias_init,
        )

    def _make_blocks(
        self,
        embed_dims: int,
        num_heads: int,
        depth: int,
        drop_path_rate: float,
        init_values: Optional[float],
        with_cls_token: bool,
        norm_layer: str,
        act_layer: str,
        use_cpp: bool,
        use_parallel: bool = False,
        **kwargs,
    ) -> nn.ModuleList:
        from spixrwkv7.kernels.optimized_block import OptimizedVision_RWKV7_Block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        return nn.ModuleList(
            [
                OptimizedVision_RWKV7_Block(
                    embed_dims,
                    num_heads,
                    depth,
                    i,
                    drop_prob=dpr[i],
                    init_values=init_values,
                    with_cls_token=with_cls_token,
                    use_cpp=use_cpp,
                    use_parallel=use_parallel,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_attnres=self.use_attnres,
                    attnres_mode=self.attnres_mode,
                    attnres_gate_type=self.attnres_gate_type,
                    attnres_num_blocks=self.attnres_num_blocks,
                    attnres_recency_bias_init=self.attnres_recency_bias_init,
                )
                for i in range(depth)
            ]
        )


def rwkv7_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run forward pass of an OptimizedVision_RWKV7 model."""
    return model(x)


def create_optimized_vision_rwkv7(
    img_size: int = 224,
    embed_dims: int = 192,
    num_heads: Optional[int] = None,
    depth: int = 12,
    drop_path_rate: float = 0.0,
    init_values: Optional[float] = 0.0,
    final_norm: bool = True,
    out_indices: Sequence[int] = (-1,),
    with_cls_token: bool = False,
    output_cls_token: bool = False,
    scatter_output: bool = False,
    num_superpixels: int = 196,
    spixel_size: Optional[int] = None,
    diff_slic_iters: int = 5,
    compactness: float = 0.5,
    register_tokens: int = 0,
    use_cpp: bool = True,
    use_parallel: bool = False,
    norm_layer: str = "layernorm",
    act_layer: str = "relu2",
    spixel_backend: str = "diff_slic",
    use_attnres: bool = False,
    attnres_mode: str = "block",
    attnres_gate_type: str = "bias",
    attnres_num_blocks: int = 8,
    attnres_recency_bias_init: float = 10.0,
) -> OptimizedVision_RWKV7:
    """Create optimized Vision_RWKV7 with 6-channel input."""
    return OptimizedVision_RWKV7(
        img_size=img_size,
        in_chans=6,
        embed_dims=embed_dims,
        num_heads=num_heads,
        depth=depth,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        final_norm=final_norm,
        out_indices=out_indices,
        with_cls_token=with_cls_token,
        output_cls_token=output_cls_token,
        scatter_output=scatter_output,
        num_superpixels=num_superpixels,
        spixel_size=spixel_size,
        diff_slic_iters=diff_slic_iters,
        compactness=compactness,
        register_tokens=register_tokens,
        use_cpp=use_cpp,
        use_parallel=use_parallel,
        norm_layer=norm_layer,
        act_layer=act_layer,
        spixel_backend=spixel_backend,
        use_attnres=use_attnres,
        attnres_mode=attnres_mode,
        attnres_gate_type=attnres_gate_type,
        attnres_num_blocks=attnres_num_blocks,
        attnres_recency_bias_init=attnres_recency_bias_init,
    )
