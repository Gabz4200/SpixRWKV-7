"""Optimized Vision_RWKV7 using C++ kernels."""

import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

from spixrwkv7.spixrwkv7 import Vision_RWKV7, Vision_RWKV7_Block, ClassificationHead
from spixrwkv7.layers.graph import HEAD_SIZE
from spixrwkv7.layers.drop import DropPath


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
    )