"""Optimized C++ CPU kernels for SpixRWKV-7.

Provides accelerated C++ implementations of:
- RWKV-7 recurrent scan (delta-rule state update + output)
- diffSLIC cluster update and pixel-to-superpixel assignment

All kernels support runtime CPU dispatch: AVX512 (when available) → generic.
"""

from spixrwkv7.kernels.optimized_block import OptimizedVision_RWKV7_Block, ParallelRecurrentScan
from spixrwkv7.kernels.optimized_vision import (
    OptimizedVision_RWKV7,
    create_optimized_vision_rwkv7,
    rwkv7_forward,
)
from spixrwkv7.kernels.rwkv7_kernel import (
    HAS_GGML,
    diff_slic_assign_pixels,
    diff_slic_update_clusters,
    rwkv7_recurrent_scan,
)

# Kernel is always available (module fails at import if _C.so not built)
HAS_CPP_KERNEL: bool = True

__all__ = [
    "HAS_CPP_KERNEL",
    "HAS_GGML",
    "OptimizedVision_RWKV7",
    "OptimizedVision_RWKV7_Block",
    "ParallelRecurrentScan",
    "create_optimized_vision_rwkv7",
    "diff_slic_assign_pixels",
    "diff_slic_update_clusters",
    "rwkv7_forward",
    "rwkv7_recurrent_scan",
]
