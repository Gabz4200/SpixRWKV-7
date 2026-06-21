"""Optimized C++ kernel bindings for SpixRWKV-7.

Provides PyTorch bindings to optimized C++ kernels:
1. rwkv7_recurrent_scan — Accelerates the RWKV-7 delta-rule recurrence (O(S²) per step)
2. diff_slic_update_clusters — Fused cluster update for differentiable SLIC
3. diff_slic_assign_pixels — Fused pixel-to-superpixel assignment

Both kernels have an AVX512-optimized path that is dispatched at runtime
via cpuid checks. On CPUs without AVX512, a generic fallback is used.
"""

import torch
from typing import Tuple, Optional

from . import _C  # type: ignore[attr-defined]  # fails fast if .so not built


# =====================================================================
# RWKV-7 Recurrent Scan
# =====================================================================

def rwkv7_recurrent_scan(
    state: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    kt: torch.Tensor,
    r_k: torch.Tensor,
    use_cpp: bool = True,
    mask: Optional[torch.Tensor] = None,
    quantization: Optional[str] = None,
) -> torch.Tensor:
    """RWKV-7 recurrent scan — optimized C++ or PyTorch fallback.

    Performs the sequential delta-rule recurrence:
      state[t+1] = state[t] * w + state[t] @ (-kk*kk*a) + v @ kt^T
      out[t] = state[t+1] @ r + bonus(r, kt, r_k) * v

    Optimized: leverages rank-1 structure for O(S²) instead of O(S³).

    Args:
        state: (B, Hd, S, S) recurrent state, updated in-place
        r: (B, N, Hd, S) receptance
        k: (B, N, Hd, S) key (accepted for interface consistency, not used)
        v: (B, N, Hd, S) value
        w: (B, N, Hd, S) decay
        a: (B, N, Hd, S) alpha (delta rule)
        kk: (B, N, Hd, S) L2-normalized key
        kt: (B, N, Hd, S) replacement key
        r_k: (Hd, S) per-head bonus key
        use_cpp: whether to use C++ kernel (default True)
        quantization: optional quantization variant ("q4_0" or "q5_1")

    Returns:
        out: (B, N, Hd, S) output tensor
    """
    if use_cpp:
        if mask is not None:
            m = mask.unsqueeze(-1).unsqueeze(-1)
            w = torch.where(m == 0, torch.ones_like(w), w)
            kk = kk * m
            a = a * m
            v = v * m
        if r.is_cuda:
            if hasattr(_C, "recurrent_scan_cuda"):
                if quantization == "q4_0":
                    out = _C.recurrent_scan_q4_0_cuda(state, r, k, v, w, a, kk, kt, r_k)
                elif quantization == "q5_1":
                    out = _C.recurrent_scan_q5_1_cuda(state, r, k, v, w, a, kk, kt, r_k)
                else:
                    out = _C.recurrent_scan_cuda(state, r, k, v, w, a, kk, kt, r_k)
            else:
                return _rwkv7_scan_pytorch(state, r, v, w, a, kk, kt, r_k, mask)
        else:
            if quantization == "q4_0":
                out = _C.rwkv7_recurrent_scan_q4_0(state, r, k, v, w, a, kk, kt, r_k)
            elif quantization == "q5_1":
                out = _C.rwkv7_recurrent_scan_q5_1(state, r, k, v, w, a, kk, kt, r_k)
            else:
                out = _C.rwkv7_recurrent_scan(state, r, k, v, w, a, kk, kt, r_k)
        if mask is not None:
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
        return out
    return _rwkv7_scan_pytorch(state, r, v, w, a, kk, kt, r_k, mask)


def _rwkv7_scan_pytorch(
    state: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    kt: torch.Tensor,
    r_k: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch fallback for RWKV-7 recurrent scan (for verification)."""
    _, N, _, _ = r.shape
    outputs = []

    for t in range(N):
        r_t = r[:, t]
        v_t = v[:, t]
        w_t = w[:, t]
        kk_t = kk[:, t]
        kt_t = kt[:, t]
        a_t = a[:, t]

        if mask is not None:
            mask_t = mask[:, t, None, None]
            w_eff = torch.where(mask_t == 0.0, torch.ones_like(w_t), w_t)
        else:
            mask_t = 1.0
            w_eff = w_t

        # Outer products (rank-1)
        vk = v_t.unsqueeze(-1) @ kt_t.unsqueeze(-2)  # (B, Hd, S, S)
        ab = (-kk_t).unsqueeze(-1) @ (kk_t * a_t).unsqueeze(-2)  # (B, Hd, S, S)

        # State update
        # w: (B,Hd,S) → (B,Hd,1,S) for column-wise decay
        state.copy_(state * w_eff.unsqueeze(-2)
                    + (state @ ab.float() + vk.float()) * mask_t)

        # Output = state @ r
        out_t = (state @ r_t.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            out_t = out_t * mask[:, t, None, None]
        outputs.append(out_t)

    return torch.stack(outputs, dim=1)


# =====================================================================
# diffSLIC Cluster Update
# =====================================================================

def diff_slic_update_clusters(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    radius: int = 1,
    tau: float = 0.01,
    normalize: bool = True,
    use_cpp: bool = True,
) -> torch.Tensor:
    """Fused cluster update for diffSLIC — C++ or PyTorch fallback.

    For each cluster center, extracts a window of pixels,
    computes similarity → softmax → weighted aggregation.

    Args:
        elem_feats: (B, C, H, W) padded image features
        clst_feats: (B, C, h_s, w_s) cluster centers
        stride: (stride_h, stride_w) pixel stride
        radius: candidate search radius
        tau: softmax temperature
        normalize: L2-normalize features before comparison
        use_cpp: use C++ kernel (default True)

    Returns:
        new_clst_feats: (B, C, h_s, w_s) updated cluster features
    """
    if use_cpp:
        stride_h, stride_w = stride
        if elem_feats.is_cuda:
            if hasattr(_C, "update_clusters_cuda"):
                return _C.update_clusters_cuda(
                    elem_feats, clst_feats,
                    stride_h, stride_w,
                    radius, tau, normalize,
                )
            else:
                return _update_clusters_pytorch(
                    elem_feats, clst_feats, stride, radius, tau, normalize,
                )
        else:
            return _C.diff_slic_update_clusters(
                elem_feats, clst_feats,
                stride_h, stride_w,
                radius, tau, normalize,
            )
    return _update_clusters_pytorch(
        elem_feats, clst_feats, stride, radius, tau, normalize,
    )


def _update_clusters_pytorch(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    radius: int = 1,
    tau: float = 0.01,
    normalize: bool = True,
) -> torch.Tensor:
    """Pure PyTorch fallback for cluster update."""
    from spixrwkv7.data.diff_slic import update_clst_feats
    result, _, _ = update_clst_feats(
        elem_feats, clst_feats, stride, tau, radius, stable=False,
    )
    if normalize:
        result = result / result.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return result


def diff_slic_assign_pixels(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    radius: int = 1,
    tau: float = 0.01,
    use_cpp: bool = True,
) -> torch.Tensor:
    """Fused pixel-to-superpixel assignment — C++ or PyTorch fallback.

    Args:
        elem_feats: (B, C, H, W) padded image features
        clst_feats: (B, C, h_s, w_s) cluster centers
        stride: (stride_h, stride_w)
        radius: candidate search radius
        tau: softmax temperature
        use_cpp: use C++ kernel (default True)

    Returns:
        assignment: (B, (2*radius+1)^2, H, W) soft assignment
    """
    if use_cpp:
        stride_h, stride_w = stride
        if elem_feats.is_cuda:
            if hasattr(_C, "assign_pixels_cuda"):
                return _C.assign_pixels_cuda(
                    elem_feats, clst_feats,
                    stride_h, stride_w,
                    radius, tau,
                )
            else:
                return _assign_pixels_pytorch(
                    elem_feats, clst_feats, stride, radius, tau,
                )
        else:
            return _C.diff_slic_assign_pixels(
                elem_feats, clst_feats,
                stride_h, stride_w,
                radius, tau,
            )
    return _assign_pixels_pytorch(
        elem_feats, clst_feats, stride, radius, tau,
    )


def _assign_pixels_pytorch(
    elem_feats: torch.Tensor,
    clst_feats: torch.Tensor,
    stride: Tuple[int, int],
    radius: int = 1,
    tau: float = 0.01,
) -> torch.Tensor:
    """Pure PyTorch fallback for pixel assignment."""
    from spixrwkv7.data.diff_slic import compute_elem_to_center_assignment
    assignments, _ = compute_elem_to_center_assignment(
        clst_feats, elem_feats, stride, tau, radius, stable=False,
    )
    return assignments


# =====================================================================
# GGML CPU path (optional — requires building spixrwkv7/kernels/cpp/ggml_bridge)
# =====================================================================
# GGML provides quantized (INT4/INT8) and AVX2-optimized FP32 SGEMM.
# On CPUs without AVX512, GGML's SGEMM outperforms the generic C path
# for large S×S matrix multiplications (S = HEAD_SIZE = 64).
#
# Integration steps (not yet implemented):
#   1. Build ggml as a CMake subproject in spixrwkv7/kernels/cpp/ggml/
#   2. Add ggml_mul_mat wrapper in torch_binding.cpp
#   3. Expose via _C.ggml_rwkv7_recurrent_scan(state, A, vk)
#   4. Toggle: rwkv7_recurrent_scan(..., use_ggml=True)
#
# Detection stub (import side-effect: checks for ggml symbol):

def _check_ggml_available() -> bool:
    """Return True if the GGML-accelerated scan is available.

    Currently always returns False — GGML bridge not yet compiled.
    To enable: build spixrwkv7/kernels/cpp with -DWITH_GGML=ON.
    """
    return hasattr(_C, "ggml_rwkv7_recurrent_scan")


HAS_GGML: bool = _check_ggml_available()
_HAS_CPP_KERNEL: bool = True


