"""Optimized C++ kernel bindings for SpixRWKV-7.

Provides PyTorch bindings to optimized C++ kernels:
1. rwkv7_recurrent_scan — Accelerates the RWKV-7 delta-rule recurrence (O(S²) per step)
2. diff_slic_update_clusters — Fused cluster update for differentiable SLIC
3. diff_slic_assign_pixels — Fused pixel-to-superpixel assignment
"""

from typing import Optional, Tuple

import torch

from . import _C  # type: ignore[attr-defined]  # loads TORCH_LIBRARY registrations

# =====================================================================
# FakeTensor kernels for torch.compile / torch.export support
# =====================================================================

@torch.library.register_fake("spixrwkv7::rwkv7_recurrent_scan")
def _rwkv7_recurrent_scan_fake(state, r, v, w, a, kk, kt):
    return torch.empty_like(r)

@torch.library.register_fake("spixrwkv7::diff_slic_update_clusters")
def _update_clusters_fake(elem_feats, clst_feats, stride_h, stride_w, radius, tau, normalize):
    return torch.empty_like(clst_feats)

@torch.library.register_fake("spixrwkv7::diff_slic_assign_pixels")
def _assign_pixels_fake(elem_feats, clst_feats, stride_h, stride_w, radius, tau):
    nn = 2 * radius + 1
    B, C, H, W = elem_feats.shape
    return torch.empty(B, nn * nn, H, W, device=elem_feats.device, dtype=elem_feats.dtype)


# =====================================================================
# RWKV-7 Recurrent Scan
# =====================================================================

def rwkv7_recurrent_scan(
    state: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    kt: torch.Tensor,
    use_cpp: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RWKV-7 recurrent scan — optimized C++ or PyTorch fallback.

    Performs the sequential delta-rule recurrence:
      state[t+1] = state[t] * w + state[t] @ (-kk*kk*a) + v @ kt^T
      out[t]     = state[t+1] @ r

    The r_k bonus (r * kt * r_k * v) is NOT computed here; the caller
    applies it after GroupNorm, matching the original PyTorch semantics.

    Args:
        state: (B, Hd, S, S) recurrent state
        r:     (B, N, Hd, S) receptance
        v:     (B, N, Hd, S) value
        w:     (B, N, Hd, S) decay weight
        a:     (B, N, Hd, S) alpha (delta rule)
        kk:    (B, N, Hd, S) L2-normalized key
        kt:    (B, N, Hd, S) replacement key
        use_cpp: route through C++ kernel when True
        mask:  optional (B, N) token validity mask

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
                out = torch.ops.spixrwkv7.rwkv7_recurrent_scan(state, r, v, w, a, kk, kt)
            else:
                return _rwkv7_scan_pytorch(state, r, v, w, a, kk, kt, mask)
        else:
            out = torch.ops.spixrwkv7.rwkv7_recurrent_scan(state, r, v, w, a, kk, kt)
        if mask is not None:
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
        return out
    return _rwkv7_scan_pytorch(state, r, v, w, a, kk, kt, mask)


def _rwkv7_scan_pytorch(
    state: torch.Tensor,
    r: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    kt: torch.Tensor,
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

        vk = v_t.unsqueeze(-1) @ kt_t.unsqueeze(-2)
        ab = (-kk_t).unsqueeze(-1) @ (kk_t * a_t).unsqueeze(-2)

        state.copy_(state * w_eff.unsqueeze(-2)
                    + (state @ ab.float() + vk.float()) * mask_t)

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
    """Fused cluster update for diffSLIC — C++ or PyTorch fallback."""
    if use_cpp:
        stride_h, stride_w = stride
        if elem_feats.is_cuda:
            if hasattr(_C, "update_clusters_cuda"):
                return torch.ops.spixrwkv7.diff_slic_update_clusters(
                    elem_feats, clst_feats, stride_h, stride_w, radius, tau, normalize,
                )
            else:
                return _update_clusters_pytorch(
                    elem_feats, clst_feats, stride, radius, tau, normalize,
                )
        else:
            return torch.ops.spixrwkv7.diff_slic_update_clusters(
                elem_feats, clst_feats, stride_h, stride_w, radius, tau, normalize,
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
    """Fused pixel-to-superpixel assignment — C++ or PyTorch fallback."""
    if use_cpp:
        stride_h, stride_w = stride
        if elem_feats.is_cuda:
            if hasattr(_C, "assign_pixels_cuda"):
                return torch.ops.spixrwkv7.diff_slic_assign_pixels(
                    elem_feats, clst_feats, stride_h, stride_w, radius, tau,
                )
            else:
                return _assign_pixels_pytorch(
                    elem_feats, clst_feats, stride, radius, tau,
                )
        else:
            return torch.ops.spixrwkv7.diff_slic_assign_pixels(
                elem_feats, clst_feats, stride_h, stride_w, radius, tau,
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


HAS_GGML: bool = False
_HAS_CPP_KERNEL: bool = True
