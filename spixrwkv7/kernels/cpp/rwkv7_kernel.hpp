// SpixRWKV-7: Common types and declarations for the RWKV-7 recurrent kernel.
//
// Design notes:
//   - `k` (raw key) and `r_k` (bonus key) are intentionally absent from
//     the kernel interface. `k` is pre-processed into `kt` by the Python
//     caller. The r_k bonus is computed after GroupNorm in Python, not inside
//     the scan, to match original RWKV-7 semantics.
//   - The separate recurrent_scan_avx2 / recurrent_scan_avx512 specialisations
//     have been removed. recurrent_scan_generic already selects AVX2 paths via
//     #if defined(__AVX2__) inline guards, so the separate TUs were dead code.
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

// =========================================================
// Generic recurrent scan (AVX2-accelerated inline paths).
//
// The rank-1 delta-rule structure reduces each O(S³) state update
// to O(S²) by factoring the outer-product updates:
//   state[t+1] = state[t]*diag(w) + v[t]⊗kt[t] - (state[t]·kk[t])⊗(kk[t]*a[t])
//
// GGML parallelisation: heads are striped across OpenMP threads so each
// thread owns a disjoint set of head slices, eliminating false sharing.
// =========================================================
torch::Tensor recurrent_scan_generic(
    const torch::Tensor& state,  // (B, Hd, S, S) recurrent state (read-only)
    const torch::Tensor& r,   // (B, N, Hd, S) receptance
    const torch::Tensor& v,   // (B, N, Hd, S) value
    const torch::Tensor& w,   // (B, N, Hd, S) decay
    const torch::Tensor& a,   // (B, N, Hd, S) alpha (delta rule)
    const torch::Tensor& kk,  // (B, N, Hd, S) L2-normalised key
    const torch::Tensor& kt   // (B, N, Hd, S) replacement (attenuated) key
);

// Public dispatcher: validates shapes/dtypes then delegates to the best
// available implementation.
torch::Tensor rwkv7_recurrent_scan(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt
);

#ifdef WT_CUDA
torch::Tensor recurrent_scan_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt
);
#endif

} // namespace kernel
} // namespace spixrwkv7
