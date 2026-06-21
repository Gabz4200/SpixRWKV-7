// SpixRWKV-7: Common types and constants for RWKV-7 recurrent kernel
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

// Forward declarations for dispatch targets
torch::Tensor recurrent_scan_generic(
    torch::Tensor& state,     // (B, Hd, S, S) - modified in-place
    const torch::Tensor& r,   // (B, N, Hd, S)
    const torch::Tensor& k,   // (B, N, Hd, S)
    const torch::Tensor& v,   // (B, N, Hd, S)
    const torch::Tensor& w,   // (B, N, Hd, S)
    const torch::Tensor& a,   // (B, N, Hd, S)
    const torch::Tensor& kk,  // (B, N, Hd, S)
    const torch::Tensor& kt,  // (B, N, Hd, S)
    const torch::Tensor& r_k  // (Hd, S)
);

#ifdef __AVX512F__
torch::Tensor recurrent_scan_avx512(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);
#endif

#ifdef __AVX2__
torch::Tensor recurrent_scan_avx2(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);

torch::Tensor recurrent_scan_q4_0(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);

torch::Tensor recurrent_scan_q5_1(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);
#endif

#ifdef WT_CUDA
torch::Tensor recurrent_scan_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);

torch::Tensor recurrent_scan_q4_0_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);

torch::Tensor recurrent_scan_q5_1_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k
);
#endif

} // namespace kernel
} // namespace spixrwkv7
