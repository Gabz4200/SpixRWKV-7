// SpixRWKV-7: RWKV-7 AVX2 + quantization kernel declarations
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

// AVX2-optimized kernel (S=64, processes 8 floats at a time)
torch::Tensor recurrent_scan_avx2(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

// Quantized variants (weights provided in Q4_0/Q5_1 format)
torch::Tensor recurrent_scan_q4_0(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

torch::Tensor recurrent_scan_q5_1(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

} // namespace kernel
} // namespace spixrwkv7