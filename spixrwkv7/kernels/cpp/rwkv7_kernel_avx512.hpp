// SpixRWKV-7: RWKV-7 AVX512 kernel declarations
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

torch::Tensor recurrent_scan_avx512(
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
