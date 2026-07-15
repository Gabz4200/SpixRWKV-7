// SpixRWKV-7: Common types and declarations for the RWKV-7 recurrent kernel.
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

torch::Tensor recurrent_scan_generic(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt);

torch::Tensor rwkv7_recurrent_scan(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt);

#ifdef WT_CUDA
torch::Tensor recurrent_scan_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt);
#endif

} // namespace kernel
} // namespace spixrwkv7
