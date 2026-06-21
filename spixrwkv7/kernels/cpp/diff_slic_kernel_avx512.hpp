// SpixRWKV-7: diffSLIC AVX512 kernel declarations
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

torch::Tensor update_clusters_avx512(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

torch::Tensor assign_pixels_avx512(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);

} // namespace kernel
} // namespace spixrwkv7
