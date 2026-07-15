// SpixRWKV-7: diffSLIC kernel declarations
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

torch::Tensor update_clusters_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

torch::Tensor assign_pixels_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);

torch::Tensor diff_slic_update_clusters(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

torch::Tensor diff_slic_assign_pixels(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);

#ifdef WT_CUDA
torch::Tensor update_clusters_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

torch::Tensor assign_pixels_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);
#endif

} // namespace kernel
} // namespace spixrwkv7
