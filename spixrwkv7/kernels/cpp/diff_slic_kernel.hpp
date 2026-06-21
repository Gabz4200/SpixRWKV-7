// SpixRWKV-7: diffSLIC kernel declarations
#pragma once
#include <torch/extension.h>

namespace spixrwkv7 {
namespace kernel {

// Fused cluster update: extract windows → compute similarity → softmax → aggregate
// Equivalent to update_clst_feats() in diff_slic.py but fused to avoid materializing
// the unfolded tensor.
//
// Args:
//   elem_feats: (B, C, H, W) padded image features
//   clst_feats: (B, C, h_s, w_s) cluster center features
//   stride_h, stride_w: pixel stride between cluster centers
//   radius: candidate search radius
//   tau: softmax temperature
//   normalize: whether to L2-normalize features before comparison
// Returns:
//   new_clst_feats: (B, C, h_s, w_s) updated cluster features
torch::Tensor update_clusters_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

// Fused pixel-to-superpixel assignment
// Equivalent to compute_elem_to_center_assignment() in diff_slic.py
//
// Args:
//   elem_feats: (B, C, H, W) padded image features
//   clst_feats: (B, C, h_s, w_s) cluster center features
//   stride_h, stride_w: pixel stride between cluster centers
//   radius: candidate search radius
//   tau: softmax temperature
// Returns:
//   p2s_assign: (B, (2*radius+1)^2, H, W) pixel-to-superpixel soft assignment
torch::Tensor assign_pixels_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);

// AVX2 variants
#if defined(__AVX2__)
torch::Tensor update_clusters_avx2(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize);

torch::Tensor assign_pixels_avx2(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);
#endif

// AVX512 variants
#ifdef __AVX512F__
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
#endif

// CUDA variants
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