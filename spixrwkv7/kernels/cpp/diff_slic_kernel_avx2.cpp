// SpixRWKV-7: diffSLIC AVX2 dispatch.
//
// The AVX2 diffSLIC paths previously duplicated the algorithm from the
// generic implementation and drifted out of sync with the PyTorch reference
// in spixrwkv7/data/diff_slic.py. To guarantee correctness they now delegate
// to the single authoritative generic implementation in diff_slic_kernel.cpp.
// The recurrent scan keeps its SIMD speed; diffSLIC is not on the critical path.

#include "diff_slic_kernel.hpp"

namespace spixrwkv7 {
namespace kernel {

torch::Tensor update_clusters_avx2(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize)
{
    return update_clusters_generic(
        elem_feats, clst_feats, stride_h, stride_w, radius, tau, normalize);
}

torch::Tensor assign_pixels_avx2(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau)
{
    return assign_pixels_generic(
        elem_feats, clst_feats, stride_h, stride_w, radius, tau);
}

} // namespace kernel
} // namespace spixrwkv7
