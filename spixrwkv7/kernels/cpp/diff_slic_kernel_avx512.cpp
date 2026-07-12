// SpixRWKV-7: diffSLIC AVX512 dispatch.
//
// Delegates to the authoritative generic implementation (see
// diff_slic_kernel_avx2.cpp for rationale). Correctness over SIMD for this
// non-critical operation.

#include "diff_slic_kernel.hpp"

namespace spixrwkv7 {
namespace kernel {

torch::Tensor update_clusters_avx512(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize)
{
    return update_clusters_generic(
        elem_feats, clst_feats, stride_h, stride_w, radius, tau, normalize);
}

torch::Tensor assign_pixels_avx512(
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
