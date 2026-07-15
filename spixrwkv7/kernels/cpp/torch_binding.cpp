// SpixRWKV-7: PyTorch custom op registration via TORCH_LIBRARY.
#include <torch/library.h>
#include <Python.h>
#include "rwkv7_kernel.hpp"
#include "diff_slic_kernel.hpp"

extern "C" {
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(spixrwkv7, m) {
    m.def("rwkv7_recurrent_scan(Tensor state, Tensor r, Tensor v, Tensor w, Tensor a, Tensor kk, Tensor kt) -> Tensor");
    m.def("diff_slic_update_clusters(Tensor elem_feats, Tensor clst_feats, int stride_h, int stride_w, int radius, float tau, bool normalize) -> Tensor");
    m.def("diff_slic_assign_pixels(Tensor elem_feats, Tensor clst_feats, int stride_h, int stride_w, int radius, float tau) -> Tensor");
}

namespace {

torch::Tensor rwkv7_recurrent_scan_wrapper(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt) {
    return spixrwkv7::kernel::rwkv7_recurrent_scan(state, r, v, w, a, kk, kt);
}

torch::Tensor diff_slic_update_clusters_wrapper(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int64_t stride_h, int64_t stride_w,
    int64_t radius, double tau,
    bool normalize) {
    return spixrwkv7::kernel::diff_slic_update_clusters(
        elem_feats, clst_feats,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(radius), static_cast<float>(tau), normalize);
}

torch::Tensor diff_slic_assign_pixels_wrapper(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int64_t stride_h, int64_t stride_w,
    int64_t radius, double tau) {
    return spixrwkv7::kernel::diff_slic_assign_pixels(
        elem_feats, clst_feats,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(radius), static_cast<float>(tau));
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(spixrwkv7, CPU, m) {
    m.impl("rwkv7_recurrent_scan", &rwkv7_recurrent_scan_wrapper);
    m.impl("diff_slic_update_clusters", &diff_slic_update_clusters_wrapper);
    m.impl("diff_slic_assign_pixels", &diff_slic_assign_pixels_wrapper);
}

#ifdef WT_CUDA

namespace {

torch::Tensor recurrent_scan_cuda_wrapper(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt) {
    return spixrwkv7::kernel::recurrent_scan_cuda(state, r, v, w, a, kk, kt);
}

torch::Tensor update_clusters_cuda_wrapper(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int64_t stride_h, int64_t stride_w,
    int64_t radius, double tau,
    bool normalize) {
    return spixrwkv7::kernel::update_clusters_cuda(
        elem_feats, clst_feats,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(radius), static_cast<float>(tau), normalize);
}

torch::Tensor assign_pixels_cuda_wrapper(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int64_t stride_h, int64_t stride_w,
    int64_t radius, double tau) {
    return spixrwkv7::kernel::assign_pixels_cuda(
        elem_feats, clst_feats,
        static_cast<int>(stride_h), static_cast<int>(stride_w),
        static_cast<int>(radius), static_cast<float>(tau));
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(spixrwkv7, CUDA, m) {
    m.impl("rwkv7_recurrent_scan", &recurrent_scan_cuda_wrapper);
    m.impl("diff_slic_update_clusters", &update_clusters_cuda_wrapper);
    m.impl("diff_slic_assign_pixels", &assign_pixels_cuda_wrapper);
}
#endif
