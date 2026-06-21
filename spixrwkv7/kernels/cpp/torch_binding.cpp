// SpixRWKV-7: PyTorch bindings for CPU kernels
#include <torch/extension.h>
#include <string>

// Forward declarations from RWKV-7 kernel
namespace spixrwkv7 {
namespace kernel {
torch::Tensor rwkv7_recurrent_scan(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

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
torch::Tensor recurrent_scan_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

torch::Tensor recurrent_scan_q4_0_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

torch::Tensor recurrent_scan_q5_1_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k);

torch::Tensor update_clusters_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau, bool normalize);

torch::Tensor assign_pixels_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau);
#endif
} // namespace kernel
} // namespace spixrwkv7

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv7_recurrent_scan", &spixrwkv7::kernel::rwkv7_recurrent_scan,
          "RWKV-7 recurrent scan (CPU optimized)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("rwkv7_recurrent_scan_q4_0", &spixrwkv7::kernel::recurrent_scan_q4_0,
          "RWKV-7 recurrent scan (Q4_0 quantized)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("rwkv7_recurrent_scan_q5_1", &spixrwkv7::kernel::recurrent_scan_q5_1,
          "RWKV-7 recurrent scan (Q5_1 quantized)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

#ifdef __AVX2__
    m.def("rwkv7_recurrent_scan_q4_0", &spixrwkv7::kernel::recurrent_scan_q4_0,
          "RWKV-7 recurrent scan with Q4_0 quantized weights",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("rwkv7_recurrent_scan_q5_1", &spixrwkv7::kernel::recurrent_scan_q5_1,
          "RWKV-7 recurrent scan with Q5_1 quantized weights",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));
#endif

    m.def("diff_slic_update_clusters", &spixrwkv7::kernel::diff_slic_update_clusters,
          "diffSLIC cluster update (fused CPU kernel)",
          py::arg("elem_feats"), py::arg("clst_feats"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("radius"), py::arg("tau"),
          py::arg("normalize") = true);

    m.def("diff_slic_assign_pixels", &spixrwkv7::kernel::diff_slic_assign_pixels,
          "diffSLIC pixel-to-superpixel assignment (CPU kernel)",
          py::arg("elem_feats"), py::arg("clst_feats"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("radius"), py::arg("tau"));

#ifdef WT_CUDA
    m.def("recurrent_scan_cuda", &spixrwkv7::kernel::recurrent_scan_cuda,
          "RWKV-7 recurrent scan (CUDA optimized)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("recurrent_scan_q4_0_cuda", &spixrwkv7::kernel::recurrent_scan_q4_0_cuda,
          "RWKV-7 recurrent scan with Q4_0 quantized weights (CUDA)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("recurrent_scan_q5_1_cuda", &spixrwkv7::kernel::recurrent_scan_q5_1_cuda,
          "RWKV-7 recurrent scan with Q5_1 quantized weights (CUDA)",
          py::arg("state"), py::arg("r"), py::arg("k"),
          py::arg("v"), py::arg("w"), py::arg("a"),
          py::arg("kk"), py::arg("kt"), py::arg("r_k"));

    m.def("update_clusters_cuda", &spixrwkv7::kernel::update_clusters_cuda,
          "diffSLIC cluster update (CUDA kernel)",
          py::arg("elem_feats"), py::arg("clst_feats"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("radius"), py::arg("tau"),
          py::arg("normalize") = true);

    m.def("assign_pixels_cuda", &spixrwkv7::kernel::assign_pixels_cuda,
          "diffSLIC pixel-to-superpixel assignment (CUDA kernel)",
          py::arg("elem_feats"), py::arg("clst_feats"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("radius"), py::arg("tau"));
#endif
}
