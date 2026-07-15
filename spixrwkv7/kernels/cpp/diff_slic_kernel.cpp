// SpixRWKV-7: diffSLIC kernel implementation (generic)
#include "diff_slic_kernel.hpp"
#include "cpu_features.hpp"
#include <vector>
#include <cmath>
#include <cstring>

using namespace spixrwkv7::kernel;

// ===================================================================
// Generic cluster update
// ===================================================================

torch::Tensor spixrwkv7::kernel::update_clusters_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize)
{
    const auto B = elem_feats.size(0);
    const auto C = elem_feats.size(1);
    const auto H = elem_feats.size(2);
    const auto W = elem_feats.size(3);
    const auto h_s = clst_feats.size(2);
    const auto w_s = clst_feats.size(3);
    auto out = torch::empty({B, C, h_s, w_s}, elem_feats.options());

    const float* elem = elem_feats.data_ptr<float>();
    const float* clst = clst_feats.data_ptr<float>();
    float* result = out.data_ptr<float>();

    const int64_t clst_sz = (int64_t)h_s * w_s;
    const int64_t clst_stride = C * clst_sz;
    const int64_t elem_sz = (int64_t)H * W;
    const int64_t elem_stride = C * elem_sz;

    const int half_h = stride_h * radius;
    const int half_w = stride_w * radius;
    const int win_h = stride_h * (2 * radius + 1);
    const int win_w = stride_w * (2 * radius + 1);
    const int n = win_h * win_w;

    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t i = 0; i < h_s; i++) {
            for (int64_t j = 0; j < w_s; j++) {
                std::vector<float> sim_buf(n);
                const int top = i * stride_h - half_h;
                const int left = j * stride_w - half_w;

                float max_sim = -1e30f;
                for (int di = 0; di < win_h; di++) {
                    for (int dj = 0; dj < win_w; dj++) {
                        const int py = top + di;
                        const int px = left + dj;
                        const int k = di * win_w + dj;
                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            float sim = 0.0f;
                            for (int c = 0; c < C; c++) {
                                float c_val = clst[b * clst_stride + c * clst_sz + i * w_s + j];
                                float p_val = elem[b * elem_stride + c * elem_sz + py * W + px];
                                sim += c_val * p_val;
                            }
                            if (sim == 0.0f) sim = -1e9f;
                            else sim = sim / tau;
                            sim_buf[k] = sim;
                            if (sim > max_sim) max_sim = sim;
                        } else {
                            sim_buf[k] = -1e9f;
                        }
                    }
                }

                float sum_exp = 0.0f;
                for (int k = 0; k < n; k++) {
                    if (sim_buf[k] > -1e8f) {
                        sum_exp += std::exp(sim_buf[k] - max_sim);
                    }
                }
                const float inv_sum = 1.0f / (sum_exp + 1e-10f);

                for (int c = 0; c < C; c++) {
                    result[b * clst_stride + c * clst_sz + i * w_s + j] = 0.0f;
                }

                for (int di = 0; di < win_h; di++) {
                    for (int dj = 0; dj < win_w; dj++) {
                        const int py = top + di;
                        const int px = left + dj;
                        const int k = di * win_w + dj;
                        if (py < 0 || py >= H || px < 0 || px >= W) continue;
                        const float wgt = std::exp(sim_buf[k] - max_sim) * inv_sum;
                        for (int c = 0; c < C; c++) {
                            float p_val = elem[b * elem_stride + c * elem_sz + py * W + px];
                            result[b * clst_stride + c * clst_sz + i * w_s + j] += wgt * p_val;
                        }
                    }
                }

                if (normalize) {
                    float norm = 0.0f;
                    for (int c = 0; c < C; c++) {
                        float val = result[b * clst_stride + c * clst_sz + i * w_s + j];
                        norm += val * val;
                    }
                    norm = std::sqrt(norm > 1e-8f ? norm : 1e-8f);
                    const float inv = 1.0f / norm;
                    for (int c = 0; c < C; c++) {
                        result[b * clst_stride + c * clst_sz + i * w_s + j] *= inv;
                    }
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Generic pixel-to-superpixel assignment
// ===================================================================

torch::Tensor spixrwkv7::kernel::assign_pixels_generic(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau)
{
    const auto B = elem_feats.size(0);
    const auto C = elem_feats.size(1);
    const auto H = elem_feats.size(2);
    const auto W = elem_feats.size(3);
    const auto h_s = clst_feats.size(2);
    const auto w_s = clst_feats.size(3);
    const int nn = 2 * radius + 1;
    const int64_t nchan = (int64_t)nn * nn;

    auto out = torch::empty({B, nchan, H, W}, elem_feats.options());
    const float* elem = elem_feats.data_ptr<float>();
    const float* clst = clst_feats.data_ptr<float>();
    float* result = out.data_ptr<float>();

    const int64_t elem_sz = (int64_t)H * W;
    const int64_t elem_stride = C * elem_sz;
    const int64_t clst_sz = (int64_t)h_s * w_s;
    const int64_t clst_stride = C * clst_sz;
    const int64_t out_stride = nchan * elem_sz;

    TORCH_CHECK(nn <= 15, "radius is too large for stack buffer");

    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t y = 0; y < H; y++) {
            for (int64_t x = 0; x < W; x++) {
                const int ci = y / stride_h;
                const int cj = x / stride_w;

                float sim_buf[225];
                int valid[225];
                float max_sim = -1e30f;
                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int ni = ci + di;
                        const int nj = cj + dj;
                        const int flat = (di + radius) * nn + (dj + radius);
                        if (ni >= 0 && ni < h_s && nj >= 0 && nj < w_s) {
                            float sim = 0.0f;
                            for (int c = 0; c < C; c++) {
                                float p_val = elem[b * elem_stride + c * elem_sz + y * W + x];
                                float c_val = clst[b * clst_stride + c * clst_sz + ni * w_s + nj];
                                sim += p_val * c_val;
                            }
                            if (sim == 0.0f) sim = -1e9f;
                            else sim = sim / tau;
                            sim_buf[flat] = sim;
                            valid[flat] = 1;
                        } else {
                            sim_buf[flat] = -1e9f;
                            valid[flat] = 0;
                        }
                        if (valid[flat] && sim_buf[flat] > max_sim) max_sim = sim_buf[flat];
                    }
                }

                float sum_exp = 0.0f;
                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int flat = (di + radius) * nn + (dj + radius);
                        if (valid[flat]) {
                            sum_exp += std::exp(sim_buf[flat] - max_sim);
                        }
                    }
                }
                const float inv_sum = 1.0f / (sum_exp + 1e-10f);

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int flat = (di + radius) * nn + (dj + radius);
                        if (valid[flat]) {
                            result[b * out_stride + flat * elem_sz + y * W + x] = std::exp(sim_buf[flat] - max_sim) * inv_sum;
                        } else {
                            result[b * out_stride + flat * elem_sz + y * W + x] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Dispatchers with validation
// ===================================================================

namespace spixrwkv7 {
namespace kernel {

torch::Tensor diff_slic_update_clusters(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau,
    bool normalize)
{
    TORCH_CHECK(elem_feats.dim() == 4, "elem_feats must be 4D (B, C, H, W)");
    TORCH_CHECK(clst_feats.dim() == 4, "clst_feats must be 4D (B, C, h, w)");
    TORCH_CHECK(elem_feats.dtype() == torch::kFloat32, "elem_feats must be float32");
    TORCH_CHECK(clst_feats.dtype() == torch::kFloat32, "clst_feats must be float32");
    TORCH_CHECK(elem_feats.size(0) == clst_feats.size(0), "Batch size mismatch");
    TORCH_CHECK(elem_feats.size(1) == clst_feats.size(1), "Channel mismatch");
    TORCH_CHECK(elem_feats.is_contiguous(), "elem_feats must be contiguous");
    TORCH_CHECK(clst_feats.is_contiguous(), "clst_feats must be contiguous");
    return update_clusters_generic(elem_feats, clst_feats,
                                    stride_h, stride_w, radius, tau, normalize);
}

torch::Tensor diff_slic_assign_pixels(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau)
{
    TORCH_CHECK(elem_feats.dim() == 4, "elem_feats must be 4D (B, C, H, W)");
    TORCH_CHECK(clst_feats.dim() == 4, "clst_feats must be 4D (B, C, h, w)");
    TORCH_CHECK(elem_feats.dtype() == torch::kFloat32, "elem_feats must be float32");
    TORCH_CHECK(clst_feats.dtype() == torch::kFloat32, "clst_feats must be float32");
    TORCH_CHECK(elem_feats.is_contiguous(), "elem_feats must be contiguous");
    TORCH_CHECK(clst_feats.is_contiguous(), "clst_feats must be contiguous");
    return assign_pixels_generic(elem_feats, clst_feats,
                                  stride_h, stride_w, radius, tau);
}

} // namespace kernel
} // namespace spixrwkv7
