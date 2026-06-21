// SpixRWKV-7: diffSLIC kernel implementation (generic + dispatcher)
#include "diff_slic_kernel.hpp"
#include "cpu_features.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace spixrwkv7::kernel;

// ===================================================================
// Generic cluster update
// For each cluster center (i,j), extract a window of pixels,
// compute similarity, softmax, and weighted aggregation
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

    for (int64_t b = 0; b < B; b++) {
        for (int i = 0; i < h_s; i++) {
            for (int j = 0; j < w_s; j++) {
                // Center pixel position in padded image
                const int cy = i * stride_h;
                const int cx = j * stride_w;

                // Get cluster center feature
                const float* center = clst + b * C * h_s * w_s + i * C * w_s + j * C;
                float* dst = result + b * C * h_s * w_s + i * C * w_s + j * C;

                // Normalize cluster center if requested
                float center_norm = 0.0f;
                if (normalize) {
                    for (int c = 0; c < C; c++)
                        center_norm += center[c] * center[c];
                    center_norm = std::max(center_norm, 1e-8f);
                    center_norm = 1.0f / std::sqrt(center_norm);
                }

                // Extract window and compute similarities
                float sim_buf[256]; // max 9x9=81 for radius=4
                int valid_count = 0;

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int py = cy + di * stride_h;
                        const int px = cx + dj * stride_w;

                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            const float* pixel = elem + b * C * H * W + py * C * W + px * C;

                            float sim = 0.0f;
                            if (normalize) {
                                float p_norm = 0.0f;
                                for (int c = 0; c < C; c++)
                                    p_norm += pixel[c] * pixel[c];
                                p_norm = std::max(p_norm, 1e-8f);
                                p_norm = 1.0f / std::sqrt(p_norm);
                                for (int c = 0; c < C; c++)
                                    sim += (center[c] * center_norm) * (pixel[c] * p_norm);
                            } else {
                                for (int c = 0; c < C; c++)
                                    sim += center[c] * pixel[c];
                            }
                            sim_buf[valid_count] = sim / tau;
                            valid_count++;
                        }
                    }
                }

                if (valid_count == 0) {
                    std::memset(dst, 0, C * sizeof(float));
                    continue;
                }

                // Stable softmax
                float max_sim = sim_buf[0];
                for (int p = 1; p < valid_count; p++)
                    max_sim = std::max(max_sim, sim_buf[p]);

                float sum_exp = 0.0f;
                for (int p = 0; p < valid_count; p++)
                    sum_exp += std::exp(sim_buf[p] - max_sim);

                float inv_sum = 1.0f / (sum_exp + 1e-10f);

                // Weighted aggregation
                std::memset(dst, 0, C * sizeof(float));
                int pixel_idx = 0;

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int py = cy + di * stride_h;
                        const int px = cx + dj * stride_w;

                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            const float* pixel = elem + b * C * H * W + py * C * W + px * C;
                            const float weight = std::exp(sim_buf[pixel_idx] - max_sim) * inv_sum;

                            for (int c = 0; c < C; c++)
                                dst[c] += weight * pixel[c];

                            pixel_idx++;
                        }
                    }
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Generic pixel-to-superpixel assignment
// For each pixel, compute similarity with nearby cluster centers
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

    auto out = torch::empty({B, nn * nn, H, W}, elem_feats.options());

    const float* elem = elem_feats.data_ptr<float>();
    const float* clst = clst_feats.data_ptr<float>();
    float* result = out.data_ptr<float>();

    for (int64_t b = 0; b < B; b++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                // Cluster grid position for this pixel
                const int ci = y / stride_h;
                const int cj = x / stride_w;

                const float* pixel = elem + b * C * H * W + y * C * W + x * C;

                // Compute similarities with nearby cluster centers
                float sim_buf[81]; // max 9x9
                int valid = 0;
                int valid_idx[81]; // maps sim index to (di,dj)

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int ni = ci + di;
                        const int nj = cj + dj;

                        if (ni >= 0 && ni < h_s && nj >= 0 && nj < w_s) {
                            const float* center = clst + b * C * h_s * w_s + ni * C * w_s + nj * C;

                            float sim = 0.0f;
                            for (int c = 0; c < C; c++)
                                sim += pixel[c] * center[c];

                            sim_buf[valid] = sim / tau;
                            valid_idx[valid] = (di + radius) * nn + (dj + radius); // flat index
                            valid++;
                        }
                    }
                }

                // Softmax
                float max_sim = sim_buf[0];
                for (int p = 1; p < valid; p++)
                    max_sim = std::max(max_sim, sim_buf[p]);

                float sum_exp = 0.0f;
                for (int p = 0; p < valid; p++)
                    sum_exp += std::exp(sim_buf[p] - max_sim);

                float inv_sum = 1.0f / (sum_exp + 1e-10f);

                // Fill output — all nn*nn positions for this pixel
                float* dst = result + b * nn * nn * H * W + y * nn * nn * W + x * nn * nn;

                // Zero out all positions first
                std::memset(dst, 0, nn * nn * sizeof(float));

                // Fill valid positions
                for (int p = 0; p < valid; p++) {
                    int flat_idx = valid_idx[p];
                    dst[flat_idx] = std::exp(sim_buf[p] - max_sim) * inv_sum;
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Python-facing dispatcher — wraps with CPU feature detection
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
    // Shape and dtype checks
    TORCH_CHECK(elem_feats.dim() == 4, "elem_feats must be 4D (B, C, H, W)");
    TORCH_CHECK(clst_feats.dim() == 4, "clst_feats must be 4D (B, C, h, w)");
    TORCH_CHECK(elem_feats.dtype() == torch::kFloat32, "elem_feats must be float32");
    TORCH_CHECK(clst_feats.dtype() == torch::kFloat32, "clst_feats must be float32");
    TORCH_CHECK(elem_feats.size(0) == clst_feats.size(0), "Batch size mismatch");
    TORCH_CHECK(elem_feats.size(1) == clst_feats.size(1), "Channel mismatch");
    TORCH_CHECK(elem_feats.is_contiguous(), "elem_feats must be contiguous");
    TORCH_CHECK(clst_feats.is_contiguous(), "clst_feats must be contiguous");

#ifdef __AVX512F__
    if (cpu::CPUFeatures::hasAVX512F()) {
        return update_clusters_avx512(elem_feats, clst_feats,
                                       stride_h, stride_w,
                                       radius, tau, normalize);
    }
#endif
#ifdef __AVX2__
    if (cpu::CPUFeatures::hasAVX2()) {
        return update_clusters_avx2(elem_feats, clst_feats,
                                     stride_h, stride_w,
                                     radius, tau, normalize);
    }
#endif
    return update_clusters_generic(elem_feats, clst_feats,
                                    stride_h, stride_w,
                                    radius, tau, normalize);
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

#ifdef __AVX512F__
    if (cpu::CPUFeatures::hasAVX512F()) {
        return assign_pixels_avx512(elem_feats, clst_feats,
                                     stride_h, stride_w, radius, tau);
    }
#endif
#ifdef __AVX2__
    if (cpu::CPUFeatures::hasAVX2()) {
        return assign_pixels_avx2(elem_feats, clst_feats,
                                   stride_h, stride_w, radius, tau);
    }
#endif
    return assign_pixels_generic(elem_feats, clst_feats,
                                  stride_h, stride_w, radius, tau);
}

} // namespace kernel
} // namespace spixrwkv7
