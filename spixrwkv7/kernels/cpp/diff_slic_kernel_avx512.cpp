// SpixRWKV-7: diffSLIC AVX512 kernel implementation

#ifdef __AVX512F__

#include "diff_slic_kernel_avx512.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>

namespace spixrwkv7 {
namespace kernel {

// Compute dot product of two vectors of length C using AVX512
// Returns accumulated result
static inline float dot_product_avx512(const float* a, const float* b, int64_t C) {
    __m512 sum = _mm512_setzero_ps();
    int64_t c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 va = _mm512_loadu_ps(a + c);
        __m512 vb = _mm512_loadu_ps(b + c);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; c < C; c++) result += a[c] * b[c];
    return result;
}

// Compute dot product with optional L2 normalization factor
// Uses pre-computed 1/norm to avoid re-normalizing center
static inline float dot_product_normalized_avx512(
    const float* a, float inv_norm_a,
    const float* b, float inv_norm_b,
    int64_t C)
{
    __m512 sum = _mm512_setzero_ps();
    __m512 ina = _mm512_set1_ps(inv_norm_a);
    __m512 inb = _mm512_set1_ps(inv_norm_b);
    int64_t c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 va = _mm512_mul_ps(_mm512_loadu_ps(a + c), ina);
        __m512 vb = _mm512_mul_ps(_mm512_loadu_ps(b + c), inb);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; c < C; c++) result += (a[c] * inv_norm_a) * (b[c] * inv_norm_b);
    return result;
}

// L2 norm of a vector using AVX512
static inline float l2_norm_sq_avx512(const float* v, int64_t C) {
    __m512 sum = _mm512_setzero_ps();
    int64_t c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 vv = _mm512_loadu_ps(v + c);
        sum = _mm512_fmadd_ps(vv, vv, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; c < C; c++) result += v[c] * v[c];
    return result;
}

// Weighted sum: dst[c] += weight * pixel[c] for all c
static inline void weighted_add_avx512(float* dst, const float* pixel, float weight, int64_t C) {
    __m512 w = _mm512_set1_ps(weight);
    int64_t c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 vp = _mm512_loadu_ps(pixel + c);
        __m512 vd = _mm512_loadu_ps(dst + c);
        _mm512_storeu_ps(dst + c, _mm512_fmadd_ps(w, vp, vd));
    }
    for (; c < C; c++) dst[c] += weight * pixel[c];
}

// ===================================================================
// AVX512 cluster update
// ===================================================================

torch::Tensor update_clusters_avx512(
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

    constexpr int MAX_PIXELS = 81; // max for radius=4

    for (int64_t b = 0; b < B; b++) {
        for (int i = 0; i < h_s; i++) {
            for (int j = 0; j < w_s; j++) {
                const int cy = i * stride_h;
                const int cx = j * stride_w;

                const float* base_elem = elem + b * C * H * W;
                const float* center = clst + b * C * h_s * w_s + i * C * w_s + j * C;
                float* dst = result + b * C * h_s * w_s + i * C * w_s + j * C;

                // Pre-compute center norm if normalizing
                float inv_center_norm = 1.0f;
                if (normalize) {
                    float norm_sq = l2_norm_sq_avx512(center, C);
                    inv_center_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-8f));
                }

                // Extract window and compute similarities
                float sim_buf[MAX_PIXELS];
                int valid_count = 0;

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int py = cy + di * stride_h;
                        const int px = cx + dj * stride_w;

                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            const float* pixel = base_elem + py * C * W + px * C;

                            float sim;
                            if (normalize) {
                                float p_norm_sq = l2_norm_sq_avx512(pixel, C);
                                float inv_p_norm = 1.0f / std::sqrt(std::max(p_norm_sq, 1e-8f));
                                sim = dot_product_normalized_avx512(
                                    center, inv_center_norm,
                                    pixel, inv_p_norm, C);
                            } else {
                                sim = dot_product_avx512(center, pixel, C);
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

                // Weighted aggregation using AVX512
                std::memset(dst, 0, C * sizeof(float));
                int pixel_idx = 0;

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int py = cy + di * stride_h;
                        const int px = cx + dj * stride_w;

                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            const float* pixel = base_elem + py * C * W + px * C;
                            float weight = std::exp(sim_buf[pixel_idx] - max_sim) * inv_sum;
                            weighted_add_avx512(dst, pixel, weight, C);
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
// AVX512 pixel-to-superpixel assignment
// ===================================================================

torch::Tensor assign_pixels_avx512(
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
                const int ci = y / stride_h;
                const int cj = x / stride_w;

                const float* pixel = elem + b * C * H * W + y * C * W + x * C;
                const float* base_clst = clst + b * C * h_s * w_s;

                float sim_buf[81];
                int valid = 0;
                int valid_idx[81];

                for (int di = -radius; di <= radius; di++) {
                    for (int dj = -radius; dj <= radius; dj++) {
                        const int ni = ci + di;
                        const int nj = cj + dj;

                        if (ni >= 0 && ni < h_s && nj >= 0 && nj < w_s) {
                            const float* center = base_clst + ni * C * w_s + nj * C;
                            float sim = dot_product_avx512(pixel, center, C);
                            sim_buf[valid] = sim / tau;
                            valid_idx[valid] = (di + radius) * nn + (dj + radius);
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

                // Fill output
                float* dst = result + b * nn * nn * H * W + y * nn * nn * W + x * nn * nn;
                std::memset(dst, 0, nn * nn * sizeof(float));

                for (int p = 0; p < valid; p++) {
                    dst[valid_idx[p]] = std::exp(sim_buf[p] - max_sim) * inv_sum;
                }
            }
        }
    }

    return out;
}

} // namespace kernel
} // namespace spixrwkv7

#endif // __AVX512F__
