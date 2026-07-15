// SpixRWKV-7: RWKV-7 recurrent scan kernel with ggml-style parallelization
// Implements the core delta-rule recurrence with O(S²) optimization
// using the rank-1 structure of the outer product updates.

#include "rwkv7_kernel.hpp"
#include "cpu_features.hpp"
#include <cstring>
#include <cmath>
#include <vector>
#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace spixrwkv7::kernel;

// ===================================================================
// AVX2 helpers (compile-time gated)
// ===================================================================
#if defined(__AVX2__)
static inline float hsum_avx2(__m256 v) {
    __m128 v_lo = _mm256_castps256_ps128(v);
    __m128 v_hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(v_lo, v_hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static inline float dot_64_avx2(const float* a, const float* b) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0),   _mm256_loadu_ps(b + 0),   sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8),   _mm256_loadu_ps(b + 8),   sum1);
    sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16),  _mm256_loadu_ps(b + 16),  sum2);
    sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24),  _mm256_loadu_ps(b + 24),  sum3);
    sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32),  _mm256_loadu_ps(b + 32),  sum4);
    sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40),  _mm256_loadu_ps(b + 40),  sum5);
    sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48),  _mm256_loadu_ps(b + 48),  sum6);
    sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56),  _mm256_loadu_ps(b + 56),  sum7);

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum4 = _mm256_add_ps(sum4, sum5);
    sum6 = _mm256_add_ps(sum6, sum7);
    sum0 = _mm256_add_ps(sum0, sum2);
    sum4 = _mm256_add_ps(sum4, sum6);
    sum0 = _mm256_add_ps(sum0, sum4);

    return hsum_avx2(sum0);
}
#endif

// ===================================================================
// Generic implementation (any CPU) - GGML-style parallelization
// ===================================================================

torch::Tensor spixrwkv7::kernel::recurrent_scan_generic(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt)
{
    const auto B = r.size(0);
    const auto N = r.size(1);
    const auto Hd = r.size(2);
    const auto S = r.size(3);

    auto out = torch::empty({B, N, Hd, S}, r.options());
    auto state_out = torch::empty_like(state);

    float* state_p = state.data_ptr<float>();
    float* state_out_p = state_out.data_ptr<float>();
    const float* r_p = r.data_ptr<float>();
    const float* v_p = v.data_ptr<float>();
    const float* w_p = w.data_ptr<float>();
    const float* kk_p = kk.data_ptr<float>();
    const float* kt_p = kt.data_ptr<float>();
    const float* a_p = a.data_ptr<float>();
    float* out_p = out.data_ptr<float>();

    const int64_t head_stride = S * S;
    const int64_t timestep_stride = Hd * S;
    const int64_t batch_head_stride = Hd * S * S;

    #pragma omp parallel
    {
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();

        for (int64_t b = 0; b < B; b++) {
            float* base_state = state_p + b * batch_head_stride;

            for (int64_t t = 0; t < N; t++) {
                for (int64_t h = ith; h < Hd; h += nth) {
                    float* state_in = (t == 0) ? base_state : state_out_p + b * batch_head_stride;
                    float* state_t = state_out_p + b * batch_head_stride;

                    const float* r_t = r_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* v_t = v_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* w_t = w_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* kk_t = kk_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* kt_t = kt_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* a_t = a_p + b * N * Hd * S + t * timestep_stride + h * S;
                    float* out_t = out_p + b * N * Hd * S + t * timestep_stride + h * S;

                    float* st_in = state_in + h * head_stride;
                    float* st_out = state_t + h * head_stride;

                    // Phase 1: sum_state_kk[i] = sum_j state[i][j] * kk[j]
                    float sum_state_kk[64];
                    for (int i = 0; i < S; i++) {
                        const float* row = st_in + i * S;
#if defined(__AVX2__)
                        sum_state_kk[i] = dot_64_avx2(row, kk_t);
#else
                        float sum = 0.0f;
                        for (int j = 0; j < S; j++) {
                            sum += row[j] * kk_t[j];
                        }
                        sum_state_kk[i] = sum;
#endif
                    }

                    // Phase 2: new_state[i][j] = old[i][j] * w[j] + v[i] * kt[j]
                    //                            - sum_state_kk[i] * kk[j] * a[j]
#if defined(__AVX2__)
                    __m256 w_vec[8], kk_vec[8], kt_vec[8], a_vec[8];
                    for (int vi = 0; vi < 8; vi++) {
                        int off = vi * 8;
                        w_vec[vi]  = _mm256_loadu_ps(w_t + off);
                        kk_vec[vi] = _mm256_loadu_ps(kk_t + off);
                        kt_vec[vi] = _mm256_loadu_ps(kt_t + off);
                        a_vec[vi]  = _mm256_loadu_ps(a_t + off);
                    }

                    for (int i = 0; i < S; i++) {
                        const float v_i = v_t[i];
                        const float s_kk_i = sum_state_kk[i];

                        __m256 vi_vec = _mm256_set1_ps(v_i);
                        __m256 skk_vec = _mm256_set1_ps(s_kk_i);

                        for (int vi = 0; vi < 8; vi++) {
                            int off = vi * 8;
                            __m256 in_v = _mm256_loadu_ps(st_in + i * S + off);
                            __m256 w_mul = _mm256_mul_ps(in_v, w_vec[vi]);
                            __m256 vt_mul = _mm256_mul_ps(vi_vec, kt_vec[vi]);
                            __m256 kk_a = _mm256_mul_ps(kk_vec[vi], a_vec[vi]);
                            __m256 skk_kk_a = _mm256_mul_ps(skk_vec, kk_a);
                            __m256 new_row = _mm256_add_ps(w_mul, vt_mul);
                            new_row = _mm256_sub_ps(new_row, skk_kk_a);
                            _mm256_storeu_ps(st_out + i * S + off, new_row);
                        }
                    }
#else
                    for (int i = 0; i < S; i++) {
                        float* row = st_out + i * S;
                        const float v_i = v_t[i];
                        const float s_kk_i = sum_state_kk[i];
                        for (int j = 0; j < S; j++) {
                            row[j] = st_in[i * S + j] * w_t[j]
                                   + v_i * kt_t[j]
                                   - s_kk_i * kk_t[j] * a_t[j];
                        }
                    }
#endif

                    // Phase 3: output = state_out @ r
#if defined(__AVX2__)
                    for (int i = 0; i < S; i++) {
                        const float* row = st_out + i * S;
                        out_t[i] = dot_64_avx2(row, r_t);
                    }
#else
                    for (int i = 0; i < S; i++) {
                        const float* row = st_out + i * S;
                        float sum = 0.0f;
                        for (int j = 0; j < S; j++) {
                            sum += row[j] * r_t[j];
                        }
                        out_t[i] = sum;
                    }
#endif
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Dispatcher with state persistence
// ===================================================================

namespace spixrwkv7 {
namespace kernel {

torch::Tensor rwkv7_recurrent_scan(
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt)
{
    TORCH_CHECK(state.dim() == 4, "state must be 4D (B, Hd, S, S)");
    TORCH_CHECK(r.dim() == 4, "r must be 4D (B, N, Hd, S)");
    TORCH_CHECK(state.dtype() == torch::kFloat32, "state must be float32");
    TORCH_CHECK(r.dtype() == torch::kFloat32, "r must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(kk.dtype() == torch::kFloat32, "kk must be float32");
    TORCH_CHECK(kt.dtype() == torch::kFloat32, "kt must be float32");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(kk.is_contiguous(), "kk must be contiguous");
    TORCH_CHECK(kt.is_contiguous(), "kt must be contiguous");

    const auto B = r.size(0);
    const auto Hd = r.size(2);
    const auto S = r.size(3);

    TORCH_CHECK(state.size(0) == B, "Batch size mismatch");
    TORCH_CHECK(state.size(1) == Hd, "Head count mismatch");
    TORCH_CHECK(state.size(2) == S, "State size mismatch (dim 2)");
    TORCH_CHECK(state.size(3) == S, "State size mismatch (dim 3)");
    TORCH_CHECK(S <= 64, "HEAD_SIZE (S) must be <= 64, got ", S);

    return recurrent_scan_generic(state, r, v, w, a, kk, kt);
}

} // namespace kernel
} // namespace spixrwkv7
