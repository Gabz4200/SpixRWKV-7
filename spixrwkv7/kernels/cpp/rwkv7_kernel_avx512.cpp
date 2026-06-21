// SpixRWKV-7: RWKV-7 recurrent scan AVX512 implementation

#ifdef __AVX512F__

#include "rwkv7_kernel_avx512.hpp"
#include <immintrin.h>

namespace spixrwkv7 {
namespace kernel {

// S = HEAD_SIZE = 64 = 4 × 16 (AVX512 registers per row)
static constexpr int VEC_SIZE = 16;
static constexpr int VEC_PER_ROW = 64 / VEC_SIZE; // 4

// ===================================================================
// Helper: dot product of two 64-float vectors using AVX512
// ===================================================================
static inline float dot_64_avx512(const float* a, const float* b) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a),      _mm512_loadu_ps(b),      sum0);
    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), _mm512_loadu_ps(b + 16), sum1);
    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 32), _mm512_loadu_ps(b + 32), sum2);
    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + 48), _mm512_loadu_ps(b + 48), sum3);

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    return _mm512_reduce_add_ps(sum0);
}

// ===================================================================
// Helper: compute sum over j of a[j] * b[j] * c[j] (element-wise product)
// ===================================================================
static inline float dot_triple_64_avx512(const float* a, const float* b, const float* c) {
    __m512 sum0 = _mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
    __m512 sum1 = _mm512_mul_ps(_mm512_loadu_ps(a + 16), _mm512_loadu_ps(b + 16));
    __m512 sum2 = _mm512_mul_ps(_mm512_loadu_ps(a + 32), _mm512_loadu_ps(b + 32));
    __m512 sum3 = _mm512_mul_ps(_mm512_loadu_ps(a + 48), _mm512_loadu_ps(b + 48));

    sum0 = _mm512_mul_ps(sum0, _mm512_loadu_ps(c));
    sum1 = _mm512_mul_ps(sum1, _mm512_loadu_ps(c + 16));
    sum2 = _mm512_mul_ps(sum2, _mm512_loadu_ps(c + 32));
    sum3 = _mm512_mul_ps(sum3, _mm512_loadu_ps(c + 48));

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    return _mm512_reduce_add_ps(sum0);
}

// ===================================================================
// Main kernel
// ===================================================================

torch::Tensor recurrent_scan_avx512(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k)
{
    const auto B = r.size(0);
    const auto N = r.size(1);
    const auto Hd = r.size(2);
    const auto S = r.size(3);

    auto out = torch::empty({B, N, Hd, S}, r.options());

    float* state_p = state.data_ptr<float>();
    const float* r_p = r.data_ptr<float>();
    const float* v_p = v.data_ptr<float>();
    const float* w_p = w.data_ptr<float>();
    const float* kk_p = kk.data_ptr<float>();
    const float* kt_p = kt.data_ptr<float>();
    const float* a_p = a.data_ptr<float>();
    float* out_p = out.data_ptr<float>();

    const int64_t head_stride = S * S;       // elements per (Hd, S, S)
    const int64_t timestep_stride = Hd * S;
    const int64_t batch_head_stride = Hd * S * S;

    // Pre-fetched vector registers for w, kk, kt, a per timestep+head
    __m512 w_vec[VEC_PER_ROW];
    __m512 kk_vec[VEC_PER_ROW];
    __m512 kt_vec[VEC_PER_ROW];
    __m512 r_vec[VEC_PER_ROW];
    __m512 a_vec[VEC_PER_ROW];
    for (int64_t b = 0; b < B; b++) {
        float* base_state = state_p + b * batch_head_stride;

        for (int64_t t = 0; t < N; t++) {
            for (int64_t h = 0; h < Hd; h++) {
                float* st = base_state + h * head_stride;

                const float* r_t = r_p + b * N * Hd * S + t * timestep_stride + h * S;
                const float* v_t = v_p + b * N * Hd * S + t * timestep_stride + h * S;
                const float* w_t = w_p + b * N * Hd * S + t * timestep_stride + h * S;
                const float* kk_t = kk_p + b * N * Hd * S + t * timestep_stride + h * S;
                const float* kt_t = kt_p + b * N * Hd * S + t * timestep_stride + h * S;
                const float* a_t = a_p + b * N * Hd * S + t * timestep_stride + h * S;
                float* out_t = out_p + b * N * Hd * S + t * timestep_stride + h * S;

                // Load per-timestep vectors
                for (int v = 0; v < VEC_PER_ROW; v++) {
                    int off = v * VEC_SIZE;
                    w_vec[v]  = _mm512_loadu_ps(w_t + off);
                    kk_vec[v] = _mm512_loadu_ps(kk_t + off);
                    kt_vec[v] = _mm512_loadu_ps(kt_t + off);
                    a_vec[v]  = _mm512_loadu_ps(a_t + off);
                    r_vec[v]  = _mm512_loadu_ps(r_t + off);
                }

                // Phase 1: sum_state_kk[i] = dot(row[i], kk) for each row i
                float sum_state_kk[64];
                for (int i = 0; i < S; i++) {
                    const float* row = st + i * S;
                    sum_state_kk[i] = dot_64_avx512(row, kk_t);
                }

                // Phase 2: state update with vectorized row operations
                for (int i = 0; i < S; i++) {
                    float* row = st + i * S;
                    const float v_i = v_t[i];
                    const float s_kk_i = sum_state_kk[i];

                    __m512 vi_vec = _mm512_set1_ps(v_i);
                    __m512 skk_vec = _mm512_set1_ps(s_kk_i);

                    for (int v = 0; v < VEC_PER_ROW; v++) {
                        int off = v * VEC_SIZE;
                        __m512 row_v = _mm512_loadu_ps(row + off);
                        // row[j] = row[j] * w[j] + v[i] * kt[j] - s_kk[i] * kk[j] * a[j]
                        __m512 w_mul = _mm512_mul_ps(row_v, w_vec[v]);
                        __m512 vt_mul = _mm512_mul_ps(vi_vec, kt_vec[v]);
                        __m512 kk_a = _mm512_mul_ps(kk_vec[v], a_vec[v]);
                        __m512 skk_kk_a = _mm512_mul_ps(skk_vec, kk_a);
                        __m512 new_row = _mm512_add_ps(w_mul, vt_mul);
                        new_row = _mm512_sub_ps(new_row, skk_kk_a);
                        _mm512_storeu_ps(row + off, new_row);
                    }
                }

                // Phase 3: output = state @ r (using updated state)
                // NOTE: bonus is NOT included here — added after GroupNorm in Python
                for (int i = 0; i < S; i++) {
                    const float* row = st + i * S;
                    float val = dot_64_avx512(row, r_t);
                    out_t[i] = val;
                }
            }
        }
    }

    return out;
}

} // namespace kernel
} // namespace spixrwkv7

#endif // __AVX512F__
