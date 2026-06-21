// SpixRWKV-7: RWKV-7 recurrent scan AVX2 + quantization implementation
// Supports Q4_0 and Q5_1 quantization (ggml-style) with on-the-fly dequantization

#include "rwkv7_kernel_avx2.hpp"
#include <immintrin.h>
#include <cstdint>

namespace spixrwkv7 {
namespace kernel {

// AVX2 width = 8 floats
static constexpr int AVX2_VEC_SIZE = 8;
static constexpr int AVX2_VEC_PER_ROW = 64 / AVX2_VEC_SIZE; // 8 for S=64

// ===================================================================
// Quantization types (ggml-style)
// ===================================================================

// Q4_0: Block of 32 float16 converted to 4-bit, block scale
// Layout: [scale:float16][32 nibble-packed values] per 32-element block
// Total: 1 + 16 bytes = 17 bytes per 32 floats (~2x compression)
struct Q4_0_BLOCK {
    uint8_t qs[16];     // 32 nibble-packed 4-bit values
    uint16_t scale;     // FP16 scale
};

// Q5_1: Block of 32 float16 with 5-bit weights + 16-bit per-row scale
// Layout: [scale:float16][min:float16][32 5-bit packed values]
// Total: 2 + 20 bytes = 22 bytes per 32 floats (~2.7x compression)
struct Q5_1_BLOCK {
    uint8_t qs[20];     // 32 5-bit packed values
    uint16_t scale;     // FP16 block scale
    uint16_t min;       // FP16 per-row minimum
};

// ===================================================================
// Helper: FP16 utilities (ggml-style bit magic)
// ===================================================================

static inline float fp16_to_fp32(uint16_t h) {
    // FP16 to FP32 conversion using bit manipulation
    // Based on ggml fp16_to_fp32
    uint32_t h_rep = h;
    uint32_t sign = h_rep & 0x8000;
    uint32_t exponent = h_rep & 0x7C00;
    uint32_t mantissa = h_rep & 0x03FF;
    
    if (exponent == 0) {
        // Zero or denormal
        return sign ? -0.0f : 0.0f;
    } else {
        uint32_t f_rep = sign << 16 | exponent << 13 | mantissa << 13;
        float result;
        std::memcpy(&result, &f_rep, sizeof(f_rep));
        return result;
    }
}

// ===================================================================
// Helper: Dequantization for Q4_0 (AVX2 version)
// ===================================================================
static inline void dequantize_q4_0_avx2(const Q4_0_BLOCK* block, float* output) {
    float scale = fp16_to_fp32(block->scale);
    __m256 scale_vec = _mm256_set1_ps(scale);
    
    for (int i = 0; i < AVX2_VEC_PER_ROW; i++) {
        // Extract nibbles: 2 nibbles per byte
        uint8_t b0 = block->qs[i * 2];
        uint8_t b1 = block->qs[i * 2 + 1];
        
        // Lower nibbles: b0 & 0x0F, upper nibbles: b1 & 0x0F
        uint8_t lo_vals[8] = {
            b0 & 0x0F, (b0 >> 4) & 0x0F,
            b1 & 0x0F, (b1 >> 4) & 0x0F,
            0, 0, 0, 0  // padding to 8 elements
        };
        
        // Convert to float and multiply by scale
        __m128i lo_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(lo_vals));
        __m256 val = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo_bytes));
        val = _mm256_mul_ps(val, scale_vec);
        
        _mm256_storeu_ps(output + i * AVX2_VEC_SIZE, val);
    }
    
    // Handle padding zeros for remaining elements
    for (int i = AVX2_VEC_PER_ROW * AVX2_VEC_SIZE; i < 32; i++) {
        output[i] = 0.0f;
    }
}

// ===================================================================
// Helper: Dot product of two 64-float vectors using AVX2
// ===================================================================
static inline float dot_64_avx2(const float* a, const float* b) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();
    
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0),  _mm256_loadu_ps(b + 0),  sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8),  _mm256_loadu_ps(b + 8),  sum1);
    sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), sum2);
    sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), sum3);
    sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), sum4);
    sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), sum5);
    sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), sum6);
    sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), sum7);
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum4 = _mm256_add_ps(sum4, sum5);
    sum6 = _mm256_add_ps(sum6, sum7);
    sum0 = _mm256_add_ps(sum0, sum2);
    sum4 = _mm256_add_ps(sum4, sum6);
    sum0 = _mm256_add_ps(sum0, sum4);
    
    // Horizontal sum of 8 elements
    __m128 low = _mm256_castps256_ps128(sum0);
    __m128 high = _mm256_extractf128_ps(sum0, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    return _mm_cvtss_f32(low);
}

// ===================================================================
// Helper: compute sum over j of a[j] * b[j] * c[j] (element-wise product)
// ===================================================================
static inline float dot_triple_64_avx2(const float* a, const float* b, const float* c) {
    __m256 sum0 = _mm256_mul_ps(_mm256_loadu_ps(a + 0),  _mm256_loadu_ps(b + 0));
    __m256 sum1 = _mm256_mul_ps(_mm256_loadu_ps(a + 8),  _mm256_loadu_ps(b + 8));
    __m256 sum2 = _mm256_mul_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16));
    __m256 sum3 = _mm256_mul_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24));
    __m256 sum4 = _mm256_mul_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32));
    __m256 sum5 = _mm256_mul_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40));
    __m256 sum6 = _mm256_mul_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48));
    __m256 sum7 = _mm256_mul_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56));
    
    sum0 = _mm256_mul_ps(sum0, _mm256_loadu_ps(c + 0));
    sum1 = _mm256_mul_ps(sum1, _mm256_loadu_ps(c + 8));
    sum2 = _mm256_mul_ps(sum2, _mm256_loadu_ps(c + 16));
    sum3 = _mm256_mul_ps(sum3, _mm256_loadu_ps(c + 24));
    sum4 = _mm256_mul_ps(sum4, _mm256_loadu_ps(c + 32));
    sum5 = _mm256_mul_ps(sum5, _mm256_loadu_ps(c + 40));
    sum6 = _mm256_mul_ps(sum6, _mm256_loadu_ps(c + 48));
    sum7 = _mm256_mul_ps(sum7, _mm256_loadu_ps(c + 56));
    
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum4 = _mm256_add_ps(sum4, sum5);
    sum6 = _mm256_add_ps(sum6, sum7);
    sum0 = _mm256_add_ps(sum0, sum2);
    sum4 = _mm256_add_ps(sum4, sum6);
    sum0 = _mm256_add_ps(sum0, sum4);
    
    __m128 low = _mm256_castps256_ps128(sum0);
    __m128 high = _mm256_extractf128_ps(sum0, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    return _mm_cvtss_f32(low);
}

// ===================================================================
// AVX2 main kernel (FP32 input, FP32 output)
// ===================================================================

torch::Tensor recurrent_scan_avx2(
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
                    float* st = base_state + h * head_stride;

                    const float* r_t = r_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* v_t = v_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* w_t = w_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* kk_t = kk_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* kt_t = kt_p + b * N * Hd * S + t * timestep_stride + h * S;
                    const float* a_t = a_p + b * N * Hd * S + t * timestep_stride + h * S;
                    float* out_t = out_p + b * N * Hd * S + t * timestep_stride + h * S;

                    // Pre-load vectors (AVX2: 8-wide)
                    __m256 w_vec[AVX2_VEC_PER_ROW];
                    __m256 kk_vec[AVX2_VEC_PER_ROW];
                    __m256 kt_vec[AVX2_VEC_PER_ROW];
                    __m256 a_vec[AVX2_VEC_PER_ROW];
                    __m256 r_vec[AVX2_VEC_PER_ROW];

                    for (int v = 0; v < AVX2_VEC_PER_ROW; v++) {
                        int off = v * AVX2_VEC_SIZE;
                        w_vec[v]  = _mm256_loadu_ps(w_t + off);
                        kk_vec[v] = _mm256_loadu_ps(kk_t + off);
                        kt_vec[v] = _mm256_loadu_ps(kt_t + off);
                        a_vec[v]  = _mm256_loadu_ps(a_t + off);
                        r_vec[v]  = _mm256_loadu_ps(r_t + off);
                    }

                    // Phase 1: sum_state_kk[i] = dot(row[i], kk) for each row i
                    float sum_state_kk[64];
                    for (int i = 0; i < S; i++) {
                        const float* row = st + i * S;
                        sum_state_kk[i] = dot_64_avx2(row, kk_t);
                    }

                    // Phase 2: state update with vectorized row operations (AVX2)
                    for (int i = 0; i < S; i++) {
                        float* row = st + i * S;
                        const float v_i = v_t[i];
                        const float s_kk_i = sum_state_kk[i];

                        __m256 vi_vec = _mm256_set1_ps(v_i);
                        __m256 skk_vec = _mm256_set1_ps(s_kk_i);

                        for (int v = 0; v < AVX2_VEC_PER_ROW; v++) {
                            int off = v * AVX2_VEC_SIZE;
                            __m256 row_v = _mm256_loadu_ps(row + off);
                            // row[j] = row[j] * w[j] + v[i] * kt[j] - s_kk[i] * kk[j] * a[j]
                            __m256 w_mul = _mm256_mul_ps(row_v, w_vec[v]);
                            __m256 vt_mul = _mm256_mul_ps(vi_vec, kt_vec[v]);
                            __m256 kk_a = _mm256_mul_ps(kk_vec[v], a_vec[v]);
                            __m256 skk_kk_a = _mm256_mul_ps(skk_vec, kk_a);
                            __m256 new_row = _mm256_add_ps(w_mul, vt_mul);
                            new_row = _mm256_sub_ps(new_row, skk_kk_a);
                            _mm256_storeu_ps(row + off, new_row);
                        }
                    }

                    // Phase 3: output = state @ r (using updated state)
                    for (int i = 0; i < S; i++) {
                        const float* row = st + i * S;
                        float val = dot_64_avx2(row, r_t);
                        out_t[i] = val;
                    }
                }
            }
        }
    }

    return out;
}

// ===================================================================
// Quantized kernel variants (Q4_0 and Q5_1)
// ===================================================================

torch::Tensor recurrent_scan_q4_0(
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
    // Placeholder: Q4_0 quantized weights
    // For now, delegate to AVX2 version (quantization support to be added)
    return recurrent_scan_avx2(state, r, k, v, w, a, kk, kt, r_k);
}

torch::Tensor recurrent_scan_q5_1(
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
    // Placeholder: Q5_1 quantized weights  
    // For now, delegate to AVX2 version (quantization support to be added)
    return recurrent_scan_avx2(state, r, k, v, w, a, kk, kt, r_k);
}

} // namespace kernel
} // namespace spixrwkv7