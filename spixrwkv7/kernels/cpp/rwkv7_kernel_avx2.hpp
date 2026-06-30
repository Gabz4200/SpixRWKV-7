// SpixRWKV-7: AVX2 kernel header.
//
// The separate recurrent_scan_avx2 implementation has been removed.
// recurrent_scan_generic (rwkv7_kernel.cpp) already selects AVX2 paths
// via #if defined(__AVX2__) inline guards and is compiled with -march=native,
// so this specialisation was dead code.
//
// This header is retained for the fp16 utility used by future quantised paths.
#pragma once
#include <cstdint>
#include <cstring>

namespace spixrwkv7 {
namespace kernel {

// =========================================================
// Q4_0 and Q5_1 block types (ggml-style layout).
// Kept for reference; quantised dispatch is not yet wired up.
// =========================================================

struct Q4_0_BLOCK {
    uint8_t  qs[16];   // 32 nibble-packed 4-bit values
    uint16_t scale;    // FP16 block scale
};

struct Q5_1_BLOCK {
    uint8_t  qs[20];   // 32 5-bit packed values
    uint16_t scale;    // FP16 block scale
    uint16_t min;      // FP16 per-row minimum
};

// =========================================================
// fp16_to_fp32: corrected exponent rebias.
//
// FP16 exponent bias is 15; FP32 is 127.  The field must be shifted
// by +112 when widening.  The original implementation omitted this and
// produced denormals (~1.93e-34 instead of 1.0f for FP16=1.0).
// =========================================================
static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp_f16  = (h >> 10) & 0x1Fu;
    const uint32_t mant     = (uint32_t)(h & 0x03FFu);

    if (exp_f16 == 0u) {
        // Zero or FP16 denormal — map to +/- 0.0f for simplicity.
        return 0.0f;
    }
    // Re-bias exponent: +112 = 127 (FP32 bias) - 15 (FP16 bias)
    const uint32_t exp_f32 = exp_f16 + 112u;
    const uint32_t f_rep   = sign | (exp_f32 << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f_rep, sizeof(f_rep));
    return result;
}

} // namespace kernel
} // namespace spixrwkv7
