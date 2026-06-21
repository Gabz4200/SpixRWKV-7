// SpixRWKV-7: CPU feature detection
// Adapted from HuggingFace kernels cpu_features.hpp pattern
#pragma once
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace spixrwkv7 {
namespace cpu {

class CPUFeatures {
public:
    static bool hasAVX512F() {
        static const bool result = detectAVX512F();
        return result;
    }

    static bool hasAVX2() {
        static const bool result = detectAVX2();
        return result;
    }

private:
    static bool detectAVX2() {
#ifdef __AVX2__
        return true;
#elif defined(__linux__)
        // CPUID check for AVX2 (ebx bit 5)
        uint32_t eax, ebx, ecx, edx;
        cpuid(7, 0, eax, ebx, ecx, edx);
        return (ebx >> 5) & 1;
#else
        return false;
#endif
    }

    static bool detectAVX512F() {
#ifdef __AVX512F__
        return true;
#elif defined(__linux__)
        // Check OS support via XGETBV (xcr0 bits 5:2 for AVX + 6:5 for AVX512)
        // First check CPUID for AVX512F (ebx bit 16, leaf 7)
        uint32_t eax, ebx, ecx, edx;
        cpuid(7, 0, eax, ebx, ecx, edx);
        if (!((ebx >> 16) & 1)) return false;

        // Check OS XSAVE support
        cpuid(1, 0, eax, ebx, ecx, edx);
        if (!((ecx >> 26) & 1)) return false; // XSAVE

        // XGETBV to verify AVX + AVX512 state enabled by OS
        uint64_t xcr0 = xgetbv(0);
        return (xcr0 & 0xE6) == 0xE6; // XMM(YMM)+OPMASK+ZMM
#else
        return false;
#endif
    }

    static void cpuid(uint32_t leaf, uint32_t subleaf,
                       uint32_t& eax, uint32_t& ebx,
                       uint32_t& ecx, uint32_t& edx) {
#ifdef _MSC_VER
        int info[4];
        __cpuidex(info, leaf, subleaf);
        eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
#elif defined(__GNUC__) || defined(__clang__)
        __asm__ volatile(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(leaf), "c"(subleaf)
        );
#endif
    }

    static uint64_t xgetbv(uint32_t xcr) {
#ifdef _MSC_VER
        return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
        uint32_t eax, edx;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#else
        return 0;
#endif
    }
};

} // namespace cpu
} // namespace spixrwkv7
