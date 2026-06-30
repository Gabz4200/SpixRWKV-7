// SpixRWKV-7: rwkv7_kernel_avx2.cpp
//
// The separate AVX2 recurrent scan implementation has been removed.
// recurrent_scan_generic in rwkv7_kernel.cpp already selects AVX2 code
// paths at compile time via #if defined(__AVX2__) guards and is built
// with -march=native, making this specialisation redundant dead code.
//
// The Q4_0 / Q5_1 quantised dispatch stubs have also been removed:
//   1. They were placeholders that delegated straight to the FP32 path.
//   2. They were bound twice in torch_binding.cpp (unconditionally AND
//      inside #ifdef __AVX2__), causing a pybind11 RuntimeError on any
//      AVX2 machine at import time.
//   3. The fp16_to_fp32 helper they relied on had an exponent-rebias bug
//      (missing +112 adjustment), silently producing denormals.
//
// The corrected fp16_to_fp32 utility lives in rwkv7_kernel_avx2.hpp for
// use by future quantised kernels.

#include "rwkv7_kernel_avx2.hpp"
// No definitions needed — all moved to rwkv7_kernel.cpp.
