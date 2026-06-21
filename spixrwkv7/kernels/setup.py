"""Setup script for building SpixRWKV-7 C++ CPU kernels.

Single-extension build with AVX512 compile-time support. The binary
includes AVX512 code paths but runtime dispatch via cpu_features.hpp
ensures they only execute on compatible CPUs.
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension, CppExtension

cpp_dir = os.path.join(os.path.dirname(__file__), "cpp")

# Check if CUDA is available for building
cuda_available = torch.cuda.is_available() and (CUDA_HOME is not None)

sources = [
    os.path.join(cpp_dir, "torch_binding.cpp"),
    os.path.join(cpp_dir, "rwkv7_kernel.cpp"),
    os.path.join(cpp_dir, "rwkv7_kernel_avx2.cpp"),
    os.path.join(cpp_dir, "rwkv7_kernel_avx512.cpp"),
    os.path.join(cpp_dir, "diff_slic_kernel.cpp"),
    os.path.join(cpp_dir, "diff_slic_kernel_avx2.cpp"),
    os.path.join(cpp_dir, "diff_slic_kernel_avx512.cpp"),
]

if cuda_available:
    sources.extend([
        os.path.join(cpp_dir, "rwkv7_kernel_cuda.cu"),
        os.path.join(cpp_dir, "diff_slic_kernel_cuda.cu"),
    ])

# Follow ggml/llama.cpp pattern: use native march for best supported ISA.
# On Kaby Lake (i5-8250U) this enables AVX2, FMA, F16C.
# AVX512 code paths exist separately but are guarded by #ifdef __AVX512F__
# and will be compiled only if the host CPU supports AVX512.
# GLOG workaround: system glog v0.7+ on Arch defines GLOG_EXPORT/GLOG_NO_EXPORT
# via export.h but logging.h checks these BEFORE including export.h.
extra_compile_args = [
    "-O3",
    "-ffast-math",
    "-fopenmp",
    "-march=native",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-DGLOG_EXPORT_H",
    "-DGLOG_EXPORT=",
    "-DGLOG_NO_EXPORT=",
    "-DGLOG_DEPRECATED=",
]

if cuda_available:
    extra_compile_args.append("-DWT_CUDA")

ext_modules = []
if cuda_available:
    ext_modules.append(
        CUDAExtension(
            "spixrwkv7.kernels._C",
            sources=sources,
            extra_compile_args={
                "cxx": extra_compile_args,
                "nvcc": ["-O3", "--use_fast_math", "-DWT_CUDA"]
            }
        )
    )
else:
    ext_modules.append(
        CppExtension(
            "spixrwkv7.kernels._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    )

setup(
    name="spixrwkv7_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
