"""Setup script for building SpixRWKV-7 C++ CPU kernels.

Single-extension build with AVX2 compile-time support. The binary
includes AVX2 code paths but runtime dispatch via cpu_features.hpp
ensures they only execute on compatible CPUs.
"""

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

_max_jobs = int(os.environ.get("MAX_JOBS") or os.environ.get("SPIXRWKV7_BUILD_JOBS") or "2")
os.environ["MAX_JOBS"] = str(_max_jobs)

cpp_dir = os.path.join(os.path.dirname(__file__), "cpp")

cuda_available = torch.cuda.is_available() and (CUDA_HOME is not None)

_mkl_root = os.environ.get("MKLROOT", "")
if not _mkl_root or not os.path.isdir(_mkl_root):
    for p in ["/opt/intel/oneapi/mkl/latest", "/opt/intel/oneapi/mkl/2026.0", "/usr"]:
        if os.path.isdir(p):
            _mkl_root = p
            break

_have_mkl = bool(_mkl_root) and os.path.isdir(os.path.join(_mkl_root, "lib", "intel64"))

sources = [
    os.path.join(cpp_dir, "torch_binding.cpp"),
    os.path.join(cpp_dir, "rwkv7_kernel.cpp"),
    os.path.join(cpp_dir, "diff_slic_kernel.cpp"),
]

if cuda_available:
    sources.extend([
        os.path.join(cpp_dir, "rwkv7_kernel_cuda.cu"),
        os.path.join(cpp_dir, "diff_slic_kernel_cuda.cu"),
    ])

extra_compile_args = [
    "-O2",
    "-ffast-math",
    "-march=native",
    "-mtune=native",
    "-fopenmp",
    "-pipe",
    "-fno-lto",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-DGLOG_EXPORT_H",
    "-DGLOG_EXPORT=",
    "-DGLOG_NO_EXPORT=",
    "-DGLOG_DEPRECATED=",
]

if _have_mkl:
    extra_compile_args.append(f"-I{_mkl_root}/include")

if _have_mkl:
    _mkl_lib_dir = os.path.join(_mkl_root, "lib", "intel64")
    extra_link_args = [
        f"-L{_mkl_lib_dir}",
        "-Wl,-rpath," + _mkl_lib_dir,
        "-lmkl_intel_lp64",
        "-lmkl_gnu_thread",
        "-lmkl_core",
        "-lgomp",
        "-lpthread",
        "-ldl",
        "-lm",
    ]
else:
    extra_link_args = []

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
            },
            extra_link_args=extra_link_args,
        )
    )
else:
    ext_modules.append(
        CppExtension(
            "spixrwkv7.kernels._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

setup(
    name="spixrwkv7_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
