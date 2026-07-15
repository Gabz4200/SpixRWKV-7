"""Benchmark: C++ vs PyTorch recurrent scan across model sizes.

Measures the RecurrentScan module directly (the hot loop that C++ accelerates)
for each architecture at tiny and small configs.
"""
import sys, time, torch
sys.path.insert(0, "/home/gabz/Projects/SpixRWKV7")

from spixrwkv7.models.spixrwkv7 import RecurrentScan

HEAD_SIZE = 64
configs = [
    ("tiny",  192, 3,  36),
    ("small", 384, 6, 144),
]

print("=" * 72)
print("  RecurrentScan: C++ vs PyTorch (the hot loop)")
print("=" * 72)
print(f"  {'Size':<8} {'D':>5} {'N':>5} {'Hd':>4} {'C++ (ms)':>10} {'Py (ms)':>10} {'Speedup':>9} {'Match':>7}")
print(f"  {'-'*62}")

for name, D, Hd, N in configs:
    scan_cpp = RecurrentScan(D, Hd, 0, 12, use_cpp=True).eval()
    scan_py  = RecurrentScan(D, Hd, 0, 12, use_cpp=False).eval()
    xn = torch.randn(1, N, D)
    xx = torch.randn(1, N, D)
    dm = torch.randn(6, 1, N, D)

    with torch.no_grad():
        for _ in range(2):
            scan_cpp(xn, xx, dm, "forward", None)
            scan_py(xn, xx, dm, "forward", None)

        t0 = time.perf_counter()
        for _ in range(5):
            out_cpp, _ = scan_cpp(xn, xx, dm, "forward", None)
        t_cpp = (time.perf_counter() - t0) / 5 * 1000

        t0 = time.perf_counter()
        for _ in range(5):
            out_py, _ = scan_py(xn, xx, dm, "forward", None)
        t_py = (time.perf_counter() - t0) / 5 * 1000

    close = torch.allclose(out_cpp, out_py, atol=1e-3)
    speedup = t_py / t_cpp if t_cpp > 0 else 0
    print(f"  {name:<8} {D:>5} {N:>5} {Hd:>4} {t_cpp:>10.2f} {t_py:>10.2f} {speedup:>8.2f}x {'PASS' if close else 'FAIL':>7}")

print()
print("  Notes:")
print("  - C++ kernel uses AVX2 FMA on i5-8250U (Kaby Lake)")
print("  - PyTorch uses optimized ATen matrix ops")
print("  - Speedup increases with N (more sequential steps)")
print("  - 'Match' verifies C++ and PyTorch produce same results (atol=1e-3)")
