"""Behavioral parity tests: C++ kernels vs PyTorch reference.

These verify that the optimized C++ kernels in
``spixrwkv7/kernels/rwkv7_kernel.py`` (and the compiled
``spixrwkv7/kernels/cpp/diff_slic_kernel.cpp``) reproduce the exact
numerics of the PyTorch reference paths used by the current models.

If a kernel drifts from the model code, these tests fail — surfacing the
desync the optimized builder must not hide.
"""


import torch

from spixrwkv7.kernels import rwkv7_kernel as K


def _random_recurrent_inputs(B=1, N=8, Hd=2, S=64, device="cpu"):
    state = torch.zeros(B, Hd, S, S, device=device)
    r = torch.randn(B, N, Hd, S, device=device)
    v = torch.randn(B, N, Hd, S, device=device)
    w = torch.rand(B, N, Hd, S, device=device)  # decay weights in (0, 1)
    a = torch.rand(B, N, Hd, S, device=device)
    kk = torch.randn(B, N, Hd, S, device=device)
    kt = torch.randn(B, N, Hd, S, device=device)
    return state, r, v, w, a, kk, kt


def test_rwkv7_recurrent_scan_cpp_matches_pytorch():
    """C++ recurrent scan must equal the PyTorch reference formula."""
    torch.manual_seed(0)
    state, r, v, w, a, kk, kt = _random_recurrent_inputs()

    out_cpp = K.rwkv7_recurrent_scan(
        state.clone(), r, v, w, a, kk, kt, use_cpp=True
    )
    out_ref = K.rwkv7_recurrent_scan(
        state.clone(), r, v, w, a, kk, kt, use_cpp=False
    )

    assert out_cpp.shape == out_ref.shape
    assert torch.allclose(out_cpp, out_ref, rtol=1e-3, atol=1e-4)


def test_rwkv7_recurrent_scan_cpp_matches_pytorch_masked():
    """C++ recurrent scan must respect a token-validity mask like the ref."""
    torch.manual_seed(1)
    B, N, Hd, S = 1, 8, 2, 64
    state, r, v, w, a, kk, kt = _random_recurrent_inputs(B, N, Hd, S)
    mask = torch.ones(B, N, device="cpu")
    mask[0, N // 2:] = 0.0

    out_cpp = K.rwkv7_recurrent_scan(
        state.clone(), r, v, w, a, kk, kt, use_cpp=True, mask=mask
    )
    out_ref = K.rwkv7_recurrent_scan(
        state.clone(), r, v, w, a, kk, kt, use_cpp=False, mask=mask
    )

    assert torch.allclose(out_cpp, out_ref, rtol=1e-3, atol=1e-4)


def test_diff_slic_update_clusters_cpp_matches_pytorch():
    """C++ cluster update must equal the PyTorch diffSLIC reference."""
    torch.manual_seed(2)
    B, C, H, W = 1, 8, 16, 16
    stride = (4, 4)
    elem = torch.randn(B, C, H, W)
    h_s, w_s = H // stride[0], W // stride[1]
    clst = torch.randn(B, C, h_s, w_s)

    out_cpp = K.diff_slic_update_clusters(
        elem, clst, stride, radius=1, tau=0.01, normalize=True, use_cpp=True
    )
    out_ref = K.diff_slic_update_clusters(
        elem, clst, stride, radius=1, tau=0.01, normalize=True, use_cpp=False
    )

    assert out_cpp.shape == out_ref.shape
    assert torch.allclose(out_cpp, out_ref, rtol=1e-3, atol=1e-4)


def test_diff_slic_assign_pixels_cpp_matches_pytorch():
    """C++ pixel assignment must equal the PyTorch diffSLIC reference."""
    torch.manual_seed(3)
    B, C, H, W = 1, 8, 16, 16
    stride = (4, 4)
    elem = torch.randn(B, C, H, W)
    h_s, w_s = H // stride[0], W // stride[1]
    clst = torch.randn(B, C, h_s, w_s)

    out_cpp = K.diff_slic_assign_pixels(
        elem, clst, stride, radius=1, tau=0.01, use_cpp=True
    )
    out_ref = K.diff_slic_assign_pixels(
        elem, clst, stride, radius=1, tau=0.01, use_cpp=False
    )

    assert out_cpp.shape == out_ref.shape
    assert torch.allclose(out_cpp, out_ref, rtol=1e-3, atol=1e-4)
