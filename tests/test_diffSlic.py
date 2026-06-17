"""Complete test and debug of diffSLIC to ensure it works on any image."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from VisualRWKV7.diffSLIC import DiffSLIC, spixel_upsampling, spixel_downsampling


def _run_diffslic_on_multiple_images():
    """Run diffSLIC on multiple image types and return results for visualization."""

    # diffSLIC settings
    n_spixels = 196  # 14x14
    n_iter = 10
    tau = 0.01
    candidate_radius = 1

    diff_slic = DiffSLIC(
        n_spixels=n_spixels,
        n_iter=n_iter,
        tau=tau,
        candidate_radius=candidate_radius,
        normalize=True,
        stable=False,
    )

    # List of test images
    test_images = []

    # 1. Synthetic image - gradient
    grad_img = torch.zeros(1, 3, 224, 224)
    for c in range(3):
        grad_img[0, c, :, :] = torch.linspace(0, 1, 224).unsqueeze(0).repeat(224, 1) * (
            c + 1
        )
    test_images.append(("Gradient", grad_img))

    # 2. Synthetic image - checkerboard
    checker = torch.zeros(1, 3, 224, 224)
    block_size = 16
    for i in range(0, 224, block_size):
        for j in range(0, 224, block_size):
            color = ((i // block_size) + (j // block_size)) % 2
            checker[0, :, i : i + block_size, j : j + block_size] = color
    test_images.append(("Checkerboard", checker))

    # 3. Random image (noisy)
    noise_img = torch.randn(1, 3, 224, 224) * 0.5
    test_images.append(("Random Noise", noise_img))

    # 4. Image with uniform colors
    uniform_img = torch.ones(1, 3, 224, 224) * 0.5
    uniform_img[0, :, :112, :] = 0.2  # Dark half
    uniform_img[0, :, 112:, :] = 0.8  # Light half
    test_images.append(("Uniform Halves", uniform_img))

    # 5. Real image (if available)
    try:
        real_img = Image.open("test_image.jpg").convert("RGB")
        real_img = (
            torch.from_numpy(np.array(real_img)).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        )
        real_img = F.interpolate(real_img, size=(224, 224), mode="bilinear")
        test_images.append(("Real Image", real_img))
    except Exception:
        print("No real image found, skipping...")

    print(f"Testing diffSLIC with {n_spixels} superpixels, {n_iter} iterations\n")
    print("=" * 80)

    results = []

    for img_name, img_tensor in test_images:
        print(f"\nTesting: {img_name}")
        print(f"  Shape: {img_tensor.shape}")
        print(f"  Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        try:
            with torch.no_grad():
                # Run diffSLIC
                clst_feats, p2s_assign, s2p_assign = diff_slic(img_tensor)

                # Get superpixel dimensions
                h_s, w_s = clst_feats.shape[-2:]
                K = h_s * w_s

                print(f"  Generated superpixels: {K} ({h_s}x{w_s})")

                # Convert to labels
                radius = diff_slic.candidate_radius
                neighbor_range = 2 * radius + 1
                hard_assign = (
                    F.one_hot(p2s_assign.argmax(1), neighbor_range**2)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                )

                label_grid = torch.arange(
                    K, dtype=torch.float, device=img_tensor.device
                ).reshape(1, 1, h_s, w_s)

                global_labels = (
                    spixel_upsampling(label_grid, hard_assign, candidate_radius=radius)
                    .squeeze(1)
                    .long()
                )

                # Superpixel statistics
                labels_np = global_labels[0].cpu().numpy()
                unique_labels, counts = np.unique(labels_np, return_counts=True)

                print(f"  Unique labels: {len(unique_labels)}")
                print(f"  Average size: {counts.mean():.1f} pixels")
                print(f"  Min size: {counts.min()} pixels")
                print(f"  Max size: {counts.max()} pixels")

                # Check for empty superpixels
                expected_labels = set(range(K))
                actual_labels = set(unique_labels.flatten())
                missing_labels = expected_labels - actual_labels

                if len(missing_labels) > 0:
                    print(f"  WARNING: {len(missing_labels)} empty superpixels!")
                else:
                    print("  [OK] All superpixels are present")

                results.append(
                    {
                        "name": img_name,
                        "img": img_tensor,
                        "labels": global_labels,
                        "K": K,
                        "unique": len(unique_labels),
                        "missing": len(missing_labels),
                    }
                )

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            results.append({"name": img_name, "error": str(e)})

    print("\n" + "=" * 80)
    print("TEST SUMMARY:")
    print("=" * 80)

    for r in results:
        if "error" in r:
            print(f"[FAIL] {r['name']}: ERROR - {r['error']}")
        else:
            status = "[OK]" if r["missing"] == 0 else "[WARN]"
            print(
                f"{status} {r['name']}: {r['unique']}/{r['K']} superpixels, {r['missing']} missing"
            )

    # Visualization
    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for i, r in enumerate(results):
        if "error" not in r:
            # Original image
            img_np = r["img"][0].permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"{r['name']}\n(Original)")
            axes[0, i].axis("off")

            # Superpixel labels
            labels_color = label_to_rgb(r["labels"][0].cpu().numpy())
            axes[1, i].imshow(labels_color)
            axes[1, i].set_title(f"{r['name']}\n(Superpixels)")
            axes[1, i].axis("off")
        else:
            axes[0, i].text(
                0.5,
                0.5,
                f"ERROR:\n{r['error']}",
                ha="center",
                va="center",
                transform=axes[0, i].transAxes,
            )
            axes[0, i].axis("off")
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("diffslic_test_results.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to: diffslic_test_results.png")
    plt.show()

    return results


def label_to_rgb(labels):
    """Converts integer labels to colored RGB via vectorized array indexing."""
    from matplotlib.colors import hsv_to_rgb

    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)

    hues = np.linspace(0, 1, n_colors, endpoint=False)
    hsv_colors = np.stack([
        hues, np.full(n_colors, 0.8), np.full(n_colors, 0.9)
    ], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)

    idx_map = np.zeros(labels.max() + 1, dtype=int)
    idx_map[unique_labels] = np.arange(n_colors)
    return rgb_colors[idx_map[labels]]


def test_diffslic_nan_safety_with_black_pixels():
    """Verify diffSLIC doesn't produce NaN with all-zero (black) pixels."""
    # All-black image - would trigger 0/0 NaN with unguarded normalize
    x = torch.zeros(1, 3, 64, 64)
    diff_slic = DiffSLIC(
        n_spixels=16,
        n_iter=5,
        tau=0.01,
        candidate_radius=1,
        normalize=True,
    )
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    assert not torch.isnan(clst_feats).any(), "clst_feats has NaN from zero-norm division"
    assert not torch.isnan(p2s_assign).any(), "p2s_assign has NaN from zero-norm division"
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"


def test_with_different_configs():
    """Tests different hyperparameter configurations."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 80)

    # Default test image
    test_img = torch.randn(1, 3, 224, 224) * 0.5

    configs = [
        {"n_spixels": 49, "n_iter": 5, "tau": 0.01},  # Few superpixels
        {"n_spixels": 196, "n_iter": 10, "tau": 0.01},  # Default config
        {"n_spixels": 400, "n_iter": 15, "tau": 0.005},  # Many superpixels
        {"n_spixels": 196, "n_iter": 20, "tau": 0.001},  # Many iterations
    ]

    for i, config in enumerate(configs):
        print(f"\nConfig {i + 1}: {config}")

        diff_slic = DiffSLIC(
            n_spixels=config["n_spixels"],
            n_iter=config["n_iter"],
            tau=config["tau"],
            candidate_radius=1,
            normalize=True,
        )

        try:
            with torch.no_grad():
                clst_feats, p2s_assign, _ = diff_slic(test_img)
                h_s, w_s = clst_feats.shape[-2:]
                K_actual = h_s * w_s
                print(f"  [OK] Success: {K_actual} superpixels generated")
        except Exception as e:
            print(f"  [FAIL] Failure: {str(e)}")


def test_diffslic_soft_assignment_probability():
    """Verify p2s_assign sums to 1 over candidate dimension (softmax probability)."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=5, tau=0.01, candidate_radius=1, normalize=True, stable=True
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        _, p2s_assign, _ = diff_slic(x)
    row_sums = p2s_assign.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "p2s_assign rows should sum to 1 (softmax probability property)"
    )


def test_diffslic_output_shapes():
    """Check output tensor shapes for batch=2 with candidate_radius=1."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=5, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert clst_feats.shape[0] == 2 and clst_feats.shape[1] == 3, "clst_feats batch or channel dim wrong"
    assert clst_feats.shape[2] * clst_feats.shape[3] <= 16, "clst_feats spatial product exceeds n_spixels"
    assert p2s_assign.shape == (2, 9, 64, 64), f"p2s_assign shape mismatch: {p2s_assign.shape}"
    assert s2p_assign is not None, "s2p_assign should not be None with n_iter=5"


def test_diffslic_zero_iter():
    """Check s2p_assign is None and outputs are finite when n_iter=0."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=0, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert s2p_assign is None, "s2p_assign should be None when n_iter=0"
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"


def test_spixel_upsampling_shape():
    """Verify spixel_upsampling restores original spatial resolution."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=3, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    upsampled = spixel_upsampling(clst_feats, p2s_assign, candidate_radius=1)
    assert upsampled.shape == (1, 3, 64, 64), (
        f"Expected (1, 3, 64, 64) but got {upsampled.shape}"
    )


def test_spixel_downsampling_shape():
    """Verify spixel_downsampling produces spixel-resolution output."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=3, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert s2p_assign is not None, "s2p_assign required for downsampling"
    downsampled = spixel_downsampling(x, s2p_assign, candidate_radius=1)
    assert downsampled.shape[-2:] == clst_feats.shape[-2:], (
        f"Expected spatial dims {clst_feats.shape[-2:]} but got {downsampled.shape[-2:]}"
    )


def test_diffslic_gradient_flow():
    """Verify gradients flow through the entire diffSLIC forward."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=3, tau=0.01, candidate_radius=1, normalize=False,
    )
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    clst_feats, p2s_assign, _ = diff_slic(x)
    loss = clst_feats.sum() + p2s_assign.sum()
    loss.backward()
    assert x.grad is not None, "x.grad is None — no gradient flowed"
    assert not x.grad.isnan().any(), "x.grad contains NaN"
    assert torch.isfinite(x.grad).all(), "x.grad has non-finite values"


def test_diffslic_single_superpixel():
    """Check behaviour with a single superpixel (n_spixels=1)."""
    diff_slic = DiffSLIC(
        n_spixels=1, n_iter=3, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        clst_feats, p2s_assign, _ = diff_slic(x)
    assert clst_feats.shape == (1, 3, 1, 1), f"clst_feats shape mismatch: {clst_feats.shape}"
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"


def test_diffslic_non_square_image():
    """Check diffSLIC works on non-square images (width != height)."""
    diff_slic = DiffSLIC(
        n_spixels=16, n_iter=3, tau=0.01, candidate_radius=1, normalize=True,
    )
    x = torch.randn(1, 3, 32, 16)
    with torch.no_grad():
        clst_feats, p2s_assign, s2p_assign = diff_slic(x)
    assert torch.isfinite(clst_feats).all(), "clst_feats has non-finite values"
    assert torch.isfinite(p2s_assign).all(), "p2s_assign has non-finite values"
    assert s2p_assign is None or torch.isfinite(s2p_assign).all(), "s2p_assign has non-finite values"
    assert clst_feats.shape[0] == 1 and clst_feats.shape[1] == 3, "clst_feats batch or channel dim wrong"
    assert p2s_assign.shape[0] == 1 and p2s_assign.shape[1] == 9, "p2s_assign batch or candidate dim wrong"
    assert p2s_assign.shape[-2:] == (32, 16), f"p2s_assign spatial dims wrong: {p2s_assign.shape[-2:]} vs (32, 16)"


if __name__ == "__main__":
    print("STARTING COMPLETE DIFFSLIC TESTS")
    print("=" * 80)

    # Test 1: Multiple images
    results = _run_diffslic_on_multiple_images()

    # Test 2: Different configurations
    test_with_different_configs()

    # Final summary
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    all_passed = all("error" not in r and r["missing"] == 0 for r in results)

    if all_passed:
        print("[OK] ALL TESTS PASSED!")
        print("diffSLIC is working correctly on all images.")
    else:
        print("[WARN] SOME TESTS FAILED")
        print("Check the errors above and adjust the parameters.")
        print("\nSuggestions:")
        print("  - Increase n_iter for more iterations (10-20)")
        print("  - Adjust tau (0.001-0.1) for smoothness control")
        print("  - Check if the image has enough contrast")
        print("  - Reduce n_spixels if there are too many empty superpixels")
