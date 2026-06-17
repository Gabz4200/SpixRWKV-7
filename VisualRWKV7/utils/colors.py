"""Helper functions for color manipulation."""

import torch


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    """Real (as in no Complex numbers) cube root, handles negatives like C's cbrtf().

    Uses eps inside abs() to avoid NaN gradients at x=0 during backprop.
    The forward value is numerically indistinguishable from true cbrt for
    any realistic pixel value.
    """
    eps = torch.finfo(x.dtype).tiny  # ~1.18e-38 for float32
    return torch.sign(x) * (torch.abs(x) + eps).pow(1.0 / 3.0)


# sRGB <-> Linear RGB
# All constants came from: https://bottosson.github.io/posts/colorwrong/


def from_srgb_to_linear_rgb(srgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to Linear RGB.

    Implements f_inv(x):
        x >= 0.04045  →  ((x + 0.055) / 1.055) ^ 2.4
        x <  0.04045  →  x / 12.92

    Args:
        srgb (torch.Tensor): Tensor in (B, C, H, W) format with values in [0, 1].

    Returns:
        torch.Tensor: Tensor in (B, C, H, W) format in Linear RGB.
    """
    srgb = srgb.clamp(0.0, 1.0)
    return torch.where(
        srgb >= 0.04045,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
        srgb / 12.92,
    )


def from_linear_rgb_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Convert Linear RGB to sRGB.

    Implements f(x):
        x >= 0.0031308  →  1.055 * x ^ (1/2.4) - 0.055
        x <  0.0031308  →  12.92 * x

    Args:
        linear_rgb (torch.Tensor): Tensor in (B, C, H, W) format with values in [0, 1].

    Returns:
        torch.Tensor: Tensor in (B, C, H, W) format in sRGB.
    """
    linear_rgb = linear_rgb.clamp(0.0, 1.0)
    return torch.where(
        linear_rgb >= 0.0031308,
        1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055,
        12.92 * linear_rgb,
    )


# Linear RGB <-> OkLAB
# All constanst came from: https://bottosson.github.io/posts/oklab/


def from_linear_rgb_to_oklab(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Convert Linear RGB to OkLAB.

    Implements linear_srgb_to_oklab(c):
        Step 1: linear RGB → LMS via matrix multiply
        Step 2: LMS → LMS' via cube root (cbrtf)
        Step 3: LMS' → OkLAB via matrix multiply

    Fully differentiable. No clamping — out-of-gamut values are preserved
    correctly via _cbrt which mirrors C's cbrtf on negative inputs.

    Args:
        linear_rgb (torch.Tensor): Tensor in (B, C, H, W) format with C=3
            representing (R, G, B) in Linear RGB.

    Returns:
        torch.Tensor: Tensor in (B, C, H, W) format with C=3
            representing (L, a, b) in OkLAB.
    """
    assert linear_rgb.ndim == 4 and linear_rgb.shape[1] == 3

    r = linear_rgb[:, 0:1, :, :]
    g = linear_rgb[:, 1:2, :, :]
    b = linear_rgb[:, 2:3, :, :]

    # Step 1: Linear RGB → LMS
    l_lms = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_lms = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_lms = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # Step 2: LMS → LMS' (cube root, like cbrtf)
    l_ = _cbrt(l_lms)
    m_ = _cbrt(m_lms)
    s_ = _cbrt(s_lms)

    # Step 3: LMS' → OkLAB
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return torch.cat([L, a, b_], dim=1)


def from_oklab_to_linear_rgb(oklab: torch.Tensor) -> torch.Tensor:
    """Convert OkLAB to Linear RGB.

    Implements oklab_to_linear_srgb(c):
        Step 1: OkLAB → LMS' via matrix multiply
        Step 2: LMS' → LMS via cube (l_*l_*l_)
        Step 3: LMS → Linear RGB via matrix multiply

    Fully differentiable. Cubing handles negatives naturally.

    Args:
        oklab (torch.Tensor): Tensor in (B, C, H, W) format with C=3
            representing (L, a, b) in OkLAB.

    Returns:
        torch.Tensor: Tensor in (B, C, H, W) format with C=3
            representing (R, G, B) in Linear RGB.
    """
    assert oklab.ndim == 4 and oklab.shape[1] == 3

    L = oklab[:, 0:1, :, :]
    a = oklab[:, 1:2, :, :]
    b = oklab[:, 2:3, :, :]

    # Step 1: OkLAB → LMS'
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Step 2: LMS' → LMS (cube)
    l_lms = l_ * l_ * l_
    m_lms = m_ * m_ * m_
    s_lms = s_ * s_ * s_

    # Step 3: LMS → Linear RGB
    r = +4.0767416621 * l_lms - 3.3077115913 * m_lms + 0.2309699292 * s_lms
    g = -1.2684380046 * l_lms + 2.6097574011 * m_lms - 0.3413193965 * s_lms
    b_ = -0.0041960863 * l_lms - 0.7034186147 * m_lms + 1.7076147010 * s_lms

    return torch.cat([r, g, b_], dim=1)
