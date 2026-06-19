"""Helper functions for color manipulation (sRGB, Linear RGB, OkLAB)."""

import torch


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    """Real cube root, handles negatives like C's cbrtf()."""
    eps = torch.finfo(x.dtype).tiny
    return torch.sign(x) * (torch.abs(x) + eps).pow(1.0 / 3.0)


# sRGB <-> Linear RGB


def from_srgb_to_linear_rgb(srgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to Linear RGB. Input (B, C, H, W) with C=3 or 4, values in [0, 1]."""
    srgb = srgb.clamp(0.0, 1.0)
    if srgb.shape[1] == 4:
        rgb = srgb[:, 0:3, :, :]
        alpha = srgb[:, 3:4, :, :]
        linear_rgb = torch.where(
            rgb >= 0.04045,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92,
        )
        return torch.cat([linear_rgb, alpha], dim=1)
    return torch.where(
        srgb >= 0.04045,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
        srgb / 12.92,
    )


def from_linear_rgb_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Convert Linear RGB to sRGB. Input (B, C, H, W) with C=3 or 4, values in [0, 1]."""
    linear_rgb = linear_rgb.clamp(0.0, 1.0)
    if linear_rgb.shape[1] == 4:
        rgb = linear_rgb[:, 0:3, :, :]
        alpha = linear_rgb[:, 3:4, :, :]
        srgb = torch.where(
            rgb >= 0.0031308,
            1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055,
            12.92 * rgb,
        )
        return torch.cat([srgb, alpha], dim=1)
    return torch.where(
        linear_rgb >= 0.0031308,
        1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055,
        12.92 * linear_rgb,
    )


# Linear RGB <-> OkLAB


def from_linear_rgb_to_oklab(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Convert Linear RGB to OkLAB. Input (B, C, H, W) with C=3 or 4."""
    assert linear_rgb.ndim == 4 and linear_rgb.shape[1] in [3, 4]

    r = linear_rgb[:, 0:1, :, :]
    g = linear_rgb[:, 1:2, :, :]
    b = linear_rgb[:, 2:3, :, :]

    # Step 1: Linear RGB -> LMS (element-wise matrix multiply)
    l_lms = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_lms = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_lms = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # Step 2: LMS -> LMS' (cube root)
    l_ = _cbrt(l_lms)
    m_ = _cbrt(m_lms)
    s_ = _cbrt(s_lms)

    # Step 3: LMS' -> OkLAB
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    res = torch.cat([L, a, b_], dim=1)
    if linear_rgb.shape[1] == 4:
        res = torch.cat([res, linear_rgb[:, 3:4, :, :]], dim=1)
    return res


def from_oklab_to_linear_rgb(oklab: torch.Tensor) -> torch.Tensor:
    """Convert OkLAB to Linear RGB. Input (B, C, H, W) with C=3 or 4."""
    assert oklab.ndim == 4 and oklab.shape[1] in [3, 4]

    L = oklab[:, 0:1, :, :]
    a = oklab[:, 1:2, :, :]
    b = oklab[:, 2:3, :, :]

    # Step 1: OkLAB -> LMS'
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Step 2: LMS' -> LMS (cube)
    l_lms = l_ * l_ * l_
    m_lms = m_ * m_ * m_
    s_lms = s_ * s_ * s_

    # Step 3: LMS -> Linear RGB
    r = +4.0767416621 * l_lms - 3.3077115913 * m_lms + 0.2309699292 * s_lms
    g = -1.2684380046 * l_lms + 2.6097574011 * m_lms - 0.3413193965 * s_lms
    b_ = -0.0041960863 * l_lms - 0.7034186147 * m_lms + 1.7076147010 * s_lms

    res = torch.cat([r, g, b_], dim=1)
    if oklab.shape[1] == 4:
        res = torch.cat([res, oklab[:, 3:4, :, :]], dim=1)
    return res
