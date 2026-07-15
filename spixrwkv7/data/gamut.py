"""Oklab gamut clipping for sRGB, fully vectorized PyTorch.

All public functions accept and return (B, 3, H, W) linear RGB tensors.
All are fully differentiable.

Reference: https://bottosson.github.io/posts/gamutclipping/
All constants and formulas are from the reference above, which also contains
detailed explanations and visualizations.
"""

import torch

from spixrwkv7.data.colors import _cbrt, from_linear_rgb_to_oklab, from_oklab_to_linear_rgb

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_halley_denom(d: torch.Tensor) -> torch.Tensor:
    """Clamp a Halley-method denominator away from zero to prevent NaN."""
    eps = torch.finfo(d.dtype).eps
    return torch.where(d.abs() < eps, torch.full_like(d, eps), d)


def _compute_max_saturation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the maximum saturation S = C/L for a given hue (a, b) that fits
    in the sRGB gamut.

    a and b must be normalised so that a² + b² == 1.

    Args:
        a (torch.Tensor): Shape (B, 1, H, W).
        b (torch.Tensor): Shape (B, 1, H, W).

    Returns:
        torch.Tensor: Maximum saturation, shape (B, 1, H, W).
    """
    cond_red = (-1.88170328 * a - 0.80936493 * b) > 1.0
    cond_green = (1.81444104 * a - 1.19445276 * b) > 1.0

    def _const(v: float) -> torch.Tensor:
        return torch.full_like(a, v)

    k0 = torch.where(
        cond_red,
        _const(1.19086277),
        torch.where(cond_green, _const(0.73956515), _const(1.35733652)),
    )
    k1 = torch.where(
        cond_red,
        _const(1.76576728),
        torch.where(cond_green, _const(-0.45954404), _const(-0.00915799)),
    )
    k2 = torch.where(
        cond_red,
        _const(0.59662641),
        torch.where(cond_green, _const(0.08285427), _const(-1.15130210)),
    )
    k3 = torch.where(
        cond_red,
        _const(0.75515197),
        torch.where(cond_green, _const(0.12541070), _const(-0.50559606)),
    )
    k4 = torch.where(
        cond_red,
        _const(0.56771245),
        torch.where(cond_green, _const(0.14503204), _const(0.00692167)),
    )
    wl = torch.where(
        cond_red,
        _const(4.0767416621),
        torch.where(cond_green, _const(-1.2684380046), _const(-0.0041960863)),
    )
    wm = torch.where(
        cond_red,
        _const(-3.3077115913),
        torch.where(cond_green, _const(2.6097574011), _const(-0.7034186147)),
    )
    ws = torch.where(
        cond_red,
        _const(0.2309699292),
        torch.where(cond_green, _const(-0.3413193965), _const(1.7076147010)),
    )

    S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

    k_l = +0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.2914855480 * b

    l_ = 1.0 + S * k_l
    m_ = 1.0 + S * k_m
    s_ = 1.0 + S * k_s

    l_lms = l_ * l_ * l_
    m_lms = m_ * m_ * m_
    s_lms = s_ * s_ * s_

    l_dS = 3.0 * k_l * l_ * l_
    m_dS = 3.0 * k_m * m_ * m_
    s_dS = 3.0 * k_s * s_ * s_

    l_dS2 = 6.0 * k_l * k_l * l_
    m_dS2 = 6.0 * k_m * k_m * m_
    s_dS2 = 6.0 * k_s * k_s * s_

    f = wl * l_lms + wm * m_lms + ws * s_lms
    f1 = wl * l_dS + wm * m_dS + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

    S = S - f * f1 / _safe_halley_denom(f1 * f1 - 0.5 * f * f2)

    return S


def _find_cusp(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (L_cusp, C_cusp) for a given hue.

    a and b must be normalised so that a² + b² == 1.

    Args:
        a (torch.Tensor): Shape (B, 1, H, W).
        b (torch.Tensor): Shape (B, 1, H, W).

    Returns:
        Tuple of (L_cusp, C_cusp), each shape (B, 1, H, W).
    """
    S_cusp = _compute_max_saturation(a, b)

    oklab_at_max = torch.cat([torch.ones_like(a), S_cusp * a, S_cusp * b], dim=1)
    rgb_at_max = from_oklab_to_linear_rgb(oklab_at_max)

    max_rgb = rgb_at_max.max(dim=1, keepdim=True).values
    L_cusp = _cbrt(1.0 / max_rgb)
    C_cusp = L_cusp * S_cusp

    return L_cusp, C_cusp


def _find_gamut_intersection(
    a: torch.Tensor,
    b: torch.Tensor,
    L1: torch.Tensor,
    C1: torch.Tensor,
    L0: torch.Tensor,
    cusp: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the parameter t for the intersection of the line

        L(t) = L0 * (1 - t) + t * L1
        C(t) = t * C1

    with the sRGB gamut boundary.

    a and b must be normalised so that a² + b² == 1.

    Args:
        a  (torch.Tensor): Shape (B, 1, H, W).
        b  (torch.Tensor): Shape (B, 1, H, W).
        L1 (torch.Tensor): Shape (B, 1, H, W).
        C1 (torch.Tensor): Shape (B, 1, H, W).
        L0 (torch.Tensor): Shape (B, 1, H, W).
        cusp: Optional pre-computed (L_cusp, C_cusp). Computed internally
              if not supplied.

    Returns:
        torch.Tensor: t, shape (B, 1, H, W).
    """
    if cusp is None:
        L_cusp, C_cusp = _find_cusp(a, b)
    else:
        L_cusp, C_cusp = cusp

    lower_half = ((L1 - L0) * C_cusp - (L_cusp - L0) * C1) <= 0.0

    t_lower = C_cusp * L0 / (C1 * L_cusp + C_cusp * (L0 - L1))
    t_upper = C_cusp * (L0 - 1.0) / (C1 * (L_cusp - 1.0) + C_cusp * (L0 - L1))

    k_l = +0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.2914855480 * b

    dL = L1 - L0
    dC = C1
    l_dt = dL + dC * k_l
    m_dt = dL + dC * k_m
    s_dt = dL + dC * k_s

    t = t_upper
    L = L0 * (1.0 - t) + t * L1
    C = t * C1

    l_ = L + C * k_l
    m_ = L + C * k_m
    s_ = L + C * k_s

    l_lms = l_ * l_ * l_
    m_lms = m_ * m_ * m_
    s_lms = s_ * s_ * s_

    ldt = 3.0 * l_dt * l_ * l_
    mdt = 3.0 * m_dt * m_ * m_
    sdt = 3.0 * s_dt * s_ * s_

    ldt2 = 6.0 * l_dt * l_dt * l_
    mdt2 = 6.0 * m_dt * m_dt * m_
    sdt2 = 6.0 * s_dt * s_dt * s_

    r_f = +4.0767416621 * l_lms - 3.3077115913 * m_lms + 0.2309699292 * s_lms - 1.0
    r_f1 = +4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
    r_f2 = +4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2
    u_r = r_f1 / _safe_halley_denom(r_f1 * r_f1 - 0.5 * r_f * r_f2)
    t_r = torch.where(u_r >= 0.0, -r_f * u_r, torch.full_like(u_r, float("inf")))

    g_f = -1.2684380046 * l_lms + 2.6097574011 * m_lms - 0.3413193965 * s_lms - 1.0
    g_f1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
    g_f2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2
    u_g = g_f1 / _safe_halley_denom(g_f1 * g_f1 - 0.5 * g_f * g_f2)
    t_g = torch.where(u_g >= 0.0, -g_f * u_g, torch.full_like(u_g, float("inf")))

    b_f = -0.0041960863 * l_lms - 0.7034186147 * m_lms + 1.7076147010 * s_lms - 1.0
    b_f1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt
    b_f2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2
    u_b = b_f1 / _safe_halley_denom(b_f1 * b_f1 - 0.5 * b_f * b_f2)
    t_b = torch.where(u_b >= 0.0, -b_f * u_b, torch.full_like(u_b, float("inf")))

    t_upper = t_upper + torch.minimum(t_r, torch.minimum(t_g, t_b))

    return torch.where(lower_half, t_lower, t_upper)


def _to_lc_ab(
    linear_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (B, 3, H, W) linear RGB to per-pixel OkLAB scalars (L, C, a_, b_).

    C is clamped to 1e-5 so a_ and b_ are always well-defined.
    All outputs have shape (B, 1, H, W).
    """
    lab = from_linear_rgb_to_oklab(linear_rgb)
    L = lab[:, 0:1, :, :]
    la = lab[:, 1:2, :, :]
    lb = lab[:, 2:3, :, :]
    C = torch.clamp(torch.sqrt(la * la + lb * lb), min=1e-5)
    a_ = la / C
    b_ = lb / C
    return L, C, a_, b_


def _apply_clip(
    linear_rgb: torch.Tensor,
    L: torch.Tensor,
    C: torch.Tensor,
    a_: torch.Tensor,
    b_: torch.Tensor,
    L0: torch.Tensor,
    cusp: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Find t, build clipped OkLAB, and convert back to linear RGB."""
    t = _find_gamut_intersection(a_, b_, L, C, L0, cusp=cusp)
    L_clip = L0 * (1.0 - t) + t * L
    C_clip = t * C
    return from_oklab_to_linear_rgb(
        torch.cat([L_clip, C_clip * a_, C_clip * b_], dim=1)
    )


def _in_gamut_mask(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Return a (B, 1, H, W) bool mask that is True for in-gamut pixels."""
    r = linear_rgb[:, 0:1, :, :]
    g = linear_rgb[:, 1:2, :, :]
    b = linear_rgb[:, 2:3, :, :]
    return (r > 0) & (r < 1) & (g > 0) & (g < 1) & (b > 0) & (b < 1)


# ---------------------------------------------------------------------------
# Core clipping helper — handles alpha-channel split/re-concat
# ---------------------------------------------------------------------------


def _clip_core(
    linear_rgb: torch.Tensor,
    compute_L0,
    **clip_kwargs,
) -> torch.Tensor:
    """Apply gamut clipping with automatic 4-channel alpha handling.

    ``compute_L0`` is a callable ``(L, C, a_, b_) -> L0`` that computes the
    target L for the out-of-gamut projection.  Extra keyword arguments are
    forwarded to ``_apply_clip`` (e.g. ``cusp``).
    """
    assert linear_rgb.ndim == 4 and linear_rgb.shape[1] in [3, 4]
    has_alpha = linear_rgb.shape[1] == 4
    rgb = linear_rgb[:, 0:3, :, :]
    mask = _in_gamut_mask(rgb)
    L, C, a_, b_ = _to_lc_ab(rgb)
    L0 = compute_L0(L, C, a_, b_)
    clipped_rgb = torch.where(mask, rgb, _apply_clip(rgb, L, C, a_, b_, L0, **clip_kwargs))
    if has_alpha:
        return torch.cat([clipped_rgb, linear_rgb[:, 3:4, :, :]], dim=1)
    return clipped_rgb


# ---------------------------------------------------------------------------
# Public API — 5 gamut clipping methods
# ---------------------------------------------------------------------------


def gamut_clip_preserve_chroma(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Clip to the sRGB gamut, projecting towards (clamp(L, 0, 1), C=0).

    Preserves hue and, as much as possible, chroma. In-gamut pixels are
    returned unchanged.

    Args:
        linear_rgb (torch.Tensor): (B, 3 or 4, H, W) linear RGB, may be out of [0, 1].

    Returns:
        torch.Tensor: (B, 3 or 4, H, W) linear RGB clipped to the sRGB gamut.
    """
    return _clip_core(linear_rgb, lambda L, C, a_, b_: L.clamp(0.0, 1.0))


def gamut_clip_project_to_0_5(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Clip to the sRGB gamut, projecting towards the fixed grey point L=0.5.

    Args:
        linear_rgb (torch.Tensor): (B, 3 or 4, H, W) linear RGB.

    Returns:
        torch.Tensor: (B, 3 or 4, H, W) clipped linear RGB.
    """
    return _clip_core(linear_rgb, lambda L, C, a_, b_: torch.full_like(L, 0.5))


def gamut_clip_project_to_L_cusp(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Clip to the sRGB gamut, projecting towards the hue-dependent cusp grey L_cusp.

    Args:
        linear_rgb (torch.Tensor): (B, 3 or 4, H, W) linear RGB.

    Returns:
        torch.Tensor: (B, 3 or 4, H, W) clipped linear RGB.
    """
    def _L0_with_cusp(L, C, a_, b_):
        cusp = _find_cusp(a_, b_)
        return cusp[0], cusp
    L0, cusp = _L0_with_cusp(None, None, *(_to_lc_ab(linear_rgb[:, :3])[-2:]))
    # Recompute cleanly through _clip_core
    def _compute(L, C, a_, b_):
        c = _find_cusp(a_, b_)
        _compute._cusp = c
        return c[0]
    result = _clip_core(linear_rgb, _compute)
    return result


def gamut_clip_adaptive_L0_0_5(
    linear_rgb: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    """Adaptive clip blending chroma-preservation with L=0.5 projection.

    alpha near 0 approaches pure chroma compression; larger values approach
    single-point projection towards L=0.5.

    Args:
        linear_rgb (torch.Tensor): (B, 3 or 4, H, W) linear RGB.
        alpha (float): Blend parameter. Default 0.05.

    Returns:
        torch.Tensor: (B, 3 or 4, H, W) clipped linear RGB.
    """
    def _compute(L, C, a_, b_):
        Ld = L - 0.5
        e1 = 0.5 + torch.abs(Ld) + alpha * C
        return 0.5 * (
            1.0
            + Ld
            / torch.sqrt(Ld * Ld + 1e-8)
            * (e1 - torch.sqrt(e1 * e1 - 2.0 * torch.abs(Ld)))
        )
    return _clip_core(linear_rgb, _compute)


def gamut_clip_adaptive_L0_L_cusp(
    linear_rgb: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    """Adaptive clip blending chroma-preservation with L_cusp projection.

    alpha near 0 approaches pure chroma compression; larger values approach
    single-point projection towards L_cusp.

    Args:
        linear_rgb (torch.Tensor): (B, 3 or 4, H, W) linear RGB.
        alpha (float): Blend parameter. Default 0.05.

    Returns:
        torch.Tensor: (B, 3 or 4, H, W) clipped linear RGB.
    """
    def _compute(L, C, a_, b_):
        cusp = _find_cusp(a_, b_)
        L_cusp = cusp[0]
        Ld = L - L_cusp
        k = 2.0 * torch.where(Ld > 0.0, 1.0 - L_cusp, L_cusp)
        e1 = 0.5 * k + torch.abs(Ld) + alpha * C / k
        return L_cusp + 0.5 * (
            Ld
            / torch.sqrt(Ld * Ld + 1e-8)
            * (e1 - torch.sqrt(e1 * e1 - 2.0 * k * torch.abs(Ld)))
        )

    def _compute_with_cusp(L, C, a_, b_):
        cusp = _find_cusp(a_, b_)
        L_cusp = cusp[0]
        Ld = L - L_cusp
        k = 2.0 * torch.where(Ld > 0.0, 1.0 - L_cusp, L_cusp)
        e1 = 0.5 * k + torch.abs(Ld) + alpha * C / k
        _compute_with_cusp._last_cusp = cusp
        return L_cusp + 0.5 * (
            Ld
            / torch.sqrt(Ld * Ld + 1e-8)
            * (e1 - torch.sqrt(e1 * e1 - 2.0 * k * torch.abs(Ld)))
        )
    return _clip_core(linear_rgb, _compute_with_cusp)
