"""TorchDynamo (torch.compile) JIT support for SpixRWKV-7.

Controls (in priority order):
1. `use_jit` kwarg on create_* functions (per-model)
2. ``SPIXRWKV7_USE_JIT`` env var (global toggle, any truthy value)
3. ``SPIXRWKV7_JIT_FLAGS`` env var → extra C++ compile flags for inductor backend
"""

import logging
import os

import torch

_log = logging.getLogger(__name__)


def _env_use_jit() -> bool:
    val = os.environ.get("SPIXRWKV7_USE_JIT", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _jit_flags() -> list[str]:
    raw = os.environ.get("SPIXRWKV7_JIT_FLAGS", "").strip()
    if raw:
        return raw.split()
    return []


def _inductor_config():
    """Get torch._inductor.config, return None if unavailable."""
    try:
        return getattr(getattr(torch, "_inductor", None), "config", None)
    except Exception:  # noqa: BLE001
        return None


def _pad_inductor_compile_args(extra_flags: list[str]) -> None:
    """Append extra C++ compile flags for inductor's C++ backend."""
    cfg = _inductor_config()
    if cfg is None:
        _log.info("JIT enabled (no inductor config found)")
        return

    # The config path varies across PyTorch versions; try the common ones.
    for attr_path in (["cpp", "compile_args"], ["compile_args"]):
        obj: object = cfg
        for attr in attr_path[:-1]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            target = getattr(obj, attr_path[-1], None)
            if isinstance(target, list):
                target.extend(extra_flags)
                _log.info(
                    "JIT inductor extra compile flags: %s (via inductor.config.%s)",
                    extra_flags,
                    ".".join(attr_path),
                )
                return

    _log.info("JIT enabled (no inductor compile_args config path found)")


def maybe_compile(
    model: torch.nn.Module,
    use_jit: bool = False,
) -> torch.nn.Module:
    """Apply torch.compile when requested.

    Toggle order (first wins):
      1. ``use_jit`` kwarg (per-model override)
      2. ``SPIXRWKV7_USE_JIT`` env var (global toggle, any truthy value)

    When JIT is on, inductor's C++ backend gets extra flags from
    ``SPIXRWKV7_JIT_FLAGS`` (space-separated), with ``-march=native`` as
    the default if no ``-march`` flag is already present.
    """
    enabled = use_jit or _env_use_jit()
    if not enabled:
        return model

    extra_flags = _jit_flags()
    if not any(f.startswith("-march") for f in extra_flags):
        extra_flags += ["-march=native"]

    if extra_flags:
        _pad_inductor_compile_args(extra_flags)

    _log.info("torch.compile enabled for %s", type(model).__name__)
    return torch.compile(model)
