# Variant Comparison Audit Summary

Date: 2026-07-11
Scope: `compare_architectures.py`, `compare_architectures_alt_vit.py`, `tasks/segmentation/ade20k/sanity.py`

## Changes Made

- Added `--compare-variants` to all three scripts.
- `tasks/segmentation/ade20k/sanity.py` now trains each requested variant in isolation, collects best validation loss / avg epoch time / target epoch, and prints a comparison table when multiple variants are requested.
- Fixed a latent 4D-shape assumption in `ADE20KStreaming` so streaming dataset tiles remain usable when the preprocessing path returns batched tensors.
- Fixed `compare_architectures_alt_vit.py` so it has a local config loader and does not fail on `ModuleNotFoundError`.

## Verified

- All three scripts show `--compare-variants` in `--help`.
- `uv run pytest` still passes 126 tests.
- Smoke compare runs completed for:
  - `scripts/compare_architectures.py` tiny spix vs conv
  - `scripts/compare_architectures_alt_vit.py` tiny spix vs conv
- Result log written to `results/variant_comparison_report.md`.

## Result

- Current CPU smoke evidence does not show `conv` beating `spix` on tiny 64px inference.
- ADE20K convergence comparison could not complete due to streaming dataset timeouts.
- Quality/convergence conclusion is therefore pending a successful cached ADE20K or HumorDB compare run.
