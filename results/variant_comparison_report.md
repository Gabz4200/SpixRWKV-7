# Variant Comparison Report

This report summarizes what was validated after wiring `spix`, `conv`, and `vq` head-to-head into the existing comparison scripts. The goal was to compare convergence speed and inference speed without using `tasks/classification/humordb/`.

## What was modified

- `scripts/compare_architectures.py`: added `--compare-variants`, printed a per-size variant comparison table, and shared helper-based construction.
- `scripts/compare_architectures_alt_vit.py`: restored valid import/config surface, added `--compare-variants`, extended the resolution sweep to run selected variants, and printed memory/throughput breakdowns.
- `tasks/segmentation/ade20k/sanity.py`: added `--compare-variants`, refactored training into a reusable `train_variant()` helper, and added a variant comparison summary table. Also fixed a latent 3D/4D tensor shape bug in `ADE20KStreaming.__iter__` that would crash on later runs.

## Smoke benchmark results

### compare_architectures.py
Command:
uv run python scripts/compare_architectures.py --sizes tiny --runs 1 --warmup 1 --batch-size 1 --img-size 64 --compare-variants "spix conv"

Key results:
- tiny @ 64px
  - spix: 79.07 ms total, 9.74 ms tokenizer, 69.32 ms backbone, 0.77M params
  - conv: 82.60 ms total, 8.43 ms tokenizer, 74.17 ms backbone, 1.02M params
- Fastest variant on this smoke run: spix
- Baseline tokenizer fraction: ~12% for spix tiny
- Resolution sweep also executed for small with 64/128/512/1024
  - tokenizer time grows with resolution
  - backbone+tokenizer both remain finite on this sample run

### compare_architectures_alt_vit.py
Command:
uv run python scripts/compare_architectures_alt_vit.py --sizes tiny --runs 1 --warmup 1 --batch-size 1 --img-size 64 --compare-variants "spix conv"

Key results:
- tiny @ 64px
  - spix: 86.95 ms total, 12.79 ms tokenizer, 74.16 ms backbone, 0.77M params
  - conv: 97.12 ms total, 5.84 ms tokenizer, 91.28 ms backbone, 1.02M params
- Fastest variant on this smoke run: spix
- Baseline tokenizer fraction: ~15% for spix tiny

## Observations and caveats

1. No quality result for `conv` yet
- There is no HumorDB or comparable head-to-head quality result for `conv` in the current result set.
- These smoke runs validate that the integration paths, config resolution, and multi-variant printing are functional, but they do not prove superiority.

2. Speed is resolution- and backend-sensitive
- In both compare scripts, spix was faster on this tiny/64px smoke run.
- The conv path reduced tokenizer time vs spix in the alt-Vit script (5.84 ms vs 12.79 ms) but total time was still higher because of backbone cost and larger parameter count.
- Prior non-head-to-head artifacts showed conv tiny at 512px was around 495.22 ms in one record; that is not directly comparable to spix tiny because model size, resolution, and script backend differ.

3. ADE20K sanity comparison could not complete
- `tasks/segmentation/ade20k/sanity.py --compare-variants spix conv` failed during dataset streaming due to repeated HuggingFace read timeouts.
- Result: no convergence curve for spix vs conv could be produced in this session.
- The script and dataset classes are fixed; the run should succeed on a network-resilient machine or cached HuggingFace environment.

## Conclusions

- Wiring is complete and already producing multi-variant output for speed comparisons.
- Current smoke evidence does not show conv outperforming spix on tiny CPU inference at 64px.
- Meaningful convergence-speed comparison still needs a successful ADE20K sanity or HumorDB-style training run.

## Recommendations

1. Re-run `tasks/segmentation/ade20k/sanity.py --compare-variants spix conv` on a network-stable cache/environment if you want convergence-speed insight.
2. Use default `img_size` for each config; do not benchmark `conv_tiny` at 512px unless that is the intended operating point.
3. If you want, I can next prepare a smaller cached ADE20K local parquet path or switch to the HumorDB inference pipeline for quality comparison.

## TDD Refactor, C++ Kernels Fix and Verification (July 2026 Update)

This update documents the completion of the TDD-focused refactor, bug fixes, and rigorous validation of C++ kernel parity.

### 1. TDD Refactor & Configuration Realignment
- **Unified Backbone Loading**: All model construction routes through the optimized `create_optimized_vision_rwkv7` builder (which maps to `create_vision_rwkv7` signature).
- **Shared Config Loader**: Added `tasks/config_loader.py` exposing `load_model_config` and `build_backbone`, centralizing config resolution via `configs/model/*.yaml`.
- **Refactored Task Scripts**: Refactored `tasks/segmentation/ade20k/sanity.py`, `tasks/segmentation/ade20k/train.py`, and `tasks/classification/humordb/train.py` to use `tasks/config_loader.py`, eliminating the local scale preset dicts (`_SCALES`) and `resolve_scale` blocks. Overrides from the command line are cleanly applied on top of the loaded YAML configurations.
- **Orphaned Task Configs Removed**: Deleted the unused/orphaned `configs/task/` directory.

### 2. C++ `diff_slic` Stride & Layout Bug Fix
- **Root Cause**: The C++ generic `diff_slic` implementations (`update_clusters_generic` and `assign_pixels_generic`) in `spixrwkv7/kernels/cpp/diff_slic_kernel.cpp` incorrectly assumed a channel-last layout `(B, H, W, C)`. However, PyTorch and the rest of the codebase use contiguous channel-first layouts `(B, C, H, W)`. This layout mismatch caused index calculations to retrieve scrambled values, producing massive numerical differences from the reference PyTorch implementation.
- **Resolution**: Redesigned the C++ indexing calculations to respect the contiguous `(B, C, H, W)` layout:
  - Element/pixel features: `elem[b * elem_stride + c * elem_sz + y * W + x]`
  - Cluster features: `clst[b * clst_stride + c * clst_sz + ni * w_s + nj]`
  - Assignment features: `result[b * out_stride + flat * elem_sz + y * W + x]`
  - Stack allocation of size 225 replaced heap vector instantiation in the pixel loop to avoid OpenMP thread contention.

### 3. Verification & Validation Results
- **Parity Verification**: Running `pytest tests/test_kernels/test_kernel_parity.py -v` confirmed 4/4 passing tests:
  - `test_rwkv7_recurrent_scan_cpp_matches_pytorch` -> PASS
  - `test_rwkv7_recurrent_scan_cpp_matches_pytorch_masked` -> PASS
  - `test_diff_slic_update_clusters_cpp_matches_pytorch` -> PASS
  - `test_diff_slic_assign_pixels_cpp_matches_pytorch` -> PASS
- **Global Test Suite**: Run `uv run pytest` successfully, achieving **132/132 passing tests**.
- **Correctness Demo**: Running `scripts/demo.py --use-cpp` completed successfully on full-scale `512x512` input images in ~340s. Outputs are fully finite, deterministic, and verify the C++ optimized recurrent scans and diffSLIC operations are fully functional in high-resolution settings.

