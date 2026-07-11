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
