# AGENTS.md — AI Agent Instructions for SpixRWKV-7

This file is for AI coding agents. It supplements `CONTRIBUTING.md` with rules specific to automated contribution workflows.

## Useful Commands

| Action | Command |
|--------|---------|
| Install deps | `uv sync` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_models/test_model.py -v` |
| Run demo | `uv run python scripts/demo.py` |
| Run training convergence test | `uv run python tasks/diagnostics/fast_test_training.py` |
| Run full training diagnostics | `uv run python tasks/diagnostics/diagnose_training.py --all` |
| Run with warnings | `uv run pytest -v -W all` |
| Train HumorDB regression | `uv run python tasks/classification/humordb/train.py --epochs 20` |
| Inference on HumorDB checkpoint | `uv run python tasks/classification/humordb/infer.py` |
| Rebuild HumorDB cache | `uv run python tasks/classification/humordb/train.py --rebuild-cache` |
| Run ADE20K sanity test | `uv run python tasks/segmentation/ade20k/sanity.py --preset tiny --epochs 10` |
| Train ADE20K segmentation | `uv run python tasks/segmentation/ade20k/train.py --preset small --epochs 50` |

## Dual Implementation Sync Requirement

**CRITICAL**: This codebase maintains TWO implementations of the core architecture:

1. **PyTorch implementation** (`spixrwkv7/spixrwkv7.py`): `Vision_RWKV7`, `Vision_RWKV7_Block`, `RecurrentScan`, `SpatialMixer`, `ChannelMix`
2. **Optimized C++ implementation** (`spixrwkv7/kernels/optimized_block.py`, `spixrwkv7/kernels/optimized_vision.py`): `OptimizedVision_RWKV7_Block`, `OptimizedVision_RWKV7`, `OptimizedRecurrentScan`

**MAIN PRIORITY**: Any change to the PyTorch implementation MUST be mirrored in the optimized implementation. If they are not synced at any moment, they MUST be synced. This includes:
- Parameter additions/removals in `Vision_RWKV7_Block.__init__` → update `OptimizedVision_RWKV7_Block`
- Recurrence logic changes in `RecurrentScan.forward` → update `OptimizedRecurrentScan.forward`
- Forward pass modifications in `SpatialMixer` → update `OptimizedSpatialMixer`
- Any change to `create_vision_rwkv7` signature → update `create_optimized_vision_rwkv7`

The optimized versions inherit from or compose the PyTorch implementations and delegate the core recurrence to C++ kernels. Breaking this sync will cause incorrect behavior when `--use-cpp` is enabled.

## Key Architecture Facts for Agents

- **Modular package**: The model is organized under the `spixrwkv7/` package.
- **Core components** (`spixrwkv7/spixrwkv7.py`):
    - `Vision_RWKV7`: Full backbone composing tokenizer, blocks, and output projection.
    - `Vision_RWKV7_Block`: Block composing `SpatialMixer` + `ChannelMix` with residual connections.
    - `SuperpixelTokenizer`: End-to-end vision-to-token pipeline (diffSLIC → embedding → graph → Hilbert reorder).
    - `SuperpixelEmbedding`: Pixel-to-token pooling with conv features, centroid encoding, Fourier positional embedding.
    - `SpatialMixer`: Graph Q-shift + `_DynamicOffset` + bidirectional `RecurrentScan` + gated fusion.
    - `RecurrentScan`: Single-direction RWKV-7 delta-rule recurrence with decoupled keys, value residual, group norm.
    - `ChannelMix`: Q-shift gated feed-forward network (ReLU² activation).
    - `ClassificationHead`: Separate classifier (GAP → LayerNorm → Linear) — NOT integrated into backbone.
    - `_DynamicOffset`, `_TimeMixParams`: Internal helpers for time-mixing.
    - `create_vision_rwkv7`: Builder function enforcing 6-channel input (Lab + alpha + xy).
- **Other core files**:
    - `spixrwkv7/data/diff_slic.py`: Differentiable superpixel tokenization.
    - `spixrwkv7/data/colors.py`: OkLAB/sRGB conversion utilities.
    - `spixrwkv7/data/gamut.py`: OkLAB gamut clipping methods.
    - `spixrwkv7/layers/graph.py`: KNN graph construction and Graph Q-Shift.
    - `spixrwkv7/layers/drop.py`: Stochastic depth (DropPath).
    - `spixrwkv7/kernels/`: Optimized C++ kernels (rwkv7_kernel.cpp, diff_slic_kernel.cpp) with Python wrappers and optimized model classes.
    - `spixrwkv7/kernels/optimized_block.py`: `OptimizedRecurrentScan`, `OptimizedSpatialMixer`, `OptimizedVision_RWKV7_Block`.
    - `spixrwkv7/kernels/optimized_vision.py`: `OptimizedVision_RWKV7`, `create_optimized_vision_rwkv7`.
- **No training code in core package**: The `spixrwkv7/` package is inference-only. Training scripts live in `tasks/`.
- **Public symbols** (exported from `spixrwkv7`): `Vision_RWKV7`, `Vision_RWKV7_Block`, `SuperpixelEmbedding`, `ClassificationHead`, `create_vision_rwkv7`, `build_knn_graph`, `q_shift_graph_multihead`, `HEAD_SIZE`, `drop_path`, `DropPath`, `DiffSLIC`, `spixel_upsampling`, `spixel_downsampling`. Do not rename or change their signatures without updating all callers.
- **Architectural enhancements**: `norm_layer` (`layernorm`/`rmsnorm`) and `act_layer` (`relu2`/`gelu`/`silu`/`swiglu`) are configurable across backbone, blocks, and head.
- **Registers**: DINOv2-style register tokens accessible via `register_tokens` parameter (prepended to sequence before blocks).
- **Dynamic img_size**: `img_size` in transforms (default `-1` = original resolution, positive = proportional scale to target height).
- **Device-agnostic**: All tensors operate on whatever device they are placed on. Never add `.cuda()` or `.cpu()` calls.
- **HEAD_SIZE = 64** (constant). `TIME_MIX_EXTRA_DIM = 32`.
- **Block init is stateful**: `Vision_RWKV7_Block._init_weights()` depends on `layer_id` and `n_layer` — blocks are NOT identical clones.
- **Backward scan mirror**: `RecurrentScan.forward(direction='backward')` flips the sequence, runs the same RWKV-7 recurrence, and flips the output. State is NOT shared between directions.
- **Test patterns**: Tests use small configs (`embed_dims=64`, `depth=2`). Each test function tests one invariant. Tests access parameters by leaf-name suffix matching via `named_parameters()` (e.g. `.r_k`, `.k_k`) — never hardcoded module paths — making them resilient to module tree reshuffling.
- **pytest only**: No other test runner. No doctests, no unittest.TestCase.
- **Numerical Stability**: Use `torch.clamp(var, min=0.0)` before `sqrt` in stats and `1e-8` clamping for L2 norms to prevent NaNs.
- **OkLAB Support**: Native support for perceptual color space via `spixrwkv7/data/colors.py`.
- **LSP Compliance**: Always `assert x.grad is not None` before checking finiteness in tests to satisfy Pyright.
- **Training convergence validated**: Single-batch overfit protocol passes (100% in ~48 steps). Systematic diagnostics (LR sweep, depth scaling, seed stability, gradient distribution, feature sanity) all pass. The `tasks/diagnostics/fast_test_training.py` script is the quickest sanity check for architectural changes.
- **Shuffle sensitivity for small models**: Models with <~3M params may fail to learn the training set under full random DataLoader shuffle. The gradient noise overwhelms the tiny recurrent state. Use structured shuffle (HuggingFace `buffer_size=100`) as a crutch for very small regressors, or increase capacity (`embed_dims=192+`, more superpixels). Key experimental evidence: `tasks/classification/humordb/train.py` with `depth=4`, `embed_dims=128`, 36 superpixels (1.24M params) got R² ≈ 0 with full shuffle but R²=0.26 with buffer=100 shuffle.
- **Dataset caching pattern**: For HuggingFace datasets with expensive image preprocessing (OkLAB conversion, diffSLIC), cache preprocessed tensors as individual `.pt` files per split. The `HumorDBCached` class in `tasks/classification/humordb/train.py` is the reference pattern. Use `--rebuild-cache` to force rebuild. Cache build takes ~2 min for 2136 images at 64×64.
- **6-channel fixed input**: Scripts pass the image size to `create_vision_rwkv7` via the `img_size` parameter and preprocess to 6-channel (balanced OkLAB + alpha + normalized xy) tensors using `prepare_balanced_superpixel_features`. The model is always called with `(B, 6, H, W)` input.
- **HumorDB scripts** (`tasks/classification/humordb/train.py`, `tasks/classification/humordb/infer.py`): Full training and inference pipeline for funniness rating regression. Both support `--help` with all arguments documented. Training saves checkpoints (`best_val_loss.pt` + `latest.pt`) and logs per-epoch metrics (R², Pearson r, RMSE, MAE, GradNorm) to `tasks/classification/humordb/checkpoints/history.json`.
- **ADE20K scripts** (`tasks/segmentation/ade20k/sanity.py`, `tasks/segmentation/ade20k/train.py`): Semantic segmentation training with streaming HuggingFace dataset. Key findings:
  - ADE20K raw `name_ndx` values range ~80–3116 (not 0–149). Use `discover_ade20k_classes()` to build compressed label map.
  - Backbone `scatter_output` features have extreme range `[-1238, 1040]` (tiny config). Add `nn.BatchNorm2d(embed_dims)` before seg head.
  - Seg head: 1×1 Conv2d with `bias=False` + preceding BatchNorm2d.
  - Scale presets: tiny (~1.3M), small (~18M), medium (~57M), 100m (~99.5M).
  - Streaming DataLoader: use `num_workers=0` or 1 to avoid warnings after `.take()`.

## AI Contribution Policy

1. **Dual file structure**: `CONTRIBUTING.md` is the main entry point for both humans and AI readers. This file (`.agents/AGENTS.md`) contains AI-specific rules. Changes to AI policy MUST be made in both files if relevant.
2. **No silent scope creep**: If the assignment asks for a bug fix, do not add features, refactor unrelated code, or restructure files. Any scope expansion must be explicitly requested.
3. **Test suite gating**: Before opening a PR, run the full test suite. A failing test is a blocker. Do not mark a PR as ready if tests fail.
4. **One concern per PR**: Do not bundle independent changes. If a fix touches multiple subsystems, split into separate PRs.
5. **Do not change README.md or CHANGELOG.md** unless explicitly asked.
6. **Public API stability**: `Vision_RWKV7.__init__` parameters, `Vision_RWKV7.forward()` return type, and `q_shift_graph_multihead` signature are considered public API. Changes to defaults or shapes must be documented in the PR description and reflected in tests.

## Coordination Before Coding

- Check for existing open issues and PRs that may overlap with the planned change.
- If the change modifies `spixrwkv7.py` architecture (adds/removes parameters, changes recurrence logic), verify alignment with the RWKV-7 paper and VRWKV6 patterns. The codebase is a port — do not introduce alternative formulations without discussion.
- When in doubt about design intent, read the docstring at the top of `spixrwkv7.py` (lines 1-6) and the block forward docstring.

## Fail-Closed Behavior

- If a test suite run produces unexpected failures after a change, revert the change and investigate. Never suppress or skip tests to make CI pass.
- If a change introduces NaN or Inf outputs (check `torch.isfinite`), it is not safe to merge.
- If the change alters parameter count or output shapes for the default configuration (tiny), these must be reported in the PR description.

## Prohibited Actions

- Do NOT add new dependencies to `pyproject.toml` unless the change absolutely requires them and no alternative exists.
- Do NOT modify `spixrwkv7/spixrwkv7.py` without also updating `spixrwkv7/kernels/optimized_block.py` and `spixrwkv7/kernels/optimized_vision.py` to maintain dual-implementation sync. The C++ kernel delegation must match the PyTorch logic exactly.
- Do NOT add training code (optimizers, schedulers, data loaders) to the `spixrwkv7/` core package. This is an inference codebase.
- Do NOT reformat `spixrwkv7.py` with an auto-formatter that changes the parameter group layout. The block's `__init__` is organized in a specific reading order (RWKV-7 head params, delta rule params, vision additions). Preserve it.
- Do NOT remove the `__slots__` declarations on `Permute` and `DropPath`.
- Do NOT add `# type: ignore` or `# noqa` to silence genuine type errors. Fix the types.
- Do NOT bake a classification head into the `Vision_RWKV7` backbone. The `ClassificationHead` is a separate module — keep it that way so the backbone remains usable for dense prediction tasks.
- Do NOT access private module internals (`_DynamicOffset`, `_TimeMixParams`) in tests. Use `named_parameters()` leaf-name matching to find parameters by their logical name (e.g., `.k_k`).

## Mandatory Checks Before PR

- [ ] `uv run pytest` passes all 96+ tests.
- [ ] If the PR touches `spixrwkv7/spixrwkv7.py`, verify `spixrwkv7/kernels/optimized_block.py` and `spixrwkv7/kernels/optimized_vision.py` are updated to maintain dual-implementation sync.
- [ ] If C++ kernel changes were made, rebuild and verify `uv run python scripts/demo.py` still produces finite outputs with `--use-cpp` flag (or verify fallback works).
- [ ] `uv run python scripts/demo.py` runs without errors and prints `All outputs finite: True`.
- [ ] No `print()` debug statements, `import pdb`, or `breakpoint()` calls remain.
- [ ] All new functions/classes have type annotations and a docstring.
- [ ] Any new `nn.Parameter` has a comment explaining its role and the formula it participates in.
- [ ] If the PR touches `spixrwkv7/`, verify that the model still produces deterministic output for identical inputs.
- [ ] LSP diagnostics (`lsp diagnostics "*"`) report 0 errors.
- [ ] If the change affects training behavior, run `uv run python tasks/diagnostics/fast_test_training.py` to verify single-batch overfit still passes.