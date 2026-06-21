# Contributing to SpixRWKV-7

SpixRWKV-7 is a recurrent vision backbone implemented in native PyTorch. It adapts the RWKV-7 language model recurrence (delta-rule linear attention with input-dependent decay) to 2D image understanding via **Superpixel Tokenization (diffSLIC)**, **Graph-Based Q-Shift** on KNN graphs, bidirectional scanning, gated fusion, **Hilbert-ordered token sequences**, and multi-scale feature output. The architecture supports interpolatable position embeddings, CLS token variants, and stochastic depth, with native support for the **OkLAB** perceptual color space.

This repository contains the inference codebase: model definitions, training convergence tests, a demo script, and a test suite. Training pipelines and pretrained weights are maintained separately.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Architecture Overview](#architecture-overview)
- [Code Structure](#code-structure)
- [Opening Issues](#opening-issues)
- [Pull Request Workflow](#pull-request-workflow)
- [Coding Guidelines](#coding-guidelines)
- [Testing & QA](#testing--qa)
- [Agentic / AI Contribution Policy](#agentic--ai-contribution-policy)
- [Session Closeout](#session-closeout)

## Setup

Requirements: Python 3.11+ and `uv` (or pip).

```bash
# Clone and enter the repository
cd Visual_RWKV7_Pytorch

# Install dependencies (PyTorch CPU via pytorch-cpu index)
uv sync

# Or with pip
pip install torch>=2.12.0 numpy>=1.26.0 pytest>=7.0.0
```

The `pyproject.toml` pins torch to the CPU-only index by default (`https://download.pytorch.org/whl/cpu`). If you need CUDA, replace the index or install torch directly with your CUDA variant.

To build the optimized C++ kernels:
```bash
cd spixrwkv7/kernels && python setup.py build_ext --inplace
```
Requires C++17 compiler, OpenMP, and PyTorch headers. The kernels provide AVX512-optimized RWKV-7 recurrence and diffSLIC operations.

## Usage

The demo script `scripts/demo.py` instantiates a backbone (default: tiny, 192-dim, 3 heads, 12 layers, ~20M params) and runs a forward pass with dummy image input.

```bash
# Run the demo
uv run python scripts/demo.py
```

To use the model in your own code:

```python
from spixrwkv7 import create_vision_rwkv7, ClassificationHead

backbone = create_vision_rwkv7(
    img_size=224,
    embed_dims=192,
    num_heads=3,
    depth=12,
    out_indices=[3, 5, 7, 11],
)
head = ClassificationHead(embed_dims=192, num_classes=1000)

x = torch.randn(2, 6, 224, 224)  # 6 channels: Lab + alpha + xy
outs = backbone(x)  # tuple of feature maps per out_indices
logits = head(outs[-1])
```

To verify the architecture converges:

```bash
# Single-batch overfit test (~30s on CPU)
uv run python tasks/diagnostics/fast_test_training.py

# Full systematic diagnostics
uv run python tasks/diagnostics/diagnose_training.py --all
```

## Architecture Overview

SpixRWKV-7 spine (`Vision_RWKV7_Block`) processes an image through these design features:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Superpixel Tokenization | Differentiable SLIC (`diffSLIC`) generates irregular tokens adapted to image content |
| 2 | Graph-Based Q-Shift | Multi-head token shift along KNN graph edges (spatial residual) |
| 3 | Bidirectional Scan | Forward + backward RWKV-7 delta-rule recurrence over the superpixel sequence |
| 4 | Gated Fusion | Learned per-token gate blending forward and backward scan outputs |
| 5 | OkLAB Support | Native differentiable OkLAB color space conversion and gamut clipping |
| 6 | Interpolatable PosEmbed | 1D Position embedding resized for variable superpixel counts |
| 7 | Flexible Decay | Input-dependent decay `w = exp(-0.606531 * sigmoid(w_raw))` bounded in (0.545, 1) |
| 8 | Bounded Exponentials | All exponentiated values remain within stable numeric ranges |
| 9 | Extra LayerNorm | Post-attention `att_ln` and post-FFN `ffn_ln` for training stability |
| 10 | Layer Scale | Learnable `gamma1`/`gamma2` per-block scaling (init 1e-5) |
| 11 | Value Residual | `v = v_0 + (v - v_0) * sigmoid(nu)` — lerp between layer-0 values and current |
|| 12 | Modular Decomposition | Each major operation is an independent `nn.Module`: `RecurrentScan`, `SpatialMixer`, `ChannelMix`, `SuperpixelTokenizer` — easy to swap, remove, or replace |
|| 13 | RMSNorm & SwiGLU | Configurable `norm_layer` (`layernorm`/`rmsnorm`) and `act_layer` (`relu2`/`gelu`/`silu`/`swiglu`) support inspired by DINOv3 and LLaMA |
|| 14 | Registers | DINOv2-style learnable register tokens (`register_tokens=N`) prepended to sequence for global context accumulation |
|| 15 | Dynamic Image Size | `img_size` parameter in transforms: `-1` = original resolution, `>0` = scale proportionally to target height |

The backbone (`Vision_RWKV7`) composes `SuperpixelTokenizer` (diffSLIC → embedding → KNN graph → Hilbert reorder), a stack of `Vision_RWKV7_Block`, optional CLS token, and final norm. The recurrence uses a generalized delta rule with decoupled removal/replacement keys and a per-head bonus term.

The `ClassificationHead` is a separate module (GAP → LayerNorm → Linear), keeping the backbone free for dense prediction tasks.

## Code Structure

```
Visual_RWKV7_Pytorch/
  spixrwkv7/           -- Core package (package name: spixrwkv7)
    __init__.py        -- Public API exports
    models/
      spixrwkv7.py     -- All modules: Vision_RWKV7, Vision_RWKV7_Block,
                            SuperpixelTokenizer, SuperpixelEmbedding,
                            SpatialMixer, RecurrentScan, ChannelMix,
                            ClassificationHead, _DynamicOffset, _TimeMixParams,
                            create_vision_rwkv7
    data/
      colors.py        -- OkLAB/sRGB conversion utilities
      gamut.py         -- OkLAB gamut clipping methods
      diff_slic.py     -- Differentiable SLIC implementation
      transforms.py    -- Image preprocessing utilities
    layers/
      graph.py         -- KNN graph construction and Graph Q-Shift
      drop.py          -- Stochastic depth (DropPath)
    utils/
      __init__.py      -- Utility module init
  tasks/               -- Training scripts organized by task type
    diagnostics/
      fast_test_training.py    -- Single-batch overfit convergence test
      diagnose_training.py     -- Systematic training diagnostics
    classification/
      humordb/
        train.py               -- HumorDB funniness regression training
        infer.py               -- HumorDB checkpoint inference + metrics
    segmentation/
      ade20k/
        sanity.py              -- ADE20K fast CPU overfit test
        train.py               -- ADE20K semantic segmentation training
  scripts/
    demo.py                -- Demo script
    visualize_model.py       -- Model feature visualization
    visualize_superpixels.py -- Superpixel + KNN graph visualization
    debug_nan.py           -- NaN debugging utilities
  tests/
    test_models/
      test_model.py        -- Backbone and block invariants
    test_data/
      test_colors.py       -- Color space conversion tests
      test_diff_slic.py    -- diffSLIC mechanics
      test_transforms.py   -- Transform utilities
    test_layers/
      __init__.py
    test_utils/
      __init__.py
    test_regression.py     -- Numerical stability and regression checks
    __init__.py
  configs/
    model/
      tiny.yaml            -- Tiny config (192-dim, 12 layers)
      small.yaml           -- Small config
      medium.yaml          -- Medium config
      large.yaml           -- Large config
    task/
      humordb.yaml         -- HumorDB training config
      ade20k.yaml          -- ADE20K training config
  pyproject.toml           -- Project metadata and dependencies
  README.md                -- Quick-start instructions, training results
  CONTRIBUTING.md          -- This file
  .agents/
    AGENTS.md              -- AI-specific contribution instructions (moved from root)
```

Key files:

- **`spixrwkv7/models/spixrwkv7.py`** — The primary architecture file. Contains all core modules organized in a clear reading order: utility classes → `RecurrentScan` → `SpatialMixer` → `ChannelMix` → `Vision_RWKV7_Block` → `SuperpixelEmbedding` → `SuperpixelTokenizer` → `Vision_RWKV7` → `ClassificationHead` → builder.
- **`spixrwkv7/data/diff_slic.py`** — Handles the irregular tokenization logic.
- **`spixrwkv7/layers/graph.py`** — KNN graph construction and Graph Q-Shift.
- **`tasks/`** — Training convergence tests and task-specific training scripts. `tasks/classification/humordb/train.py`/`infer.py` demonstrate the full pipeline: HuggingFace dataset loading, disk caching of preprocessed 6-channel tensors, regression head, checkpointing, and metrics. Not part of the core inference package.
- **`tests/`** — Granular tests for each subsystem (96 tests total).
- **`scripts/demo.py`** — Standalone demo showing model instantiation, forward pass, parameter count, and determinism verification.

## Opening Issues

- **Bug reports**: Include the full error trace, Python/PyTorch versions, and a minimal reproduction.
- **Feature requests**: Describe the use case, desired API, and any relevant prior art (VRWKV6, RWKV-7 paper, etc.).
- **Performance concerns**: Include profiling output or benchmark numbers.

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep the scope narrow — a PR should address exactly one concern.
3. Run the existing test suite (see [Testing & QA](#testing--qa)).
4. Add tests for new functionality or bug fixes.
5. Ensure all tests pass before opening the PR.
6. In the PR description, explain what changed and why. Reference any related issues.
7. CI will run the test suite automatically. The PR must be reviewed by at least one maintainer.

## Coding Guidelines

- **Language**: Python 3.11+. Type hints required for all function signatures (`typing` imports are already present).
- **Style**: Follow PEP 8. Use descriptive names. Prefer explicit `nn.Parameter` definitions over `nn.Linear` where the linear algebra is non-standard (RWKV-7 has many bespoke parameter groups).
- **Imports**: Standard library first, then `torch`, then `torch.nn.functional`, then project modules.
- **Comments**: Document the purpose of each parameter group and the formula it implements (see `RecurrentScan.forward` for the delta-rule annotation pattern).
- **Device**: All tensors must be device-agnostic. Never hardcode `cpu()` or `cuda()`.
- **No global state**: The model should be fully re-entrant. Avoid module-level mutable state.
- **Backwards compatibility**: Do not rename or remove public class names (`Vision_RWKV7`, `Vision_RWKV7_Block`, `ClassificationHead`, `SuperpixelEmbedding`, `create_vision_rwkv7`, `q_shift_graph_multihead`). Add new parameters as optional with sensible defaults.
- **Modularity**: Keep components independent — `ClassificationHead` is separate from `Vision_RWKV7`, `RecurrentScan` handles one direction, etc. Do not merge them.
- **Dual-implementation sync**: The PyTorch implementation (`spixrwkv7/models/spixrwkv7.py`) and the optimized C++ implementation (`spixrwkv7/kernels/optimized_block.py`, `spixrwkv7/kernels/optimized_vision.py`) must be kept in SYNC. Any architectural change to the core model must be reflected in the optimized versions. This is a main priority — if they are not synced at any moment, they MUST be synced.

## Testing & QA

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run the full test suite (103 tests)
uv run pytest

# Run a specific test file
uv run pytest tests/test_models/test_model.py -v

# Run with warnings (useful for catching device/dtype issues)
uv run pytest -v -W all
```

The test suite covers:

- **Model Architecture** (test_model.py) — forward pass finiteness, determinism, multi-scale output, CLS token, gradient flow, superpixel embedding modes, graph Q-shift logic, non-square inputs, dynamic resolution via `spixel_size`, alternative superpixel backends ("grid", "slic", "slico", "lnsnet"), Attention Residuals (AttnRes) modes and gates.
- **RWKV-7-specific features** — decoupled keys (`.r_k`, `.k_k`), vector-valued decay (`.w0`) and ICLR (`.a0`), state update formula, v_first propagation, input-dependent mixing.
- **Color Space Correctness** (test_colors.py) — OkLAB/sRGB conversions, gamut clipping stability, finite gradients.
- **diffSLIC Stability** (test_diff_slic.py) — no NaNs on black/uniform images, gradient flow, soft/hard modes.
- **Data Loading** (test_transforms.py) — dataset mean/std calculation, batch consistency, OkLAB preprocessing.
- **Regression** (test_regression.py) — output shape regression, seed determinism, dtype matching.

### Training Convergence Tests

Beyond pytest, the `tasks/diagnostics/` directory contains training convergence validation:

```bash
# Step 1: Single-batch overfit (fast — ~30s on CPU)
uv run python tasks/diagnostics/fast_test_training.py

# Run with Attention Residuals enabled
uv run python tasks/diagnostics/fast_test_training.py --use-attnres
```
# Step 2: Systematic diagnostics
uv run python tasks/diagnostics/diagnose_training.py --all
```

These are NOT part of the pytest suite (they require a training setup), but should be run before merging architectural changes to verify the model still converges.

### HumorDB Regression Experiment

The `tasks/classification/humordb/train.py` and `tasks/classification/humordb/infer.py` scripts exercise SpixRWKV-7 on a real regression task (HumorDB funniness rating, scale 1–10). They serve as:
- A practical validation of the full inference-to-training pipeline
- A demonstration of the disk caching pattern for HuggingFace datasets with expensive image preprocessing
- A stress test for the architecture on noisy subjective targets

```bash
# Train (20 epochs on CPU, ~2.5h with caching)
uv run python tasks/classification/humordb/train.py --epochs 20

# Evaluate best checkpoint
uv run python tasks/classification/humordb/infer.py

# Rebuild cache from scratch
uv run python tasks/classification/humordb/train.py --rebuild-cache
```

See the [HumorDB Results section in README.md](README.md#humordb-regression-funniness-rating) for findings.

### ADE20K Semantic Segmentation Experiment

The `tasks/segmentation/ade20k/sanity.py` and `tasks/segmentation/ade20k/train.py` scripts exercise SpixRWKV-7 on semantic segmentation (ADE20K, 27K+ images, 3K+ object categories). They serve as:
- A dense prediction validation using `scatter_output=True` for full-resolution feature maps
- A test of the architecture on large-scale segmentation with irregular label space
- A demonstration of streaming dataset handling for CPU training

```bash
# Fast overfit test (128 train / 32 val, ~10 epochs on CPU)
uv run python tasks/segmentation/ade20k/sanity.py --preset tiny --epochs 10

# Full training with streaming
uv run python tasks/segmentation/ade20k/train.py --preset small --epochs 50
```

Key findings:
- ADE20K uses raw `name_ndx` (80–3116+), not the standard 150-class mapping. Use `discover_ade20k_classes()` to build a compressed label map.
- Backbone `scatter_output` features have extreme range `[-1238, 1040]` (tiny config). Add `nn.BatchNorm2d(embed_dims)` before the seg head to normalize.
- Streaming DataLoader: use `num_workers=0` or 1 to avoid warnings after `.take()`.
- Scale presets: tiny (~1.3M), small (~18M), medium (~57M), 100m (~99.5M).

See the [ADE20K Results section in README.md](README.md#ade20k-semantic-segmentation) for findings.

### When adding tests

- Each test should verify exactly one behavior or invariant.
- Use small model configurations (`embed_dims=64`, `depth=2`) for fast iteration.
- Prefer assertions over print-based verification.
- Access parameters by their logical leaf name via `named_parameters()` (e.g., `next(p for n, p in block.named_parameters() if n.endswith('.k_k'))`) instead of hardcoded module paths — this makes tests resilient to module tree refactoring.
- For new architectural features, add at least one test that exercises the feature and one that verifies it integrates with the existing forward pass.

## Agentic / AI Contribution Policy

AI agents (including large language models, code generation tools, and automated coding assistants) are welcome to contribute to this repository under these conditions:

1. **Verify before submitting** — AI-generated code must be run through the existing test suite. A PR that breaks tests will be rejected regardless of authorship.
2. **Match project conventions** — follow the coding guidelines above. Do not introduce alternative patterns, additional abstractions, or unrelated "improvements."
3. **Disclose AI assistance** — if a PR is substantially generated by an AI system, note it in the PR description. This helps reviewers understand the context.
4. **Respect scope** — do not refactor code outside the PR's stated purpose. Do not add documentation, comments, or type annotations that aren't directly related to the change.

Detailed AI-specific rules, prohibited actions, and mandatory checks are in [`.agents/AGENTS.md`](.agents/AGENTS.md).

## Session Closeout

Before closing a PR or marking a change as complete:

- [ ] All modified files are free of debug prints, TODO comments, and commented-out code.
- [ ] If the PR touches `spixrwkv7/models/spixrwkv7.py`, verify `spixrwkv7/kernels/optimized_block.py` and `spixrwkv7/kernels/optimized_vision.py` are updated to maintain dual-implementation sync.
- [ ] If C++ kernel changes were made, rebuild and verify `uv run python scripts/demo.py` still produces finite outputs with `--use-cpp` flag (or verify fallback works).
- [ ] The test suite passes cleanly (103 tests).
- [ ] Any new parameters or public APIs are reflected in the relevant docstrings.
- [ ] No stale branches, merge artifacts, or temporary files remain.
- [ ] If the change affects inference behavior, update `scripts/demo.py` or add a new demo path.
- [ ] If the change touches `spixrwkv7/`, verify `uv run python tasks/diagnostics/fast_test_training.py` still passes.

## License

This project is licensed under GPLv3. By contributing, you agree that your contributions will be licensed under the same license.