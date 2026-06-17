# Contributing to Vision-RWKV-7 PyTorch

Vision-RWKV-7 is a vision backbone implemented natively in pure PyTorch, designed as a drop-in replacement for ViT. It adapts the RWKV-7 language model recurrence (delta-rule linear attention with input-dependent decay) to 2D image understanding via **Superpixel Tokenization (diffSLIC)**, **Graph-Based Q-Shift** on KNN graphs, bidirectional scanning, gated fusion, and multi-scale feature output. The architecture supports interpolatable position embeddings, CLS token variants, and stochastic depth, with native support for the **OkLAB** perceptual color space.

This repository contains the inference codebase: model definitions, a demo script, and a test suite. Training pipelines and pretrained weights are maintained separately.

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

## Usage

The demo script `main.py` instantiates a Vision_RWKV7 backbone (default: tiny, 192-dim, 3 heads, 12 layers, ~20M params) and runs a forward pass with dummy image input.

```bash
# Run the demo
uv run python main.py
```

To use the model in your own code:

```python
from model import Vision_RWKV7

model = Vision_RWKV7(
    img_size=224, patch_size=16, in_chans=3,
    embed_dims=192, num_heads=3, depth=12,
    init_values=1e-5, final_norm=True,
    out_indices=[3, 5, 7, 11],
)

x = torch.randn(2, 3, 224, 224)
outs = model(x)  # tuple of feature maps per out_indices
```

## Architecture Overview

Vision-RWKV-7 spine (`Vision_RWKV7_Block`) processes an image through 12 design features:

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
| 12 | Multi-Scale Output | Features scattered back to grid at configurable block indices, reshaped to `(B, C, H, W)` |

The backbone (`Vision_RWKV7`) wraps `diffSLIC`, superpixel embedding, KNN graph construction, a stack of `Vision_RWKV7_Block`, optional CLS token, and final norm. The recurrence uses a generalized delta rule with decoupled removal/replacement keys and a per-head bonus term.

## Code Structure

```
Visual_RWKV7_Pytorch/
  VisualRWKV7/      -- Core package
    model.py        -- Vision_RWKV7 and Vision_RWKV7_Block definitions
    diffSLIC.py     -- Differentiable SLIC implementation
    utils/
      colors.py     -- OkLAB/sRGB conversion utilities
      gamut.py      -- OkLAB gamut clipping methods
      data.py       -- Image loading and dataset statistics
      graph.py      -- KNN graph construction and Graph Q-Shift
      drop.py       -- Stochastic depth (DropPath)
      diffSLIC_funcs.py -- diffSLIC helper kernels
  tests/            -- Comprehensive test suite
    test_model.py   -- Backbone and block invariants
    test_diffSlic.py -- diffSLIC correctness and stability
    test_colors.py  -- Color space conversion and gamut clipping
    test_dataload.py -- Data loading and statistics
    test_regression.py -- Numerical stability and regression checks
  main.py           -- Demo / verification script
  pyproject.toml    -- Project metadata and dependencies
  README.md         -- Quick-start instructions
  CONTRIBUTING.md   -- This file
  .agents/
    AGENTS.md       -- AI-specific contribution instructions
```

- **`VisualRWKV7/model.py`** is the primary architecture file.
- **`VisualRWKV7/diffSLIC.py`** handles the irregular tokenization logic.
- **`VisualRWKV7/utils/`** contains modularized mathematical and data utilities.
- **`tests/`** contains granular tests for each subsystem.
- **`main.py`** is a standalone demo showing model instantiation, forward pass, parameter count, and determinism verification.

## Opening Issues

- **Bug reports**: include the full error trace, Python/PyTorch versions, and a minimal reproduction.
- **Feature requests**: describe the use case, desired API, and any relevant prior art (VRWKV6, RWKV-7 paper, etc.).
- **Performance concerns**: include profiling output or benchmark numbers.

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep the scope narrow — a PR should address exactly one concern.
3. Run the existing test suite (see [Testing](#testing--qa)).
4. Add tests for new functionality or bug fixes.
5. Ensure all tests pass before opening the PR.
6. In the PR description, explain what changed and why. Reference any related issues.
7. CI will run the test suite automatically. The PR must be reviewed by at least one maintainer.

## Coding Guidelines

- **Language**: Python 3.11+. Type hints required for all function signatures (`typing` imports are already present).
- **Style**: Follow PEP 8. Use descriptive names. Prefer explicit `nn.Parameter` definitions over `nn.Linear` where the linear algebra is non-standard (RWKV-7 has many bespoke parameter groups).
- **Imports**: Standard library first, then `torch`, then `torch.nn.functional`, then project modules.
- **Comments**: Document the purpose of each parameter group and the formula it implements (see `_scan` for the delta-rule annotation pattern).
- **Device**: All tensors must be device-agnostic. Never hardcode `cpu()` or `cuda()`.
- **No global state**: The model should be fully re-entrant. Avoid module-level mutable state.
- **Backwards compatibility**: Do not rename or remove public class names (`Vision_RWKV7`, `Vision_RWKV7_Block`, `q_shift_multihead`). Add new parameters as optional with sensible defaults.

## Testing & QA

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/test_colors.py -v

# Run with warnings (useful for catching device/dtype issues)
uv run pytest -v -W all
```

The test suite covers:

- **Color Space Correctness** — verifies OkLAB/sRGB conversions and gamut clipping stability.
- **diffSLIC Stability** — ensures no NaNs on black/uniform images and verifies gradient flow.
- **Graph Q-Shift logic** — verifies token movement along KNN graph edges.
- **Multi-scale indices** — checks that `out_indices` selects the expected block outputs.
- **Dataset Statistics** — verifies mean/std calculation accuracy across batch sizes.
- **Numerical stability** — checks large-resolution inputs produce finite outputs.
- **RWKV-7-specific features** — decay bounds, input-dependent mixing, decoupled keys, bonus term.
- **Determinism** — verifies identical input produces identical output.

When adding tests:
- Each test should verify exactly one behavior or invariant.
- Use small model configurations (`embed_dims=64`, `depth=2`) for fast iteration.
- Prefer assertions over print-based verification.
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
- [ ] The test suite passes cleanly.
- [ ] Any new parameters or public APIs are reflected in the relevant docstrings.
- [ ] No stale branches, merge artifacts, or temporary files remain.
- [ ] If the change affects inference behavior, update `main.py` or add a new demo path.

## License

This project is licensed under GPLv3. By contributing, you agree that your contributions will be licensed under the same license.
