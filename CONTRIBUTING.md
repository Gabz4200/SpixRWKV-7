# Contributing to Vision-RWKV-7 PyTorch

Vision-RWKV-7 is a vision backbone implemented natively in pure PyTorch, designed as a drop-in replacement for ViT. It adapts the RWKV-7 language model recurrence (delta-rule linear attention with input-dependent decay) to 2D image understanding via bidirectional scanning, Q-Shift spatial token shifting, gated fusion, and multi-scale feature output. The architecture supports interpolatable position embeddings, CLS token variants, and stochastic depth.

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

Vision-RWKV-7 spine (`Vision_RWKV7_Block`) processes an image through 11 design features:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Q-Shift | 4-directional 2D token shift along channel groups (spatial residual) |
| 2 | Bidirectional Scan | Forward + backward RWKV-7 delta-rule recurrence over the flattened sequence |
| 3 | Gated Fusion | Learned per-token gate blending forward and backward scan outputs |
| 4 | Interpolatable PosEmbed | Position embedding resized via bicubic interpolation for variable-resolution inputs |
| 5 | Flexible Decay | Input-dependent decay `w = exp(-0.606531 * sigmoid(w_raw))` bounded in (0.545, 1) |
| 6 | Bounded Exponentials | All exponentiated values remain within stable numeric ranges |
| 7 | Extra LayerNorm | Post-attention `att_ln` and post-FFN `ffn_ln` for training stability |
| 8 | Layer Scale | Learnable `gamma1`/`gamma2` per-block scaling (init 1e-5) |
| 9 | Value Residual | `v = v_0 + (v - v_0) * sigmoid(nu)` — lerp between layer-0 values and current |
| 10 | Input-Dependent Mixing | Dynamic Q-Shift offsets via low-rank MLP (`time_maa_w1`/`w2`) |
| 11 | Multi-Scale Output | Features extracted at configurable block indices, reshaped to `(B, C, H, W)` |

The backbone (`Vision_RWKV7`) wraps patch embedding, position embedding, a stack of `Vision_RWKV7_Block`, optional CLS token, and final norm. The recurrence uses a generalized delta rule with decoupled removal/replacement keys and a per-head bonus term.

## Code Structure

```
Visual_RWKV7_Pytorch/
  model.py         -- All model code: Vision_RWKV7, Vision_RWKV7_Block,
                     q_shift_multihead, resize_pos_embed, utility classes
  main.py          -- Demo / verification script
  test_model.py    -- pytest test suite
  pyproject.toml   -- Project metadata and dependencies
  README.md        -- Quick-start instructions
  CONTRIBUTING.md  -- This file
  .agents/
    AGENTS.md      -- AI-specific contribution instructions
```

- **`model.py`** is the single source of truth for the architecture. It contains:
  - Utility modules: `Permute`, `DropPath`, `q_shift_multihead()`, `resize_pos_embed()`, `drop_path()`
  - `Vision_RWKV7_Block` (nn.Module) — one transformer-style block with time-mix (bidirectional RWKV-7 recurrence + gated fusion) and channel-mix (ReLU^2 MLP)
  - `Vision_RWKV7` (nn.Module) — the full backbone: patch embed, pos embed, block stack, final norm, multi-scale feature extraction
- **`main.py`** is a standalone demo showing model instantiation, forward pass, parameter count, and determinism verification.
- **`test_model.py`** contains all tests (see [Testing](#testing--qa)).

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

Tests are in `test_model.py` and use `pytest`.

```bash
# Run the full test suite
uv run pytest test_model.py -v

# Run a specific test
uv run pytest test_model.py::test_q_shift_logic -v

# Run with warnings (useful for catching device/dtype issues)
uv run pytest test_model.py -v -W all
```

The test suite covers:

- **Q-Shift correctness** — verifies that 4-directional spatial shifting produces correct pixel movement and zero-padding at boundaries.
- **Multi-scale indices** — checks that `out_indices` selects the expected block outputs with correct feature map shapes.
- **Resolution interpolation** — verifies the model handles input resolutions different from `img_size` (pos-embed interpolation).
- **CLS token behavior** — verifies CLS token inclusion and separate output.
- **v_first propagation** — confirms that Value Residual (`v_first`) changes the block output.
- **Numerical stability** — checks large-resolution inputs produce finite outputs.
- **RWKV-7-specific features** — decay bounds, input-dependent mixing, decoupled keys, bonus term, state update formula.
- **Determinism** — verifies identical input produces identical output (no random state leakage).

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
