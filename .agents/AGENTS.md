# AGENTS.md — AI Agent Instructions for Vision-RWKV-7 PyTorch

This file is for AI coding agents. It supplements `CONTRIBUTING.md` with rules specific to automated contribution workflows.

## Useful Commands

| Action | Command |
|--------|---------|
| Install deps | `uv sync` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_model.py -v` |
| Run demo | `uv run python main.py` |
| Run with warnings | `uv run pytest -v -W all` |

## Key Architecture Facts for Agents

- **Modular package**: The model is organized under the `VisualRWKV7/` package.
- **Core components**:
    - `VisualRWKV7/model.py`: Backbone and block definitions.
    - `VisualRWKV7/diffSLIC.py`: Differentiable superpixel tokenization.
    - `VisualRWKV7/utils/`: Modularized utilities (colors, gamut, graph, data).
- **No training code**: This repo is inference-only. No optimizers, dataloaders, or training loops.
- **Public symbols**: `Vision_RWKV7` (backbone), `Vision_RWKV7_Block` (block), `q_shift_graph_multihead` (shift function). Do not rename or change their signatures without updating all callers.
- **Device-agnostic**: All tensors operate on whatever device they are placed on. Never add `.cuda()` or `.cpu()` calls.
- **HEAD_SIZE = 64** (constant). `TIME_MIX_EXTRA_DIM = 32`.
- **Block init is stateful**: `Vision_RWKV7_Block._init_weights()` depends on `layer_id` and `n_layer` — blocks are NOT identical clones.
- **Backward scan mirror**: `_scan(direction='backward')` flips the sequence, runs the same RWKV-7 recurrence, and flips the output. State is NOT shared between directions.
- **Test patterns**: Tests use small configs (`embed_dims=64`, `depth=2`). Each test function tests one invariant.
- **pytest only**: No other test runner. No doctests, no unittest.TestCase.
- **Numerical Stability**: Use `torch.clamp(var, min=0.0)` before `sqrt` in stats and `1e-8` clamping for L2 norms to prevent NaNs.
- **OkLAB Support**: Native support for perceptual color space via `VisualRWKV7/utils/colors.py`.
- **LSP Compliance**: Always `assert x.grad is not None` before checking finiteness in tests to satisfy Pyright.

## AI Contribution Policy

1. **Dual file structure**: `CONTRIBUTING.md` is the main entry point for both humans and AI readers. This file (`.agents/AGENTS.md`) contains AI-specific rules. Changes to AI policy MUST be made in both files if relevant.
2. **No silent scope creep**: If the assignment asks for a bug fix, do not add features, refactor unrelated code, or restructure files. Any scope expansion must be explicitly requested.
3. **Test suite gating**: Before opening a PR, run the full test suite. A failing test is a blocker. Do not mark a PR as ready if tests fail.
4. **One concern per PR**: Do not bundle independent changes. If a fix touches multiple subsystems, split into separate PRs.
5. **Do not change README.md or CHANGELOG.md** unless explicitly asked.
6. **Public API stability**: `Vision_RWKV7.__init__` parameters, `Vision_RWKV7.forward()` return type, and `q_shift_multihead` signature are considered public API. Changes to defaults or shapes must be documented in the PR description and reflected in tests.

## Coordination Before Coding

- Check for existing open issues and PRs that may overlap with the planned change.
- If the change modifies `model.py` architecture (adds/removes parameters, changes recurrence logic), verify alignment with the RWKV-7 paper and VRWKV6 patterns. The codebase is a port — do not introduce alternative formulations without discussion.
- When in doubt about design intent, read the docstring at the top of `model.py` (lines 1-6) and the block forward docstring.

## Fail-Closed Behavior

- If a test suite run produces unexpected failures after a change, revert the change and investigate. Never suppress or skip tests to make CI pass.
- If a change introduces NaN or Inf outputs (check `torch.isfinite`), it is not safe to merge.
- If the change alters parameter count or output shapes for the default configuration (tiny), these must be reported in the PR description.

## Prohibited Actions

- Do NOT add new dependencies to `pyproject.toml` unless the change absolutely requires them and no alternative exists.
- Do NOT add training code (optimizers, schedulers, data loaders) to this repository. This is an inference codebase.
- Do NOT reformat `model.py` with an auto-formatter that changes the parameter group layout. The block's `__init__` is organized in a specific reading order (RWKV-7 head params, delta rule params, vision additions). Preserve it.
- Do NOT remove the `__slots__` declarations on `Permute` and `DropPath`.
- Do NOT add `# type: ignore` or `# noqa` to silence genuine type errors. Fix the types.

## Mandatory Checks Before PR

- [ ] `uv run pytest` passes all 79+ tests.
- [ ] `uv run python main.py` runs without errors and prints `All outputs finite: True`.
- [ ] No `print()` debug statements, `import pdb`, or `breakpoint()` calls remain.
- [ ] All new functions/classes have type annotations and a docstring.
- [ ] Any new `nn.Parameter` has a comment explaining its role and the formula it participates in.
- [ ] If the PR touches `VisualRWKV7/`, verify that the model still produces deterministic output for identical inputs.
- [ ] LSP diagnostics (`lsp diagnostics "*"`) report 0 errors.
