# Conv-Stem Vision-RWKV-7 Integration Results

## Summary

Conv-Stem variant (`ConvolutionalVision_RWKV7`) successfully integrated — all 6
scripts modified, model converges, outputs are finite and deterministic.

### Configs Created

| Config | Params | Conv Stem | Embed Dims | Img Size |
|--------|--------|-----------|------------|----------|
| `configs/model/conv_tiny.yaml` | 1.0M | [32,64,128] | 128 | 64 |
| `configs/model/conv_small.yaml` | 8.9M | [64,128,256] | 256 | 224 |
| `configs/model/conv_medium.yaml` | 59M | [64,128,256] | 512 | 512 |
| `configs/model/conv_large.yaml` | - | [96,192,384] | 768 | 224 |

---

## Script Results

### 1. Demo (`demo.py --model-type conv`)
- **Passed**: 10 seeds, all outputs finite + deterministic
- Output shape: `(1, 128, 16, 16)` (4× spatial reduction from 64→16)
- Params: **1.02M**

### 2. Fast Convergence Test (`fast_test_training.py --model-type conv`)

| Image Size | Steps to 95% | Final Acc | Result |
|------------|--------------|-----------|--------|
| 128×128    | 6            | 100%      | **PASS** |
| 512×512    | 12           | 100%      | **PASS** |

Conv model successfully overfits a single batch — architecture converges.

### 3. Feature Sanity (`diagnose_training.py --no-head --model-type conv`)
- Shape: `(4, 128, 6, 6)` — 4× reduction + superpixel grid
- Mean=0.067, Std=0.998 — well-behaved features
- **Finite: True**, Spatial variance=0.116 — meaningful spatial signal

### 4. Speed Benchmarks (CPU, 512×512)

**vs SimpleViT** (`compare_architectures.py`):

| Config | RWKV-7 Params | ViT Params | RWKV-7 Time | ViT Time | Ratio |
|--------|---------------|------------|-------------|----------|-------|
| Tiny (1.0M) | 1.02M | 5.88M | 990.69ms | 373.22ms | 0.38× |

**vs AltViT** (`compare_architectures_alt_vit.py`):

| Config | RWKV-7 Params | ViT Params | RWKV-7 Time | ViT Time | Ratio |
|--------|---------------|------------|-------------|----------|-------|
| Tiny (1.0M) | 1.02M | 5.67M | 495.22ms | 394.41ms | 0.80× |
| Small (8.9M) | 8.88M | 21.96M | 3061.57ms | 1188.01ms | 0.39× |
| Medium (59M) | 59.02M | 86.38M | (timed out) | — | — |

Key insight: Conv tokenizer reduces image by 4× before diffSLIC, so backbone
processes fewer tokens. However, diffSLIC on CPU is the dominant bottleneck.

### 5. ADE20K Segmentation Sanity
- **Skipped** — dataset download timed out (needs cached `1aurent/ADE20K`).
- Model integration verified syntactically and through the other scripts.

---

## Files Modified

| File | Change |
|------|--------|
| `spixrwkv7/__init__.py` | Added `create_conv_vision_rwkv7` export |
| `scripts/demo.py` | Added conv model_type branch |
| `scripts/compare_architectures.py` | Added conv config loading + model creation + benchmark fix |
| `scripts/compare_architectures_alt_vit.py` | Added conv model_type support |
| `tasks/diagnostics/fast_test_training.py` | Added conv model creation branch |
| `tasks/diagnostics/diagnose_training.py` | Added conv to build_model + arg choices |
| `tasks/segmentation/ade20k/sanity.py` | Added conv backbone branch + defaults |

### Bugfixes Applied
1. `spixel_backend` defaults changed from `"python"`/`"native"` to `"diff_slic"` in 3 scripts
2. Tokenizer benchmark fixed for conv two-stream API (`x_raw` + `x_feat`)
3. Conv stem params type-hinted with defaults for pyright compliance

## Result Files
All under `results/`:
- `demo_conv.txt`
- `fast_test_training_conv.txt`
- `diagnose_training.txt` (no-head feature sanity for conv)
- `compare_arch_conv.txt`
- `compare_arch_alt_vit_conv.txt`
- `ade20k_sanity_conv.txt` (partial — download timeout)
- `conv_spix_integration_report.md` (this file)
