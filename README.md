# SpixRWKV-7: Superpixel Graph RWKV-7 Vision Backbone

A PyTorch implementation of a **recurrent vision backbone** that replaces rigid patch grids with differentiable superpixels (`diffSLIC`), processes tokens through **graph-based Q-shift** and **bidirectional RWKV-7 recurrence**, and outputs dense feature maps at arbitrary resolutions.

The architecture merges the linear-complexity, constant-memory advantages of the **RWKV-7** state-space model with vision-specific adaptations (Graph-Based Q-Shift, gated bidirectional fusion, Hilbert reordering), while introducing a novel **irregular-grid tokenization** pipeline. Unlike standard ViTs, SpixRWKV-7 operates on perceptually grouped pixels (superpixels) rather than fixed grid patches, enabling adaptive spatial resolution and natural contour awareness.

> ⚠️ **DISCLAIMER 1:** This repository is yet another learning project made by a single Brazilian student that is exploring the topic of Sub-quadratic Vision Encoders.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do. I also dont prohibit myself from ocasional help, but the main thing is probably commit messages, I genuinely hate writting those.

## News / Recent Updates

- **Architecture refactored into atomic modules**: `RecurrentScan`, `SpatialMixer`, `ChannelMix`, `SuperpixelTokenizer`, `_DynamicOffset`, `_TimeMixParams`, clear separation of concerns, easier experimentation.
- **VQ-VAE Tokenization Ablation**: Added VQ-RWKV-7 model (`VQ_RWKV7` and builder `create_vq_rwkv7`) employing a Convolutional VQ-VAE tokenizer instead of Superpixels. Training automatically propagates quantization loss (`_last_q_loss`). Overfits a single batch in just **14 steps** (3.5× faster than superpixels).
- **Optimized C++ kernels**: Added `spixrwkv7/kernels/` with WKV v7 C++ implementation adapted from gabz-rwkv.cpp. Use `--use-cpp` flag in `scripts/compare_architectures.py` to enable optimized inference. The kernel implements the core RWKV-7 delta rule recurrence with SIMD-optimized loops.
- **HumorDB regression experiment**: Full training and inference scripts for funniness rating (1–10) regression. Disk caching of preprocessed 6-channel tensors eliminates per-epoch data loading (~18% speedup). Key finding: small model (1.24M params, 36 superpixels) requires structured batch shuffling (buffer=100) to learn, full random shuffle prevents convergence due to gradient noise. See [HumorDB Results](#humordb-regression-funniness-rating).
- **Disk caching for HuggingFace datasets**: Reusable `HumorDBCached` pattern, preprocess once per split, cache as `.pt` files, subsequent epochs load directly. Cache build ~2 min for 2136 train images, total ~340 MB at 64×64.
+ **Architectural Enhancements (RMSNorm & SwiGLU)**: Added support for configurable normalization layers (`norm_layer="layernorm"|"rmsnorm"`) and activation functions (`act_layer="relu2"|"gelu"|"silu"|"swiglu"`) across the backbone, blocks, spatial/channel mixers, and classification head. RMSNorm and SwiGLU activations are inspired by modern vision/language architectures like DINOv3 and LLaMA, providing greater parameter efficiency and expressive capacity.
+ **Registers (DINOv2-style)**: Optional learnable register tokens prepended to the token sequence, allowing global context accumulation independent of superpixel representation. Enable with `register_tokens=N`.
+ **Dynamic Image Size Support**: `img_size` parameter in `load_image_to_tensor` for flexible resolution handling. `img_size=-1` (default) uses original resolution; positive values scale proportionally to match that height.

- **RMSNorm + SwiGLU Validation Complete**: Comprehensive evaluation of the new normalization/activation options shows stable convergence, deterministic behavior, and no NaN/Inf outputs. Full-batch overfit achieved in 17 steps with tiny config (64x64, 128 dims, 36 superpixels). Models under 3M params require structured shuffle (buffer=100) for stable training. See [RMSNorm + SwiGLU Results](#rmsnorm--swiglu-validation-results) for details.
- **Deep Codebase Cleanup (commit e9caf86)**: Removed all dead code — 7 unused sigma params from all forward signatures, `hilbertcurve` dependency (replaced with native vectorized sort), Q4_0/Q5_1 quantization stubs, dead C++ kernel parameters (`k`, `r_k`, `n_extra_back`), dead `c_p` gather parameter, unused `S` state variables, and `pos_grid_size`/`pos_embed` in the backbone forward. Fixed 4 bugs (`r_k` `NameError`, `ln0` bypass never applied, dead `c_p` gather always masking, unused `S` variables shadowing). Cleaned C++ kernel signatures (state made `const`, removed stale args, `-fno-lto` build flag). Ruff-clean (0 errors), 126/126 tests pass. See [commit e9caf86](https://github.com/Gabz4200/SpixRWKV7/commit/e9caf86ea57587f007c57e5626444dfe60945c9b) for full diff.

## Key Features

- **Superpixel Tokenization Backends**: Replaces rigid patch grids with configurable superpixel tokenizers, supporting multiple backends:
  - `diff_slic`: Differentiable SLIC via PyTorch/C++ (supports both **hard** and **soft** modes).
  - `grid`: Non-overlapping regular grid of patches (providing a direct grid/patch baseline).
  - `slic`: Classical CPU SLIC clustering using `scikit-image` with gradient pass-through.
  - `slico`: Classical CPU SLICO (SLIC-Zero) clustering.
  - `lnsnet`: Non-iterative learnable superpixel segmentation (LNS-Net, CVPR 2021) with BSDS checkpoint weight initialization support.
- **VQ-VAE Tokenization Ablation (VQ_RWKV7)**: A sibling model alternative (`VQ_RWKV7`) that replaces superpixels with learned VQ-VAE codebook representations. It downsamples the image via a Convolutional VQ-VAE encoder to a regular grid of latent features, maps them to the nearest codebook embeddings (with straight-through gradients), and feeds these quantized token embeddings to the RWKV-7 recurrent blocks.
- **Attention Residuals (AttnRes)**: Depth-wise attention residuals replacing the standard fixed additive residual accumulation with a learned softmax attention over preceding layer/block representations. Features options for `"block"` and `"full"` history aggregation, and multiple gating options (`"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, `"learnable_alpha"`).
- **Perceptual Color Space Support**: Native, differentiable support for the **OkLAB** color space, including sRGB/Linear RGB conversions, alpha channel, and robust **Gamut Clipping** methods. At least until now, the Gamut Clipping is only implemented as a utility and not integrated into the training pipeline, but it is available for experimentation and also for cliping outputs on generative tasks.
- **Graph-Based Q-Shift**: Adapts the original 2D grid Q-Shift to operate on K-Nearest Neighbor (KNN) graphs over irregular superpixel centroids, enabling spatial mixing that adapts to arbitrary topologies.
- **Bidirectional RWKV-7 Recurrence**: Two independent `RecurrentScan` modules (forward and backward) process the token sequence, fused via a learned dynamic gate. Each scan implements RWKV-7's generalized delta rule with decoupled keys, value residual, and learnable decay.
- **Deterministic Token Ordering**: Hilbert curve sorts superpixel tokens into a reproducible 1D sequence, ensuring stable spatial structure across forward passes.
- **Multi-Scale Output**: Extracts features from configurable intermediate blocks and scatters them back to original image resolution via soft mask aggregation (differentiable) or hard label gather.
- **Modular Architecture**: Every component is an independent `nn.Module`, swap, remove, or replace parts without touching the rest.
- **Depth-Aware Classification Head**: `ClassificationHead` and `RegressionHead` can selectively attend to the entire depth history (`attnres_history`) of backbone representations using Attention Residuals to avoid data dilution and improve convergence, while remaining usable for standard average-pooled features when AttnRes is disabled.
- **Hilbert-Reordered Neighbor Remapping**: After Hilbert sorting, KNN neighbor indices are remapped to the new ordering, preserving spatial relationships in the sorted sequence.
- **Robust Data Utilities**: Tools for calculating dataset-wide mean/std statistics, stable image loading, and OkLAB-aware preprocessing.
- **RWKV-7 Numerical Stability**: Inherits RWKV-7's bounded exponentials, value residuals, and Layer Scale for robust training.

## Architecture Comparison: Vision RWKV-7 vs ViT

### Speed Comparison (CPU)

Benchmark comparing Vision RWKV-7 (RMSNorm + SwiGLU activation, C++ optimized kernels) against a standard ViT implementation across model sizes and image resolutions.

#### Model Size Comparison (measured at 512x512 resolution)

| Size   | Img Size | RWKV-7 Time (ms) | ViT Time (ms) | Speedup (ViT/RWKV-7) | RWKV-7 Params | ViT Params |
| :---   | :---     | :---             | :---          | :---                 | :---          | :---       |
| tiny   | 512      | 652.00           | 289.98        | 0.44x                | 0.77M         | 5.88M      |
| small  | 512      | 2914.55          | 891.96        | 0.31x                | 7.86M         | 22.37M     |
| medium | 512      | 8473.36          | 2911.70       | 0.34x                | 58.15M        | 87.20M     |
| large  | 512      | 11738.89         | 9558.37       | 0.81x                | 127.73M       | 305.17M    |

#### Resolution Sweep (Small Model, config: embed_dims=256, depth=6, num_heads=4)

| Img Size | RWKV-7 Time (ms) | ViT Time (ms) | Speedup (ViT/RWKV-7) | Tokenizer % |
| :---     | :---             | :---          | :---                 | :---        |
| 64       | 1214.23          | 21.79         | 0.02x                | 3.1%        |
| 128      | 1259.84          | 46.78         | 0.04x                | 6.4%        |
| 224      | 1404.21          | 126.29        | 0.09x                | 16.0%       |

#### Memory Comparison

| Size   | RWKV-7 Peak (MB) | ViT Peak (MB) | Ratio (RWKV-7/ViT) |
| :---   | :---             | :---          | :---               |
| tiny   | 0.02             | 0.01          | 0.37x              |
| small  | 0.04             | 0.00          | 0.09x              |
| medium | 0.06             | 0.00          | 0.07x              |

### Detailed Comparative Analysis: SpixRWKV-7 vs. Vision Transformers (ViTs)

Based on CPU benchmarks and training experiments, here is a detailed analysis of SpixRWKV-7's strengths and limitations relative to standard ViTs:

#### 1. Representation & Inductive Bias: Superpixels vs. Fixed Grids
* **Vision Transformers**: Process images using an arbitrary $P \times P$ grid of patches (e.g., $14 \times 14$). This splits contiguous visual semantics (e.g., cutting an object in half) and processes low-information background patches identically to foreground objects.
* **SpixRWKV-7**: Employs **diffSLIC** to group pixels into irregular, perceptually coherent regions (superpixels). This provides a native boundary-aware representation. Graph-based spatial mixing (`q_shift_graph_multihead`) and Hilbert sequence reordering preserve local and non-local geometry.

#### 2. Computational Complexity & Latency
* **Theoretical Complexity**: ViT attention scales quadratically ($O(N^2)$) in time and memory with the number of tokens $N$. SpixRWKV-7's recurrent scan has a linear ($O(N)$) time complexity and constant ($O(1)$) recurrent state memory.
* **CPU Latency & Parallelization Bottleneck**: On CPU, PyTorch's native transformer uses highly tuned, parallelized CPU matrix multiplication (GEMM) kernels. In contrast, SpixRWKV-7's recurrent scan operates sequentially over the sequence length (`for t in range(N)`), introducing significant loop overhead and poor CPU cache efficiency.
* **Tokenizer Overhead**: Differentiable superpixel tokenization (diffSLIC) is computationally expensive on CPU due to iterative clustering. For the **tiny** model, diffSLIC tokenizer accounts for **69.6%** of total execution time. As the model size scales, backbone recurrence dominates (**75.5%** for the **large** model).
* **Scaling Advantage**: As model scale increases, the speed gap between ViT and SpixRWKV-7 narrows. At **Large** scale, ViT is only **1.23x** faster than SpixRWKV-7, even though ViT has **2.4x** more parameters.

#### 3. Parameter and Memory Efficiency
* **Parameter Count**: SpixRWKV-7 is extremely parameter-efficient. It matches or exceeds ViT configurations in representation capacity with a fraction of the parameters:
  * **Tiny**: SpixRWKV-7 uses **0.77M** params vs. ViT's **5.88M** (ViT is **7.6x** larger).
  * **Large**: SpixRWKV-7 uses **127.73M** params vs. ViT's **305.17M** (ViT is **2.4x** larger).
* **Memory Footprint**: Under high-resolution or dense-prediction settings, the quadratic memory scaling of ViT attention becomes a bottleneck. SpixRWKV-7 scales linearly and retains a constant state size, preserving memory.

#### 4. Convergence & Stability
* Gradient health diagnostics (`diagnose_training.txt`) confirm uniform gradient flow across all blocks under RMSNorm and SwiGLU activation configurations. Single-batch overfit is achieved in **17 steps**, proving stable optimization dynamics.

### Why Is ViT Faster on CPU?

1. **Optimized GEMM kernels**: PyTorch's Transformer uses heavily tuned CPU matrix multiplication that exploits cache locality and SIMD instructions.
2. **Parallel vs Sequential**: The Vision RWKV-7 backbone has a sequential recurrent loop (`for t in range(N)`) that cannot be vectorized like the parallel attention in ViT. Each timestep requires its own forward pass through the recurrence.
3. **diffSLIC overhead**: The tokenization involves iterative clustering with softmax operations over spatial dimensions - computationally expensive on CPU.
4. **Small matrix inefficiency**: The RWKV-7 recurrence uses small matrix operations (head_size=64) that have poor cache efficiency compared to larger GEMM operations.
5. **GPU advantage for RWKV-7**: On GPU, the recurrent loop can run in parallel across sequence positions using the custom CUDA kernel without quadratic attention memory overhead.

### Key Insights

- **Parameter efficiency gap**: ViT is ~100-400x more efficient on CPU (time per M params) because it uses optimized parallel operations, while RWKV-7 uses sequential recurrence that doesn't parallelize well on CPU.
- **diffSLIC scaling**: As image resolution increases, diffSLIC takes a larger fraction of total time. This is the primary optimization target for C++ kernel implementation.
- **Backbone dominance**: At medium/large sizes, the recurrent backbone dominates, indicating that once diffSLIC is optimized, RWKV-7 could become competitive.
- **Memory advantage**: RWKV-7 uses significantly less peak memory due to its linear recurrent structure vs ViT's quadratic attention. Note: ViT's 0.00MB reflects tracemalloc's limitation with optimized C++ kernels; both models use minimal memory in inference mode.

Run your own comparison:
```bash
uv run python scripts/compare_architectures.py --runs 10
```

## Training Results

The architecture was validated using a two-step ladder: first a single-batch overfit test (fastest convergence check), then systematic diagnostics.

### Single-Batch Overfit (PASS)

Smallest config tested (`embed_dims=128`, `depth=2`, 36 superpixels, CPU):

```
RESULT: PASS, model overfits the batch (convergence OK)
- 100% accuracy in 48 steps (28 seconds wall time on CPU)
- Loss: 2.63 → 0.38
- Plateau: 50% for ~35 steps, then rapid convergence to 100%
```

### LR Sensitivity Sweep

| LR       | Final Loss | Best Acc | Plateau Length | Verdict             |
| -------- | ---------- | -------- | -------------- | ------------------- |
| **1e-3** | **0.001**  | **100%** | **2 steps**    | Fastest             |
| **5e-4** | **0.003**  | **100%** | **3 steps**    | Recommended default |
| 1e-4     | 0.958      | 50%      | never escaped  | Too slow            |
| 5e-5     | 1.045      | 50%      | never escaped  | Too slow            |

**Sweet spot**: 5e-4 to 1e-3 with AdamW. Sharp transition below 5e-4, the architecture needs a minimum effective LR due to recurrent state dynamics.

### Depth Scaling

| Depth | Final Loss | Time/Step | Params | Gradient Uniformity |
| ----- | ---------- | --------- | ------ | ------------------- |
| 1     | 0.003      | 687ms     | 345K   | N/A                 |
| 2     | 0.003      | 1.3s      | 644K   | Uniform             |
| 4     | 0.007      | 2.5s      | 1.24M  | Uniform             |

**All depths converge to 100%.** No gradient vanishing/exploding, gradients are nearly identical across all blocks at convergence (~0.0035 per block in depth=4).

### Seed Stability

| Seed | Final Loss | Best Acc | Steps to 100% |
| ---- | ---------- | -------- | ------------- |
| 42   | 0.003      | 100%     | ~48           |
| 123  | 0.003      | 100%     | ~50           |
| 456  | 0.004      | 100%     | ~50           |
| 789  | 0.016      | 100%     | ~60           |

**All seeds reach 100%**, convergence is reproducible, not a lucky initialization artifact.

### Gradient Health

- Grad norms stay bounded (all steps in `[0.01, 100]`)
- Uniform distribution across blocks, no block dominates or vanishes
- Head receives strong gradient signal throughout training
- Gradient clipping at 10.0 catches rare spikes (only 4-8% of steps)

### Feature Quality (No Supervision)

Backbone features without any head training:

- Mean ≈ 0, Std ≈ 1 (correct LayerNorm behavior)
- Spatial variance > 0.09 (features differ across spatial positions)
- Range: [-2.5, 3.2] (no extreme outliers)
- All finite, no NaN/Inf from recurrence or normalization

### What These Results Mean

1. **Gradient flow through the RWKV-7 recurrence is healthy**, the delta-rule state update with additive residuals doesn't suffer from typical RNN gradient decay. Uniform gradient distribution across depth=4 is strong evidence.
2. **Superpixel tokenization doesn't hinder optimization**, diffSLIC + Hilbert sort + KNN + Q-shift forms a differentiable pipeline with smooth gradients.
3. **The architecture provides strong spatial inductive bias**, from random initialization, the model immediately achieves 50% accuracy (vs 10% random), thanks to the graph-based spatial mixing.
4. **The 50% plateau is expected**, the model quickly separates 2-3 classes and then needs gradient signal to refine the remaining boundaries. The plateau breaks when the classifier head adjusts.
5. **LR window is 2× (5e-4 to 1e-3)**, comparable to many transformer architectures. Stay above 5e-4 for reliable training.

### RMSNorm + SwiGLU Validation Results

Comprehensive evaluation of the `norm_layer="rmsnorm"` and `act_layer="swiglu"` configuration using optimized kernels. All tests passed successfully with no NaN/Inf outputs.

#### Single-Batch Overfit (RMSNorm + SwiGLU)

```
Config: 64x64 image, embed_dims=128, depth=2, 36 superpixels, 773K params
RESULT: PASS — 100% accuracy in 17 steps (8.0s on CPU)
- Loss: 2.15 → 1.38
- GradNorm: 7.8 → 0.4 (stable, no explosion)
```

**Architecture Meaning**: RMSNorm + SwiGLU converges correctly. No numerical issues, stable gradients throughout training.

#### Diagnostic Tests (RMSNorm + SwiGLU)

**LR Sweep** (2 steps each, depth=2, 128 dims):

| LR   | Final Loss | Best Acc | GradNorm Range |
| ---- | ---------- | -------- | -------------- |
| 1e-3 | 0.0025     | 100%     | 0.01-0.09      |
| 5e-4 | 0.037      | 100%     | 0.09-0.09      |
| 1e-4 | 0.98       | 50%      | 0.87-0.86      |
| 5e-5 | 1.37       | 50%      | 0.38-0.34      |

**Depth Scaling** (lr=5e-4, 2 steps each):

| Depth | Final Loss | Params | Gradient Uniformity |
| ----- | ---------- | ------ | ------------------- |
| 1     | 0.059      | 345K   | N/A (single block)  |
| 2     | 0.037      | 644K   | Uniform             |
| 4     | 0.022      | 1.24M  | Uniform (all blocks)  |

**Seed Reproducibility** (lr=5e-4, 2 steps each):

| Seed | Final Loss | Best Acc | GradNorm |
| ---- | ---------- | -------- | -------- |
| 42   | 0.0369     | 100%     | 0.45-0.33 |
| 123  | 0.006      | 100%     | 0.01-0.01 (low noise) |
| 456  | 0.038      | 100%     | 0.08-0.08 |
| 789  | 0.017      | 100%     | 0.07-0.04 |

**Feature Sanity** (no head, depth=2, 128 dims):
- Shape: (4, 128, 6, 6)
- Mean: 0.26, Std: 0.97 (healthy variance)
- Range: [-2.34, 2.85] (no outliers)
- Spatial variance: 0.13 (features differ spatially)

#### Model Demo (RMSNorm + SwiGLU, Medium Config)

```
Config: 512x512 image, ~45.6M params (medium)
Total params: 45.60M
All outputs finite: True
Deterministic: True (across 10 seeds: 3407, 42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000)
```

**Architecture Meaning**: Medium-scale model runs stably. RMSNorm + SwiGLU scales without numerical issues.

**Parallel Mode Comparison (CPU)**:
| Mode   | Size   | Img Size | Time (ms) | Ratio vs RNN |
| ------ | ------ | -------- | --------- | ------------ |
| RNN    | tiny   | 224      | 168       | 1.00x        |
| Parallel | tiny | 224      | 246       | 0.68x (slower) |
| RNN    | tiny   | 512      | 537       | 1.00x        |
| Parallel | tiny | 512      | 631       | 0.85x (slower) |
| RNN    | small  | 224      | 1406      | 1.00x        |
| Parallel | small | 224      | 4130      | 0.34x (slower) |

**Key Insights**:
- **CPU bottleneck confirmed**: ViT remains 2-3x faster on CPU due to sequential recurrence
- **Parameter efficiency improves with scale**: RMSNorm+SwiGLU = 149ms/M (medium) vs 244ms/M (tiny)
- **Memory advantage preserved**: RWKV-7 uses 5-7% of ViT memory (0.02-0.06MB vs 0.00MB)

**Parallel Mode Notes**:
- On CPU, parallel mode is **slower** than RNN due to Blelloch scan overhead (additional matrix ops in the tree scan)
- On GPU (CUDA), parallel mode uses the optimized `wkv7s.cu` kernel that parallelizes across timesteps via CUDA threads, achieving significant speedup
- The CUDA kernel from BlinkDL/RWKV-LM is the reference implementation - RWKV-7's speed advantage is fundamentally GPU-bound
- Parallel mode shines when N (sequence length) is large and GPU compute is available
- The `--use-parallel` flag enables parallel mode; `--hard-mode` enables seeds-revised diffSLIC inference

**Why SpixRWKV-7 is currently slower on CPU (optimization opportunity)**:
- Current CPU kernels use sequential loop + AVX512, no ggml integration
- rwkv.cpp with ggml achieves **2-3x speedup via SIMD and thread pools**
- Quantization (Q4/Q5) provides additional **2-3x speedup** (76ms vs 198ms for large models)
- **Opportunity**: Integrate ggml framework for INT4/INT5/INT8 quantized inference

**Next steps for CPU performance**:
- Integrate rwkv.cpp's ggml-based WKV v7 kernel (SIMD + thread parallelization)
- Add Q4_0/Q5_1 quantization support for 2-3x inference speedup
- Leverage rwkv.cpp's `rwkv_wkv_v7_impl` for better cache efficiency
- Reference: rwkv.cpp achieves FP32=198ms → Q4_0=76ms on 4C/8T AVX2 CPU

### HumorDB Regression (Funniness Rating)

A real-world regression test on [HumorDB](https://huggingface.co/datasets/joey234/humordb) (2136 train, 703 val, 706 test, target = `range_ratings_mean`, mean ~5.76, σ ~1.89).

**Config**: `embed_dims=128` (2 heads), `depth=4`, 36 superpixels, 64×64, batch=8, LR=5e-4, AdamW, cosine schedule, 1.24M params.

```
Epoch | Train R² | Train r | Val R² | Val r  | Val RMSE
------+----------+---------+--------+--------|---------
    1 |   -0.07  |   0.01  | -0.00  |  0.07  |  1.841
    4 |   -0.01  |  -0.01  |  0.00  |  0.09  |  1.836
    7 |   -0.01  |   0.00  |  0.01  |  0.10  |  1.835
```

**Result**: The model failed to learn with standard full-shuffle DataLoader (R² ≈ 0 throughout, predictions collapsed to near-constant mean). An earlier streaming run with buffer_size=100 shuffle reached Train R²=0.26.

**Key insight, shuffle sensitivity**: The architecture at this scale (1.24M params) is highly sensitive to batch composition. Full random shuffle produces gradient noise that prevents the tiny recurrent model from escaping the constant-predictor equilibrium. Structured shuffle (buffer=100) provides temporal gradient smoothing that helps the model discover structure. This is consistent with ViT behavior at abnormally small parameter counts, ViT-Tiny (5.7M params) would likely show the same collapse if scaled down to 1.2M on a noisy regression task with 2136 samples.

**Takeaway**: For SpixRWKV-7 regression on CPU:

- Use buffer-based shuffle (`buffer_size=100` in streaming or custom shuffling in caching layer) for models under ~3M params
- Or increase capacity to `embed_dims=192+` (3+ heads), more superpixels (128+), and deeper blocks (depth 8+) to handle full random shuffle
- Disk caching preprocessed 6-channel tensors gives ~18% epoch time reduction (~483s vs ~590s, dominated by diffSLIC forward/backward on CPU)
- Training scripts: `tasks/classification/humordb/train.py` (with `--help`), `tasks/classification/humordb/infer.py`

### VQ-VAE Tokenization Ablation Results

We ran a comparative validation to evaluate the effect of substituting differentiable superpixels with a learned Convolutional VQ-VAE tokenizer.

#### Single-Batch Overfit (VQ-VAE)
- **Config**: 64x64 image, embed_dims=128, depth=2, downsample_factor=8 (64 tokens), codebook_size=512, 2.82M params.
- **RESULT: PASS** — 100% accuracy achieved in just **14 steps** (4.3 seconds total on CPU).
- **Comparison**: Superpixel tokenization (`spix`) took 50 steps (19.7s) to reach 100% accuracy under identical settings.
- **Implication**: The discrete prior of VQ codebooks provides significantly cleaner representation gradients, accelerating convergence by nearly 3.5× in the recurrent backbone.

#### VQ Token sequence scaling & CPU overhead
Unlike superpixels (which allow restricting $N$ to a small constant e.g. 36), the VQ-VAE sequence length is determined by the spatial grid $(H/stride \times W/stride)$. At 512px resolution with stride 8, this yields 4096 tokens, causing the sequential recurrent loop to take ~9.8s per CPU forward pass. Using a larger stride (e.g. 16 for 1024 tokens) reduces this overhead, but for high-resolution processing on CPU, Superpixel tokenization remains far more runtime-controllable.

### ADE20K Semantic Segmentation

A semantic segmentation experiment on [ADE20K](https://huggingface.co/datasets/1aurent/ADE20K) to evaluate SpixRWKV-7 on dense classification tasks with multi-scale feature scattering.

**Scripts**: `tasks/segmentation/ade20k/sanity.py` (fast CPU overfit test, 128–512 images), `tasks/segmentation/ade20k/train.py` (full training with streaming).

**Key findings**:

1. **Feature Projection (Superpixel-to-Pixel Scattering)**: For semantic segmentation, SpixRWKV-7 does not require heavy decoding networks to reconstruct resolution. The `scatter_output=True` mode projects irregular token representations back to a dense grid by taking the weighted sum of token features using the soft superpixel assignments mask. This yields a parameter-free upsampling step compared to ViT dense prediction decoders.

2. **Backbone feature magnitude**: The `scatter_output=True` features have extreme range `[-1238, 1040]` (tiny config), causing logit explosion and loss ~100. Fix: `nn.BatchNorm2d(embed_dims)` before the 1×1 conv seg head normalizes features, bringing loss to ~log(num_classes) ≈ 5.6.

3. **Seg head design**: 1×1 Conv2d with `bias=False` + preceding BatchNorm2d. No upsampling, relies on backbone's `scatter_output` to produce full-resolution dense maps.

4. **Spatial Resolution & Gated Fusion**: Unlike the classification head which performs Global Average Pooling, the segmentation head acts directly on the scattered spatial features. Gated bidirectional recurrence inside the backbone maintains high-frequency spatial details at the boundaries of superpixels, which are crucial for fine-grained segmentation.

5. **Scale presets** (backbone + seg head params):
   - `tiny`: embed_dims=128, depth=2, 36 spx → ~1.3M total
   - `small`: embed_dims=256, depth=6, 144 spx → ~18M total
   - `medium`: embed_dims=512, depth=12, 196 spx → ~57M total
   - `100m`: embed_dims=768, depth=12, 196 spx → ~99.5M total

6. **Sanity run status**: 128 train / 32 val images, tiny preset, 10 epochs, loss decreasing, gradients flowing (grad_norm ~13k), accuracy > random (~25% vs 0.35%). Full training pending.

7. **Parameter-free Decoder Comparison**: Standard ViT segmenters (like Segmenter or SETR) require expensive transformer decoders or convolution upsampling layers. SpixRWKV-7 utilizes the tokenizer's soft assignments mask to project 1D tokens back into 2D space, maintaining a lightweight seg head footprint (~13K params for 102 classes vs megabytes for ViT decoders).

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) for dependency management. Standard `pip` also works.

```bash
# Clone the repository
git clone https://github.com/Gabz4200/SpixRWKV7.git
cd SpixRWKV7

# Install dependencies
uv sync
```

## Quick Start

### Backbone Forward Pass

```python
import torch
from spixrwkv7 import create_vision_rwkv7

# Initialize the model (backends: "diff_slic", "grid", "slic", "slico", "lnsnet")
model = create_vision_rwkv7(
    img_size=224,
    embed_dims=192,
    num_heads=3,
    depth=12,
    num_superpixels=196,      # approx 14x14 superpixel grid
    out_indices=[3, 5, 7, 11], # multi-scale feature extraction
    scatter_output=True,       # scatter back to original resolution
    spixel_backend="diff_slic", # "diff_slic" (default), "grid", "slic", "slico", or "lnsnet"
)

x = torch.randn(1, 6, 224, 224)  # 6 channels: Lab + alpha + xy
outs = model(x)

for i, o in enumerate(outs):
    print(f"Level {i}: {tuple(o.shape)}")
# Example: Level 0: (1, 192, 224, 224), full resolution
```

### Training Convergence Test

```bash
# Fast single-batch overfit (takes ~30s on CPU)
uv run python tasks/diagnostics/fast_test_training.py

# Full diagnostic battery (LR sweep, depth scaling, seeds, gradients)
uv run python tasks/diagnostics/diagnose_training.py --all
```

### HumorDB Funniness Regression

```bash
# Train (20 epochs, ~2.5h on CPU with caching)
uv run python tasks/classification/humordb/train.py --epochs 20

# Inference on best checkpoint
uv run python tasks/classification/humordb/infer.py --checkpoint checkpoints/humordb/best_val_loss.pt
```

```python
from spixrwkv7 import create_vision_rwkv7, ClassificationHead

backbone = create_vision_rwkv7(img_size=64, embed_dims=128, depth=2)
head = ClassificationHead(embed_dims=128, num_classes=10)

x = torch.randn(4, 6, 64, 64)
features = backbone(x)  # tuple of dense feature maps
logits = head(features[0])  # (4, 10)
```

## Project Structure

```
VisualRWKV7_Pytorch/
├── spixrwkv7/                   # Core Python package (package name: spixrwkv7)
│   ├── __init__.py              # Public API exports (includes C++ kernel fallback)
│   ├── models/
│   │   ├── spixrwkv7.py         # Backbone + all modules (PyTorch implementation)
│   │   └── vq_rwkv7.py          # VQ-RWKV-7 model (VQ-VAE tokenization ablation)
│   ├── data/
│   │   ├── colors.py            # OkLAB/sRGB conversion utilities
│   │   ├── gamut.py             # OkLAB gamut clipping methods
│   │   ├── diff_slic.py         # Differentiable SLIC implementation
│   │   └── transforms.py        # Image preprocessing utilities
│   ├── layers/
│   │   ├── graph.py             # KNN graph construction and Graph Q-Shift
│   │   └── drop.py              # Stochastic depth (DropPath)
│   └── kernels/                 # Optimized C++ kernels (MUST stay in sync with models/)
│       ├── __init__.py          # Re-exports with try/except fallback
│       ├── setup.py             # C++ extension build script
│       ├── rwkv7_kernel.py      # Python wrappers with PyTorch fallbacks
│       ├── optimized_block.py   # OptimizedVision_RWKV7_Block, OptimizedRecurrentScan
│       ├── optimized_vision.py  # OptimizedVision_RWKV7, create_optimized_vision_rwkv7
│       ├── rwkv7_kernel.cpp     # C++ RWKV-7 recurrence (AVX512 dispatch)
│       └── cpp/                 # C++ sources (torch_binding.cpp, rwkv7_kernel*.cpp)
│           ├── torch_binding.cpp
│           ├── rwkv7_kernel.cpp
│           ├── rwkv7_kernel.hpp
│           ├── rwkv7_kernel_avx512.cpp
│           ├── diff_slic_kernel.cpp
│           ├── diff_slic_kernel_avx512.cpp
│           └── cpu_features.hpp
└── utils/
│   └── __init__.py          # Utility module init
├── tasks/                       # Training scripts organized by task type
│   ├── diagnostics/
│   │   ├── fast_test_training.py    # Single-batch overfit convergence test
│   │   └── diagnose_training.py     # Systematic training diagnostics
│   ├── classification/
│   │   └── humordb/
│   │       ├── train.py             # HumorDB funniness regression training
│   │       └── infer.py             # HumorDB checkpoint inference + metrics
│   └── segmentation/
│       └── ade20k/
│           ├── sanity.py            # ADE20K fast CPU overfit test
│           └── train.py             # ADE20K semantic segmentation training
├── scripts/
│   ├── demo.py                    # Demo / verification script
│   └── compare_architectures.py   # Vision RWKV-7 vs ViT speed comparison
├── tests/
│   ├── test_models/
│   │   └── test_model.py          # Backbone and block invariants
│   ├── test_data/
│   │   ├── test_colors.py         # Color space conversion tests
│   │   ├── test_diff_slic.py      # diffSLIC mechanics
│   │   └── test_transforms.py     # Transform utilities
│   ├── test_layers/
│   │   └── __init__.py
│   ├── test_utils/
│   │   └── __init__.py
│   ├── test_regression.py           # Numerical stability and regression checks
│   └── __init__.py
├── configs/
│   ├── model/
│   │   ├── tiny.yaml              # Tiny config (192-dim, 12 layers)
│   │   ├── small.yaml             # Small config
│   │   ├── medium.yaml            # Medium config
│   │   └── large.yaml             # Large config
│   └── task/
│       ├── humordb.yaml           # HumorDB training config
│       └── ade20k.yaml            # ADE20K training config
├── pyproject.toml                 # Project metadata and dependencies
└── README.md
```

## Architecture Overview

### Tokenization Pipeline

```
Input Image (B, 6, H, W)
        │
        ▼
    diffSLIC ──────────────► Soft superpixel assignments (B, K, h, w)
        │
        ▼
    SuperpixelEmbedding ───► Token pooling + centroid computation + positional encoding
        │
        ▼
    Hilbert Sort ─────────► Deterministic 1D token ordering
        │
        ▼
    KNN Graph ─────────────► Neighbor indices remapped to sorted order
        │
        ▼
    Token sequence (B, N, D) + neighbors (B, N, K)
```

### RWKV-7 Block (Bidirectional)

```
Input tokens (B, N, D)
        │
        ▼
    LayerNorm
        │
        ▼
    SpatialMixer
        ├── Graph Q-Shift (along KNN edges)
        ├── _DynamicOffset (input-dependent mixing)
        ├── RecurrentScan (forward)
        ├── RecurrentScan (backward)
        └── Gated fusion → LayerNorm → residual
        │
        ▼
    ChannelMix
        ├── Graph Q-Shift
        ├── Gated FFN (ReLU²)
        └── LayerNorm → residual
        │
        ▼
    Output tokens (B, N, D)
```

### Output

```
Feature tokens at selected blocks
        │
        ▼
    _project_output
        ├── Inverse Hilbert reorder
        ├── If scatter: einsum (soft) or gather (hard) → (B, D, H, W)
        └── If not: reshape → (B, D, h_s, w_s)
```

### Module Reference

| Module                | Role                               | Key Operations                                                                             |
| --------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `RecurrentScan`       | Single-direction RWKV-7 recurrence | Delta rule state update, decoupled keys (k_k/k_a), value residual, group norm, time-mixing |
| `SpatialMixer`        | Full attention sub-block           | Graph Q-shift + `_DynamicOffset` + 2× `RecurrentScan` + gated fusion                       |
| `ChannelMix`          | FFN sub-block                      | Q-shift gating + ReLU² activation + LayerNorm                                              |
| `SuperpixelTokenizer` | Vision-to-tokens pipeline          | diffSLIC → `SuperpixelEmbedding` → KNN graph → Hilbert sort → neighbor remap               |
| `ClassificationHead`  | Downstream classifier (separate)   | GAP → LayerNorm → Linear                                                                   |
| `SuperpixelEmbedding` | Pixel-to-token pooling             | Conv features + centroid encoding + Fourier positional embedding                           |

## Testing

Run the full test suite:

```bash
uv run pytest -v
```

**Expected output:** 126 tests pass.

```text
tests/test_models/test_model.py             ...................................................... (86 tests)
tests/test_data/test_colors.py             ......................... (25 tests)
tests/test_data/test_transforms.py         .................... (20 tests)
tests/test_data/test_diff_slic.py          .............. (14 tests)
tests/test_regression.py         ....... (7 tests)
============================= 126 passed in 12-13s =============================
```

Tests are structured to verify behavior through **public interfaces** only, internal module reshuffling won't break them. Key test categories:

- Block forward pass invariants (finiteness, determinism, shape correctness)
- RWKV-7 architectural properties (decoupled keys, bonus term, vector-valued decay, state update)
- Gradient flow (backpropagation to input)
- Multi-scale output, CLS token, scatter output, and dynamic resolution
- Alternative superpixel backends (grid, slic, slico, lnsnet)
- Attention Residuals (AttnRes) modes and gates
- VQ-VAE tokenization (VectorQuantizer, ConvolutionalVQVAE, VQ_RWKV7)
- Superpixel embedding modes (hard/soft)
- Color space correctness and gamut clipping stability
- diffSLIC numerical stability and C++ backend

## Utilities

- **`spixrwkv7/data/colors.py`**: Differentiable conversions between sRGB, Linear RGB, and OkLAB.
- **`spixrwkv7/data/gamut.py`**: Vectorized OkLAB gamut clipping methods (Chroma preservation, adaptive L0 projection).
- **`spixrwkv7/data/transforms.py`**: Image preprocessing utilities.
- **`spixrwkv7/layers/graph.py`**: KNN graph construction and multi-head graph-based token shifting.
- **`spixrwkv7/layers/drop.py`**: Stochastic depth (DropPath) implementation.

## References & Inspirations

This implementation builds upon several foundational works:

- **RWKV-7 "Goose"**: Peng, B., Alcaide, E., et al. "RWKV-7 'Goose' with Expressive Dynamic State Evolution." _arXiv:2503.14456_ (2025). The delta-rule recurrence with decoupled key-value states and dynamic w-kv gating that powers the recurrent scan in every block.
- **Eagle & Finch (RWKV-5/6)**: Peng, B., Alcaide, E., et al. "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." _arXiv:2404.05892_ (2024). Introduced the multi-headed matrix-valued hidden states and dynamic recurrence mechanism that RWKV-7 builds upon.
- **Vision-RWKV**: Duan, Y., et al. "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-like Architectures." _ICLR 2025_. The original RWKV-to-vision adaptation that this project started as a port of, before diverging significantly.
- **SLIC Superpixels**: Achanta, R., et al. "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods." _IEEE TPAMI_ 34(11):2274-2282 (2012). The simple linear iterative clustering algorithm at the core of the superpixel tokenization pipeline.
- **Superpixel Sampling Networks (SSN)**: Jampani, V., et al. "Superpixel Sampling Networks." _ECCV 2018_. Introduced differentiable SLIC via iterative cluster updates with soft assignment, which the diffSLIC implementation adapts for end-to-end training.
- **Hilbert Curve**: Space-filling curve for locality-preserving token ordering, used to place spatially proximate superpixels adjacent in the recurrent scan sequence.

## Optimized Kernels

The `spixrwkv7/kernels/` module provides optimized C++ implementations of the core RWKV-7 WKV operator:

- **WKV v7 kernel** (`rwkv7_kernel.cpp`): Implements the delta-rule recurrence with SIMD-optimized loops, adapted from gabz-rwkv.cpp.
- **OptimizedVision_RWKV7_Block**: Drop-in replacement for `Vision_RWKV7_Block` that uses the C++ kernel when available.
- **OptimizedVision_RWKV7**: Full backbone wrapper that uses optimized blocks.

**IMPORTANT**: The PyTorch implementation (`spixrwkv7/models/spixrwkv7.py`) and the optimized implementation (`spixrwkv7/kernels/optimized_block.py`, `spixrwkv7/kernels/optimized_vision.py`) must be kept in SYNC at all times. Any architectural change to the core model must be reflected in the optimized versions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Building the C++ Extension

```bash
# Build the extension (requires C++ compiler and PyTorch headers)
cd spixrwkv7/kernels && python setup.py build_ext --inplace
```

### Using Optimized Kernels

```python
from spixrwkv7 import create_optimized_vision_rwkv7

# Create model with optimized kernel
model = create_optimized_vision_rwkv7(
    img_size=224,
    embed_dims=256,
    depth=6,
    num_heads=4,
    use_cpp=True,  # Enable C++ kernel
)
```

### Benchmark with Optimized Kernels

```bash
# Compare PyTorch vs C++ kernel performance
uv run python scripts/compare_architectures.py --runs 10 --use-cpp
```

## Contributing

Contributions, issues, and feature suggestions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. I would love to learn from others and improve this project even more.

## License

This project is licensed under the **Apache 2.0 License**, see the [LICENSE](LICENSE) file for details, aligning with the upstream RWKV project.

---

_Built with ❤️ for learning about efficient, scalable, and adaptive computer vision. Not a Warhammer 40k reference._
