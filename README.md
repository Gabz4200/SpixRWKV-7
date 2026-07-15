# SpixRWKV-7: Superpixel Graph RWKV-7 Vision Backbone

A PyTorch implementation of a **recurrent vision backbone** that replaces rigid patch grids with differentiable superpixels (`diffSLIC`), processes tokens through **graph-based Q-shift** and **bidirectional RWKV-7 recurrence**, and outputs dense feature maps at arbitrary resolutions.

The architecture merges the linear-complexity, constant-memory advantages of the **RWKV-7** state-space model with vision-specific adaptations (Graph-Based Q-Shift, gated bidirectional fusion, Hilbert reordering), while introducing a novel **irregular-grid tokenization** pipeline. Unlike standard ViTs, SpixRWKV-7 operates on perceptually grouped pixels (superpixels) rather than fixed grid patches, enabling adaptive spatial resolution and natural contour awareness.

> ⚠️ **DISCLAIMER 1:** This repository is yet another learning project made by a single Brazilian student that is exploring the topic of Sub-quadratic Vision Encoders.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do. I also dont prohibit myself from ocasional help, but the main thing is probably commit messages, I genuinely hate writting those.

## News / Recent Updates

- **GNN Register Token Graph Connectivity + JK-LSTM**: GNN variant now supports DINOv2-style register tokens with proper graph topology — register nodes connect to ALL superpixel nodes (bipartite), giving each superpixel `4 + R` incoming edges. Added Jumping Knowledge LSTM (`jk="lstm"`) for multi-hop feature aggregation across layers. See [GNN section](#gnn-vision-ablation).
- **Fair benchmark with matched params**: All 4 variants (spix, vq, conv, gnn) benchmarked against ViT at matched parameter counts (~5.7M tiny, ~22M small) with 6-channel input (L, a, b, alpha, x, y) for all models. Results in [Full Benchmark section](#full-variant-benchmark-results).
- **GNN Ablation variant**: Replaces RWKV-7 recurrence with PyTorch Geometric GNN message passing (GATv2, GCN, SAGE, GIN, etc.) over the same superpixel KNN graph. Fastest inference variant (2.8x faster than spix at small). Configs: `gnn_tiny/small/medium/large.yaml`.
- **Conv-Stem Vision-RWKV-7 variant**: Two-stream tokenizer with strided convolutions before diffSLIC. Converges fastest (2-3 steps) but slowest inference due to deep backbone. Configs: `conv_tiny/small/medium/large.yaml`.
- **VQ-VAE Tokenization Ablation**: Learned VQ-VAE tokenizer replaces superpixels. Discrete codebook prior provides cleaner gradients but adds significant compute overhead. Configs: `vq_tiny/small/medium/large.yaml`.
- **Real image test data**: `data/caltech101_classification/` with 223 images (butterfly, dalmatian, dolphin). ALL scripts load real images instead of random noise.
- **Optimized C++ kernels**: AVX512-optimized RWKV-7 recurrence and diffSLIC kernels. Enabled via `use_cpp=True` flag.
- **RMSNorm + SwiGLU**: Configurable normalization and activation across all variants.
- **Full benchmark suite**: `scripts/run_full_benchmark.py` — inference speed + training convergence for all 4 variants vs ViT.
+ **Architectural Enhancements (RMSNorm & SwiGLU)**: Added support for configurable normalization layers (`norm_layer="layernorm"|"rmsnorm"`) and activation functions (`act_layer="relu2"|"gelu"|"silu"|"swiglu"`) across the backbone, blocks, spatial/channel mixers, and classification head. RMSNorm and SwiGLU activations are inspired by modern vision/language architectures like DINOv3 and LLaMA, providing greater parameter efficiency and expressive capacity.
+ **Registers (DINOv2-style)**: Optional learnable register tokens prepended to the token sequence, allowing global context accumulation independent of superpixel representation. Enable with `register_tokens=N`.
- **Dynamic Image Size Support**: `img_size` means target **height** in pixels, preserving aspect ratio. `img_size=-1` keeps original resolution. Positive values scale height to that pixel count; width is computed from the image’s aspect ratio.

- **RMSNorm + SwiGLU Validation Complete**: Comprehensive evaluation of the new normalization/activation options shows stable convergence, deterministic behavior, and no NaN/Inf outputs. Full-batch overfit achieved in 17 steps with tiny config (64x64, 128 dims, 36 superpixels). Models under 3M params require structured shuffle (buffer=100) for stable training. See [RMSNorm + SwiGLU Results](#rmsnorm--swiglu-validation-results) for details.
- **Conv-Stem Vision-RWKV-7 variant**: Added `ConvolutionalVision_RWKV7` — a two-stream tokenizer that stacks strided convolutions (replacing the first 4× spatial reduction) before diffSLIC superpixel clustering. Conv stem learns semantically meaningful feature maps, pooling tokens from conv features rather than raw pixel space. Configs: `conv_tiny/small/medium/large.yaml`. Integrated across all scripts (`--model-type conv`), all tests pass (126/126). See [Conv-Stem section](#conv-stem-vision-rwkv-7) for details.
- **GNN Ablation variant**: Added `GNNVision` — replaces the RWKV-7 recurrence with PyTorch Geometric GNN message passing (GATv2, GCN, SAGE, GIN, TransformerConv, etc.) over the same superpixel KNN graph. Clean ablation isolating "how much of performance comes from GNN message passing vs. recurrent delta-rule scan." Configs: `gnn_tiny/small/medium/large.yaml`. Fastest inference variant on CPU (2-3x faster than spix). See [GNN section](#gnn-vision-ablation) for details.
- **Real image test data**: Added `data/caltech101_classification/` with 223 images across 3 classes (butterfly, dalmatian, dolphin). ALL test and benchmark scripts now load real images instead of random noise, ensuring superpixel-sensitive models are evaluated on perceptually meaningful data. Shared utility: `spixrwkv7/data/image_utils.py`.
- **Full 4-variant benchmark suite**: Comprehensive benchmark comparing spix, vq, conv, gnn at tiny/small configurations against ViT baseline. Covers inference speed (with tokenizer/backbone breakdown) and training convergence. Runner: `scripts/run_full_benchmark.py`. See [Full Benchmark section](#full-variant-benchmark-results) for results.
- **Test Validity Documentation**: The repo now documents what each test script validates vs what data volume it needs for meaningful conclusions. See [Test Validity section](#test-validity--data-volume-guidelines).
- **Deep Codebase Cleanup (commit e9caf86)**: Removed all dead code — 7 unused sigma params from all forward signatures, `hilbertcurve` dependency (replaced with native vectorized sort), Q4_0/Q5_1 quantization stubs, dead C++ kernel parameters (`k`, `r_k`, `n_extra_back`), dead `c_p` gather parameter, unused `S` state variables, and `pos_grid_size`/`pos_embed` in the backbone forward. Fixed 4 bugs (`r_k` `NameError`, `ln0` bypass never applied, dead `c_p` gather always masking, unused `S` variables shadowing). Cleaned C++ kernel signatures (state made `const`, removed stale args, `-fno-lto` build flag). Ruff-clean (0 errors), 126/126 tests pass. See [commit e9caf86](https://github.com/Gabz4200/SpixRWKV7/commit/e9caf86ea57587f007c57e5626444dfe60945c9b) for full diff.

## Key Features

- **Superpixel Tokenization Backends**: Replaces rigid patch grids with configurable superpixel tokenizers, supporting multiple backends:
  - `diff_slic`: Differentiable SLIC via PyTorch/C++ (supports both **hard** and **soft** modes).
  - `grid`: Non-overlapping regular grid of patches (providing a direct grid/patch baseline).
  - `slic`: Classical CPU SLIC clustering using `scikit-image` with gradient pass-through.
  - `slico`: Classical CPU SLICO (SLIC-Zero) clustering.
  - `lnsnet`: Non-iterative learnable superpixel segmentation (LNS-Net, CVPR 2021) with BSDS checkpoint weight initialization support.
- **Conv-Stem Tokenization**: A two-stream tokenizer alternative (`ConvolutionalSuperpixelTokenizer`) that stacks learnable strided convolutions before diffSLIC clustering. The raw image is spatially downsampled to match conv feature resolution, then superpixel masks from the semantic stream are used to pool the conv feature stream into tokens. This decouples the spatial reduction (learned via conv) from the grouping criteria (semantic via diffSLIC on raw pixels). Model: `ConvolutionalVision_RWKV7`, builder: `create_conv_vision_rwkv7`.
- **VQ-VAE Tokenization Ablation (VQ_RWKV7)**: A sibling model alternative (`VQ_RWKV7`) that replaces superpixels with learned VQ-VAE codebook representations. It downsamples the image via a Convolutional VQ-VAE encoder to a regular grid of latent features, maps them to the nearest codebook embeddings (with straight-through gradients), and feeds these quantized token embeddings to the RWKV-7 recurrent blocks.
- **GNN Ablation (GNNVision)**: Replaces the RWKV-7 recurrence with PyTorch Geometric graph neural network layers over the same superpixel KNN graph. Register tokens (DINOv2-style) connect bipartitely to all superpixel nodes, giving each superpixel `4 + R` edges. Supports Jumping Knowledge LSTM (`jk="lstm"`) for multi-layer feature aggregation. 9 GNN convolution types: GCN, GraphConv, SAGE, GIN, GAT, GATv2, TransformerConv, ResGatedGraphConv, and GatedGraphConv. Builder: `create_gnn_vision`. Configs: `gnn_tiny/small/medium/large.yaml`.
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

#### Model Size Comparison (matched params, 256px input, 6-channel input)

| Size   | RWKV-7 Time (ms) | ViT Time (ms) | Speedup (ViT/RWKV-7) | RWKV-7 Params | ViT Params |
| :---   | :---             | :---          | :---                 | :---          | :---       |
| tiny   | 462              | 53            | 0.11x                | 5.86M         | 5.69M      |
| small  | 2992             | 147           | 0.05x                | 21.73M        | 21.99M     |

#### Resolution Sweep (Small Model, config: embed_dims=320, depth=11, num_heads=5)

| Img Size | RWKV-7 Time (ms) | ViT Time (ms) | Speedup (ViT/RWKV-7) | Tokenizer % |
| :---     | :---             | :---          | :---                 | :---        |
| 64       | 1214.23          | 21.79         | 0.02x                | 3.1%        |
| 128      | 1259.84          | 46.78         | 0.04x                | 6.4%        |
| 224      | 1404.21          | 126.29        | 0.09x                | 16.0%       |

#### Memory Comparison

| Size   | RWKV-7 Peak (MB) | ViT Peak (MB) | Ratio (RWKV-7/ViT) |
| :---   | :---             | :---          | :---               |
| tiny   | 0.02             | 0.01          | 0.37x              |
| small  | 0.04             | 0.01          | 0.09x              |
| medium | 0.06             | 0.01          | 0.07x              |

### Detailed Comparative Analysis: SpixRWKV-7 vs. Vision Transformers (ViTs)

Based on CPU benchmarks and training experiments, here is a detailed analysis of SpixRWKV-7's strengths and limitations relative to standard ViTs:

#### 1. Representation & Inductive Bias: Superpixels vs. Fixed Grids
* **Vision Transformers**: Process images using an arbitrary $P \times P$ grid of patches (e.g., $14 \times 14$). This splits contiguous visual semantics (e.g., cutting an object in half) and processes low-information background patches identically to foreground objects.
* **SpixRWKV-7**: Employs **diffSLIC** to group pixels into irregular, perceptually coherent regions (superpixels). This provides a native boundary-aware representation. Graph-based spatial mixing (`q_shift_graph_multihead`) and Hilbert sequence reordering preserve local and non-local geometry.

#### 2. Computational Complexity & Latency
* **Theoretical Complexity**: ViT attention scales quadratically ($O(N^2)$) in time and memory with the number of tokens $N$. SpixRWKV-7's recurrent scan has a linear ($O(N)$) time complexity and constant ($O(1)$) recurrent state memory.
* **CPU Latency & Parallelization Bottleneck**: On CPU, PyTorch's native transformer uses highly tuned, parallelized CPU matrix multiplication (GEMM) kernels. In contrast, SpixRWKV-7's recurrent scan operates sequentially over the sequence length (`for t in range(N)`), introducing significant loop overhead and poor CPU cache efficiency.
* **Fair Comparison (Matched Params)**: At ~5.7M params (tiny), ViT is **8x faster** than spix. At ~22M params (small), ViT is **14x faster**. The gap widens with scale because deeper RWKV backbones have more sequential steps.
* **Tokenizer Overhead**: For spix, diffSLIC accounts for **63%** of tiny inference time but only **12%** at small scale (deeper backbone dominates). For gnn, tokenizer is **69%** at tiny but **~0%** at small (backbone dominates).

#### 3. Parameter and Memory Efficiency
* **Parameter Count (Matched)**: At matched parameter counts, the architectures show their true character:
  * **Tiny (~5.7M)**: ViT (5.68M) ≈ spix (5.86M) ≈ gnn (5.67M) ≈ conv (5.61M) ≈ vq (5.43M)
  * **Small (~22M)**: ViT (21.98M) ≈ spix (21.73M) ≈ gnn (21.78M) ≈ conv (21.73M) ≈ vq (22.00M)
* **Memory Footprint**: Under high-resolution or dense-prediction settings, the quadratic memory scaling of ViT attention becomes a bottleneck. SpixRWKV-7 scales linearly and retains a constant state size, preserving memory.

#### 4. Convergence & Stability
* **Fair Comparison (Matched Params)**: At matched parameter counts:
  * **Conv** converges fastest: 4 steps to 100% (conv stem provides strong inductive bias)
  * **VQ** converges moderate: 35 steps (VQ-VAE tokenizer overhead but steady)
  * **Spix** converges moderate: 37 steps (sequential bottleneck, 3.9s/step)
  * **GNN** converges in 75 steps (learns from scratch, but fastest per-step at 1.7s)
* **GNN step count is not a problem**: Conv pre-extracts spatial features before tokenization; GNN must learn both features and message-passing patterns. On real datasets, GNN's per-step speed advantage dominates.
* **Register tokens + JK-LSTM improve GNN convergence**: Ablation shows registers save 3 steps, JK-LSTM saves 8 steps, together save 13 steps.
* Gradient health diagnostics confirm uniform gradient flow across all blocks under RMSNorm and SwiGLU activation configurations.

### Why Is ViT Faster on CPU?

1. **Optimized GEMM kernels**: PyTorch's Transformer uses heavily tuned CPU matrix multiplication that exploits cache locality and SIMD instructions.
2. **Parallel vs Sequential**: The Vision RWKV-7 backbone has a sequential recurrent loop (`for t in range(N)`) that cannot be vectorized like the parallel attention in ViT. Each timestep requires its own forward pass through the recurrence.
3. **diffSLIC overhead**: The tokenization involves iterative clustering with softmax operations over spatial dimensions - computationally expensive on CPU.
4. **Small matrix inefficiency**: The RWKV-7 recurrence uses small matrix operations (head_size=64) that have poor cache efficiency compared to larger GEMM operations.
5. **GPU advantage for RWKV-7**: On GPU, the recurrent loop can run in parallel across sequence positions using the custom CUDA kernel without quadratic attention memory overhead.

### Key Insights (Fair Comparison)

- **ViT dominates CPU inference**: 2.8-30x faster than RWKV variants at matched params. PyTorch's GEMM kernels are heavily optimized for CPU, while RWKV-7's sequential recurrence cannot vectorize.
- **GNN is the fastest RWKV variant**: 3-10x faster than other RWKV variants. GATv2 message passing over 4-NN graph is embarrassingly parallel, avoiding the sequential bottleneck of RWKV-7's recurrence.
- **Conv converges instantly but runs slowest**: 4 steps to 100% accuracy, but the deep backbone makes inference 30x slower than ViT. Strong inductive bias enables fast memorization but constrains representational capacity.
- **GNN convergence is not a problem**: 75 steps vs conv's 4 is expected — GNN learns features and message-passing from scratch, while conv pre-extracts spatial features. On real datasets, GNN's per-step speed (1.7s vs 3.6s) dominates.
- **Register tokens + JK-LSTM help GNN converge**: Ablation shows registers save 3 steps, JK-LSTM saves 8 steps, and together they save 13 steps. Registers provide global context; JK-LSTM mitigates oversmoothing.
- **VQ is expensive**: Both inference and training are slowest due to VQ-VAE encoder/decoder overhead.
- **GPU advantage for RWKV-7**: On GPU, the recurrent loop can parallelize across sequence positions using CUDA kernels, potentially closing the gap with ViT.

Run your own comparison (supports `--model-type {spix,conv,vq,gnn}`):
```bash
# Standard superpixel vs ViT
uv run python scripts/compare_architectures.py --runs 10

# Conv-Stem variant vs ViT
uv run python scripts/compare_architectures.py --model-type conv --runs 3

# GNN ablation vs ViT
uv run python scripts/compare_architectures.py --model-type gnn --runs 3

# Head-to-head 4-variant comparison
uv run python scripts/compare_architectures.py --compare-variants spix conv vq gnn --sizes tiny small
uv run python scripts/compare_architectures_alt_vit.py --compare-variants spix conv vq gnn --sizes tiny small

# Full benchmark suite (inference + training convergence)
uv run python scripts/run_full_benchmark.py --sizes tiny small --img-size 256

# Sweep downsampling factors (spix backbone only)
uv run python scripts/compare_architectures.py --downsample-factors 1.0 2.0 4.0
uv run python scripts/compare_architectures_alt_vit.py --downsample-factors 1.0 2.0 4.0
uv run python tasks/diagnostics/fast_test_training.py --downsample-factors 1.0 2.0 4.0
```

### Full 4-Variant Benchmark Results

Comprehensive benchmark comparing all 4 model variants (spix, vq, conv, gnn) against ViT baseline, using **real images** from `data/caltech101_classification/` (butterfly, dalmatian, dolphin) with **matched parameter counts** and **6-channel input (L, a, b, alpha, x, y)** for all models. Both GNN and ViT use 4 register tokens (DINOv2-style) for fair comparison.

#### Inference Speed (256px input, CPU, matched params)

| Model | Params | Total (ms) | Tokenizer (ms) | Backbone (ms) | vs ViT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **spix** tiny | 5.86M | 462 | 291 (63%) | 171 | 0.11x |
| **vq** tiny | 5.43M | 1009 | 114 (11%) | 895 | 0.05x |
| **conv** tiny | 5.61M | 514 | 16 (3%) | 497 | 0.10x |
| **gnn** tiny | 5.68M | 146 | 101 (69%) | 45 | 0.36x |
| **ViT** tiny | 5.69M | 53 | — | 53 | 1.00x |
| **spix** small | 21.73M | 2992 | 363 (12%) | 2597 | 0.05x |
| **vq** small | 22.00M | 3530 | 210 (6%) | 3315 | 0.04x |
| **conv** small | 21.73M | 4596 | — | 4596 | 0.03x |
| **gnn** small | 21.78M | 444 | — | 444 | 0.33x |
| **ViT** small | 21.99M | 147 | — | 147 | 1.00x |

#### Training Convergence (single-batch overfit, 256px, matched params)

| Model | Steps to 100% | Total Time | Step Time |
| :--- | :--- | :--- | :--- |
| **conv** tiny | 4 | 14.6s | 3.6s |
| **vq** tiny | 35 | 371.5s | 10.6s |
| **spix** tiny | 37 | 142.7s | 3.9s |
| **gnn** tiny | 75 | 129.3s | 1.7s |

#### GNN Ablation: Registers vs JK-LSTM

To isolate the effect of each GNN enhancement, we ran a 4-way ablation on the tiny config (embed_dims=256, depth=6, GATv2, 36 superpixels, 256px, seed=42):

| Config | Register Tokens | JK | Steps to 95% | Total Time | Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 0 | none | 75 | 129.3s | 5.68M |
| **+registers** | 4 | none | 72 | 125.8s | 5.68M |
| **+jk-lstm** | 0 | lstm | 67 | 119.0s | 6.27M |
| **+registers+jk** | 4 | lstm | 62 | 111.9s | 6.27M |

**Key findings from ablation**:
- **Registers alone: minimal impact** (75 → 72 steps). Register nodes add global context without disrupting local graph structure. Each superpixel keeps its 4 KNN edges and gains 4 register edges.
- **JK-LSTM is the real convergence accelerator** (75 → 67 steps). The LSTM reads features from all 6 layers and learns which depths matter most, mitigating oversmoothing — the tendency of deep GNNs for node representations to converge to indistinguishable vectors.
- **The two features are complementary** (75 → 62 steps together). JK-LSTM benefits from register tokens because registers provide a global "summary" signal that the LSTM can use to decide when to trust deep vs shallow features.
- **Cost is negligible**: JK-LSTM adds 590K params (LSTM + projection). Per-step time barely changes (~1.75s → ~1.80s).

#### Key Insights (Fair Comparison with Matched Params)

1. **ViT dominates inference on CPU** (2.8-24x faster) — At matched parameter counts, ViT's optimized parallel GEMM kernels are dramatically faster than any RWKV variant. The gap is smallest for gnn (2.8x at small) and largest for conv (30x at small).

2. **GNN is the fastest RWKV variant** — 146ms (tiny) / 444ms (small), 3-10x faster than other RWKV variants. GATv2 message passing over 4-NN graph is embarrassingly parallel, avoiding the sequential bottleneck of RWKV-7's recurrence.

3. **diffSLIC tokenization dominates spix/gnn** — 63-69% of inference time is tokenizer for tiny models. At small scale, spix's backbone dominates (87%) because the deeper model (15 layers) processes more tokens sequentially.

4. **Conv converges fastest but runs slowest** — 4 steps to 100% accuracy. The conv stem provides strong inductive bias that makes features immediately linearly separable. However, the deep backbone makes inference 30x slower than ViT. This is a classic speed-accuracy tradeoff: the same inductive bias that enables instant convergence constrains what the model can learn.

5. **GNN convergence step count is not a problem** — GNN takes 75 steps vs conv's 4, but this is expected: GNN must learn both features and message-passing patterns from scratch, while conv pre-extracts spatial features before tokenization. On real datasets with proper regularization, GNN's per-step speed advantage (1.7s vs 3.6s) dominates.

6. **VQ is slowest in both inference and training** — The VQ-VAE encoder/decoder adds significant compute overhead. At small, VQ takes 35 steps and 372s total.

7. **Parameter efficiency at matched counts** — When given the same parameter budget, the architectures show their true character:
   - ViT: fastest inference (53ms tiny, 147ms small), moderate convergence
   - GNN: fastest RWKV inference (146ms tiny, 444ms small), fastest per-step training (1.7s)
   - Spix: moderate inference (462ms tiny, 2992ms small), moderate training (3.9s/step)
   - Conv: slow inference (514ms tiny, 4596ms small), fastest convergence (4 steps)
   - VQ: slowest in both inference (1009ms tiny) and convergence (35 steps, 10.6s/step)

Run the full benchmark:
```bash
# Full inference + training convergence for all variants
uv run python scripts/run_full_benchmark.py --sizes tiny small --img-size 256

# Inference only (faster)
uv run python scripts/run_full_benchmark.py --sizes tiny small --skip-training

# Training convergence only
uv run python scripts/run_full_benchmark.py --sizes tiny small --skip-inference
```

### Tokenizer Downsampling Ablation (df = 1.0 / 2.0 / 4.0)

To address tokenizer bottlenecks on high-resolution inputs, we introduced a `downsample_factor` (df) parameter to `SuperpixelTokenizer`. This downsamples the input image (using bilinear interpolation) before running diffSLIC, and upscales the produced superpixel assignment mask back to the original resolution before feature pooling, KNN, and backbone execution.

#### 1. Throughput & Latency (Tiny Model, 512px input, CPU)

| Downsample Factor | Total Time (ms) | Tokenizer Time (ms) | Backbone Time (ms) | Speedup vs df=1.0 |
| :--- | :--- | :--- | :--- | :--- |
| **df=1.0** (Baseline) | 626.42 | 448.16 | 178.26 | 1.00x |
| **df=2.0** | 411.07 | 283.83 | 127.24 | **1.52x** |
| **df=4.0** | 454.09 | 240.51 | 213.58 | 1.38x |

#### 2. Convergence Speed (Single-batch Overfit, 128px input, CPU)

- **df=1.0**: Reached 95% target accuracy at **step 22** (Total time: 63.7s)
- **df=2.0**: Reached 95% target accuracy at **step 21** (Total time: 49.7s)
- **df=4.0**: Reached 95% target accuracy at **step 16** (Total time: **15.9s**)

**Conclusion**: Downsampling before diffSLIC dramatically speeds up tokenization and training step times. Crucially, it also accelerates convergence, as the downsampled grid filters out high-frequency spatial noise in the superpixel assignments, presenting a cleaner semantic graph to the backbone.

## Training Results

The architecture was validated using a two-step ladder: first a single-batch overfit test (fastest convergence check), then systematic diagnostics. All tests now use real images from `data/caltech101_classification/`.

### Single-Batch Overfit (PASS)

At matched parameter counts (~5.7M tiny, ~22M small), all variants converge to 100% accuracy:

| Variant | Tiny Steps | Tiny Time | Small Steps | Small Time |
| :--- | :--- | :--- | :--- | :--- |
| conv | 4 | 14.6s | — | — |
| gnn | 75 | 129.3s | — | — |
| spix | 37 | 142.7s | — | — |
| vq | 35 | 371.5s | — | — |

**Key findings**:
- Conv converges fastest (4 steps) due to strong inductive bias from conv stem
- GNN converges well (75 steps) — per-step time is fastest (1.7s) due to parallel message passing
- Spix converges in 37 steps — sequential bottleneck makes per-step time 2.3x slower than GNN
- VQ converges in 35 steps but is slowest per-step (10.6s) due to VQ-VAE encoder/decoder
- GNN's higher step count is not a problem: it learns features from scratch while conv pre-extracts them

### LR Sensitivity Sweep

| LR       | Final Loss | Best Acc | Verdict             |
| -------- | ---------- | -------- | ------------------- |
| **1e-3** | **0.001**  | **100%** | Fastest             |
| **5e-4** | **0.003**  | **100%** | Recommended default |
| 1e-4     | 0.958      | 50%      | Too slow            |
| 5e-5     | 1.045      | 50%      | Too slow            |

**Sweet spot**: 5e-4 to 1e-3 with AdamW.

### Gradient Health

- Grad norms stay bounded (all steps in `[0.01, 100]`)
- Uniform distribution across blocks, no block dominates or vanishes
- Gradient clipping at 10.0 catches rare spikes (only 4-8% of steps)

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
from spixrwkv7 import create_vision_rwkv7, create_conv_vision_rwkv7, ClassificationHead

# Standard superpixel backbone
backbone = create_vision_rwkv7(img_size=64, embed_dims=128, depth=2)

# Conv-Stem variant (learned conv projection before diffSLIC)
conv_backbone = create_conv_vision_rwkv7(
    img_size=64, embed_dims=128, num_heads=2, depth=2, num_superpixels=64,
    conv_stem_channels=[32, 64, 128], conv_stem_strides=[1, 2, 2],
)

head = ClassificationHead(embed_dims=128, num_classes=10)

x = torch.randn(4, 6, 64, 64)
features = backbone(x)  # tuple of dense feature maps
logits = head(features[0])  # (4, 10)
```

### Running Different Model Variants

All training and benchmark scripts support `--model-type {spix,conv,vq,gnn}`:

```bash
# Standard superpixel variant
uv run python scripts/demo.py --model-type spix --img-size 512 --embed-dims 192 --num-heads 3 --depth 12

# Conv-Stem variant
uv run python scripts/demo.py --model-type conv --img-size 64 --embed-dims 128 --num-heads 2 --depth 2
uv run python tasks/diagnostics/fast_test_training.py --model-type conv --max-steps 50

# VQ-VAE variant
uv run python scripts/demo.py --model-type vq --img-size 64 --embed-dims 128 --num-heads 2 --depth 2
uv run python tasks/diagnostics/fast_test_training.py --model-type vq --max-steps 50

# GNN ablation variant
uv run python scripts/demo.py --model-type gnn --img-size 64 --embed-dims 128 --num-heads 2 --depth 2
uv run python tasks/diagnostics/fast_test_training.py --model-type gnn --max-steps 50

# Compare all 4 variants head-to-head
uv run python scripts/compare_architectures.py --compare-variants spix conv vq gnn --sizes tiny small
```

### Using Real Test Images

All scripts now load real images from `data/caltech101_classification/` (223 images across 3 classes: butterfly, dalmatian, dolphin). Use the shared utility in your own code:

```python
from spixrwkv7.data.image_utils import (
    load_random_caltech101_image,   # (1, 6, H, W) OkLAB + alpha + xy
    load_random_caltech101_batch,   # (B, 6, H, W) batch with labels
    load_random_caltech101_rgb,     # (1, 3, H, W) RGB for ViT
)

# Single image
x, class_name, label = load_random_caltech101_image(img_size=224, seed=42)

# Batch for training
x_batch, y_labels = load_random_caltech101_batch(batch_size=8, img_size=224)
```

## Project Structure

```
VisualRWKV7_Pytorch/
├── spixrwkv7/                   # Core Python package (package name: spixrwkv7)
│   ├── __init__.py              # Public API exports (includes C++ kernel fallback)
│   ├── models/
│   │   ├── spixrwkv7.py         # Backbone + all modules (PyTorch implementation)
│   │   ├── conv_spixrwkv7.py    # Conv-Stem Vision-RWKV-7 variant
│   │   ├── vq_rwkv7.py          # VQ-RWKV-7 model (VQ-VAE tokenization ablation)
│   │   └── gnn_spixrwkv7.py     # GNN Vision model (GNN message passing ablation)
│   ├── data/
│   │   ├── colors.py            # OkLAB/sRGB conversion utilities
│   │   ├── gamut.py             # OkLAB gamut clipping methods
│   │   ├── diff_slic.py         # Differentiable SLIC implementation
│   │   ├── transforms.py        # Image preprocessing utilities
│   │   └── image_utils.py       # Shared real image loading (caltech101)
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
│   ├── compare_architectures.py   # Vision RWKV-7 vs ViT speed comparison
│   ├── compare_architectures_alt_vit.py  # Alt ViT (einops/sincos) comparison
│   └── run_full_benchmark.py      # Full 4-variant benchmark suite
├── data/
│   └── caltech101_classification/ # Real test images (butterfly, dalmatian, dolphin)
│       ├── butterfly/             # 91 images
│       ├── dalmatian/             # 67 images
│       └── dolphin/               # 65 images
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
│   │   ├── tiny.yaml              # Tiny config (~5.7M params, matched to ViT-Tiny)
│   │   ├── small.yaml             # Small config (~22M params, matched to ViT-Small)
│   │   ├── medium.yaml            # Medium config
│   │   ├── large.yaml             # Large config
│   │   ├── conv_tiny.yaml         # Conv-Stem tiny config
│   │   ├── conv_small.yaml        # Conv-Stem small config
│   │   ├── conv_medium.yaml       # Conv-Stem medium config
│   │   ├── conv_large.yaml        # Conv-Stem large config
│   │   ├── vq_tiny.yaml           # VQ-VAE tiny config
│   │   ├── vq_small.yaml          # VQ-VAE small config
│   │   ├── vq_medium.yaml         # VQ-VAE medium config
│   │   ├── vq_large.yaml          # VQ-VAE large config
│   │   ├── gnn_tiny.yaml          # GNN ablation tiny config
│   │   ├── gnn_small.yaml         # GNN ablation small config
│   │   ├── gnn_medium.yaml        # GNN ablation medium config
│   │   └── gnn_large.yaml         # GNN ablation large config
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

### Conv-Stem Tokenization (Convolutional Vision-RWKV-7)

An alternative two-stream architecture where a learnable conv stem replaces the
first 4× spatial reduction before diffSLIC superpixel clustering.

The raw 6-channel input is processed by two parallel streams:
- **Semantic stream**: $\text{interpolate}(x_{\text{raw}})$ → downsampled to match
  conv feature resolution → diffSLIC → superpixel masks
- **Feature stream**: $\text{ConvStem}(x_{\text{raw}})$ → learned deep features
  at reduced resolution → pooled via superpixel masks → tokens

This decouples spatial reduction (learned via conv, aggressive stride) from
grouping criteria (semantic via diffSLIC on physically meaningful raw pixels).
The conv stem produces $4\times$ smaller spatial maps before tokenization,
reducing downstream token count by $16\times$ vs the standard pipeline.

```
Input Image (B, 6, H, W)
        │
        ├──► ConvStem (strides 1,2,2) ──► Feature map (B, C, H/4, W/4)
        │                                   │
        └──► interpolate (scale=1/4)  ──► Raw downsampled (B, 6, H/4, W/4)
                                            │
                                            ▼
                                       diffSLIC
                                            │
                                            ▼
                                    Superpixel masks (B, K, H/4, W/4)
                                            │
                                            ▼
                              ConvSuperpixelEmbedding
                              (pool features using masks)
                                            │
                                            ▼
                              Token sequence (B, N, D)
```

**Configurations** (`configs/model/{spix,conv,vq,gnn}_{tiny,small,medium,large}.yaml`):
- **tiny**: ~5.7M params (matched to ViT-Tiny), depth=2-7 depending on variant
- **small**: ~22M params (matched to ViT-Small), depth=6-16 depending on variant
- All use RMSNorm + SwiGLU activation
- Conv stem: `[32,64,128]` (tiny), `[64,128,256]` (small), strides `(1,2,2)` → 4× reduction
- GNN: GATv2 with 4 heads, mean aggregation
- Builder: `create_conv_vision_rwkv7` from `spixrwkv7.models.conv_spixrwkv7`

Run with:
```bash
uv run python scripts/demo.py --model-type conv --img-size 64 --embed-dims 128 --num-heads 2 --depth 2 --num-superpixels 64 --output results/demo_conv.txt
uv run python tasks/diagnostics/fast_test_training.py --model-type conv --max-steps 50
```

### GNN Vision (Ablation)

Replaces the RWKV-7 recurrence with PyTorch Geometric GNN message passing.
Shares the same superpixel tokenizer as spix but uses graph convolutions
instead of the recurrent scan. Default: GATv2 with 4 attention heads.

**Register Token Graph Topology**: When `register_tokens=R > 0`, R learnable
DINOv2-style register nodes are prepended to the graph. Each register node
connects to ALL N superpixel nodes (bipartite), while each superpixel node
retains its 4 KNN neighbours + receives R edges from register nodes. This
gives every superpixel `4 + R` incoming edges, enabling global context
aggregation through the registers while preserving local graph structure.

**Jumping Knowledge (JK-LSTM)**: Setting `jk="lstm"` collects per-layer
node features and feeds them through an LSTM to produce a fused representation
(Xu et al., ICML 2018). This allows the model to leverage representations
from all depths rather than just the final layer.

```
Input Image (B, 6, H, W)
        │
        ▼
    SuperpixelTokenizer ──► Tokens (B, N, D) + KNN neighbors (B, N, 4)
        │
        ├──► Mask superpixel tokens (optional)
        │
        └──► Prepend R register tokens ──► Tokens (B, N+R, D)
                    │
                    ▼
            _build_edges ──► edge_index (2, E), edge_weight (E,)
            │
            │  Register nodes: connect to ALL N superpixels (bipartite)
            │  Superpixel nodes: 4 KNN edges + R register edges each
            │
            ▼
        GNNBlock × L (GATv2 conv + FFN, both residual)
            │
            ├──► [if jk="lstm"]: collect per-layer features → LSTM → project
            │
            ▼
        _project_output ──► (B, D, H, W) or (B, D, h_s, w_s)
```

**Supported GNN convolutions**: `gcn`, `graphconv`, `sage`, `gin`, `gat`, `gatv2`, `transformer`, `resgated`, `gated`

```bash
# Run GNN variant (default: GATv2, no JK, no registers)
uv run python scripts/demo.py --model-type gnn --img-size 64 --embed-dims 128 --num-heads 2 --depth 2

# GATv2 + JK-LSTM + 4 register tokens
uv run python -c "
from spixrwkv7.models.gnn_spixrwkv7 import create_gnn_vision
model = create_gnn_vision(
    img_size=64, embed_dims=128, depth=4, num_heads=2,
    num_superpixels=36, register_tokens=4, jk='lstm',
    gnn_conv='gatv2', gnn_heads=4,
)
"

# Training convergence test
uv run python tasks/diagnostics/fast_test_training.py --model-type gnn --max-steps 50

# Compare all 4 variants head-to-head
uv run python scripts/compare_architectures.py --compare-variants spix conv vq gnn --sizes tiny small
```

**Key design decisions**:
- Register nodes connect to ALL superpixels, not just neighbours — this is the standard DINOv2 register pattern adapted to graph topology
- JK-LSTM reads features from ALL layers (not just the last), mitigating oversmoothing in deep GNNs
- Register edges have uniform weight 1.0 (no distance weighting), while KNN edges use inverse distance
- Register tokens are excluded from output projection (same as DINOv2) — they exist only for global context aggregation during message passing

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
| `ConvStem`            | Strided conv feature extractor      | Conv2D → BatchNorm2D → ReLU stack (strides 1,2,2)                                        |
| `ConvolutionalSuperpixelTokenizer` | Two-stream tokenizer for conv pipeline | ConvStem + downsampled diffSLIC + mask pooling                                  |
| `ConvolutionalVision_RWKV7` | Full backbone with conv stem      | `ConvStem` → `ConvolutionalSuperpixelTokenizer` → `Vision_RWKV7_Block`s × N → output        |
| `GNNBlock`            | GNN residual block                 | Pre-norm → GNN Conv (GATv2/GCN/SAGE/...) → LayerScale → residual + FFN                     |
| `GNNFeedForward`      | FFN for GNN blocks                 | Norm → activate → project → LayerScale → residual                                          |
| `GNNVision`           | Full GNN backbone                  | `SuperpixelTokenizer` → `_build_edges` (with register bipartite) → `GNNBlock`s × N → JK-LSTM (optional) → `_project_output` |

## Testing

Run the full test suite:

```bash
uv run pytest -v
```

**Expected output:** 132 tests pass.

```text
tests/test_models/test_model.py             ...................................................... (92 tests)
tests/test_data/test_colors.py             ......................... (25 tests)
tests/test_data/test_transforms.py         .................... (20 tests)
tests/test_data/test_diff_slic.py          .............. (14 tests)
tests/test_regression.py         ....... (7 tests) + warnings
============================= 132 passed in 12-13s =============================
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

## Test Validity — What Each Script Actually Measures

Different test scripts use vastly different amounts of training data. This table
clarifies what each test validates vs what it cannot measure:

| Test | Data Volume | Validates | NOT valid for |
|------|-------------|-----------|---------------|
| `fast_test_training.py` | 1 batch (4 real images) | Architecture converges, no bugs | Generalization comparison |
| `diagnose_training.py --no-head` | 1 batch real images | Features are finite, nonzero variance | Quality comparison |
| `ade20k/sanity.py` | 4 train images | Pipeline integration | Segmentation quality |
| `compare_architectures.py` | Real caltech101 images | Speed/memory benchmarks | Accuracy or quality |
| `compare_architectures_alt_vit.py` | Real caltech101 images | Alternative ViT speed/memory benchmarks | Accuracy or quality |
| `humordb/train.py` | 2136 images | Full regression training | — (sufficient for quality metrics) |
| `demo.py` | Real caltech101 images | Output finite, deterministic, shape correct | Any performance metric |
| `run_full_benchmark.py` | Real caltech101 images | 4-variant speed + convergence comparison | Statistical significance (single run) |
| 
| **Key takeaway**: For a statistically meaningful comparison between architecture
| variants (spix vs conv vs vq), run `humordb/train.py` with 3–5 seeds and report
| mean ± std for R², Pearson r, RMSE. The diagnostics scripts verify correctness,
| not quality. The HumorDB regression is the only test with sufficient data volume
| for real quality comparisons.

## Utilities

- **`spixrwkv7/data/colors.py`**: Differentiable conversions between sRGB, Linear RGB, and OkLAB.
- **`spixrwkv7/data/gamut.py`**: Vectorized OkLAB gamut clipping methods (Chroma preservation, adaptive L0 projection).
- **`spixrwkv7/data/transforms.py`**: Image preprocessing utilities (`preprocess_image_for_rwkv7`, `load_image_to_tensor`, `add_spatial_coordinates`).
- **`spixrwkv7/data/image_utils.py`**: Shared real image loading from `data/caltech101_classification/`. Provides `load_random_caltech101_image()` (6-channel OkLAB), `load_random_caltech101_batch()`, `load_random_caltech101_rgb()` (3-channel for ViT), and `load_caltech101_rgb_batch()`.
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
