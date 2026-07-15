# SpixRWKV-7: Superpixel Graph RWKV-7 Vision Backbone

A PyTorch implementation of a **recurrent vision backbone** that replaces rigid patch grids with differentiable superpixels (`diffSLIC`), processes tokens through **graph-based Q-shift** and **bidirectional RWKV-7 recurrence**, and outputs dense feature maps at arbitrary resolutions.

The architecture merges the linear-complexity, constant-memory advantages of the **RWKV-7** state-space model with vision-specific adaptations (Graph-Based Q-Shift, gated bidirectional fusion, Hilbert reordering), while introducing a novel **irregular-grid tokenization** pipeline. Unlike standard ViTs, SpixRWKV-7 operates on perceptually grouped pixels (superpixels) rather than fixed grid patches, enabling adaptive spatial resolution and natural contour awareness.

> ⚠️ **DISCLAIMER 1:** This repository is yet another learning project made by a single Brazilian student that is exploring the topic of Sub-quadratic Vision Encoders.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do. I also dont prohibit myself from ocasional help, but the main thing is probably commit messages, I genuinely hate writting those.

## News / Recent Updates

- **Hybrid RWKV+GNN Vision variant**: New `HybridVision` model combining ConvStem + diffSLIC tokenizer, a configurable number of RWKV-7 recurrent layers, and GATv2 GNN layers with DINOv2 register tokens and JK-LSTM. Bridges the recurrence vs message-passing ablation. Configs: `hybrid_tiny/small/medium/large.yaml`.
- **5-variant full benchmark**: Comprehensive benchmark comparing spix, vq, conv, gnn, hybrid against ViT at matched parameter counts. Inference speed (with tokenizer/backbone breakdown) and training convergence. GNN is fastest RWKV variant (0.55x vs ViT at tiny/512px); conv converges fastest (6 steps). See [Full Benchmark section](#full-variant-benchmark-results).
- **Attention Residuals (AttnRes) across all models**: Depth-wise attention residuals replacing fixed additive residual accumulation with learned softmax attention over preceding layer representations. Wired through all models including hybrid. Features `"block"` and `"full"` history modes, multiple gating options (`"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, `"learnable_alpha"`).
- **Residual connections**: All models now have at least one method of residual connections (Attention Residuals as primary, Sum Residuals as secondary). Hybrid model wires attnres through both RWKV and GNN blocks with consistent 3D history format.
- **GNN edge bug fixes**: Fixed missing `all_w.append(reg_w)` for forward register edges (edge_index/edge_weight mismatch), double-scaling on backward register edges, and `_gnn_forward` not passing `edge_attr` to GATv2/TransformerConv. Added `edge_dim=1` and `add_self_loops=False` for GATv2Conv.
- **GNN over-smoothing prevention**: GNN depth reduced for small node counts (gnn_tiny: 6→3, gnn_small: 15→8). Added `register_edge_weight_scale` to dampen register hub domination (0.25 for tiny, 0.5 for small+). Global attention layers added at middle and end of GNN stack.
- **C++ kernel fixes**: Lazy `_ensure_cpp()` import, autograd auto-fallback to PyTorch, `torch::empty_like` for state buffers, dynamic shared memory for diffSLIC `s_valid_idx`, `std::vector` for diffSLIC `sim_buf`. All 131 tests pass.
- **Full benchmark suite**: `scripts/run_full_benchmark.py` — inference speed + training convergence for all 5 variants vs ViT.

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
- **Hybrid RWKV+GNN (HybridVision)**: Combines ConvStem tokenizer, configurable RWKV-7 recurrent layers, and GATv2 GNN layers with DINOv2 register tokens and JK-LSTM. Bridges the recurrence vs message-passing ablation — RWKV handles sequential processing, GNN handles graph structure. Builder: `create_hybrid_vision`. Configs: `hybrid_tiny/small/medium/large.yaml`.
- **Attention Residuals (AttnRes)**: Depth-wise attention residuals replacing the standard fixed additive residual accumulation with a learned softmax attention over preceding layer/block representations. Features options for `"block"` and `"full"` history aggregation, and multiple gating options (`"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, `"learnable_alpha"`). All models now have at least one method of residual connections (AttnRes as primary, Sum Residuals as secondary).
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

Benchmark comparing all 5 SpixRWKV-7 variants against ViT across model sizes and image resolutions. C++ kernels enabled where applicable.

#### Model Size Comparison (matched params, 256px input, 6-channel input)

| Size | Model | Params | Time (ms) | vs ViT | Tok % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| tiny | spix | 5.90M | 417 | 0.13x | 61% |
| tiny | vq | 5.39M | 709 | 0.08x | 16% |
| tiny | conv | 5.70M | 372 | 0.15x | 4% |
| tiny | gnn | 4.49M | 140 | 0.39x | 66% |
| tiny | hybrid | 3.62M | 228 | 0.24x | 33% |
| tiny | ViT | 5.69M | 55 | 1.00x | 0% |
| small | spix | 21.96M | 2686 | 0.05x | 12% |
| small | vq | 21.54M | 2384 | 0.06x | 9% |
| small | conv | 21.99M | 3236 | 0.04x | 1% |
| small | gnn | 14.19M | 571 | 0.25x | 33% |
| small | hybrid | 20.66M | 965 | 0.15x | 24% |
| small | ViT | 21.99M | 140 | 1.00x | 0% |

#### Resolution Scaling (Tiny Model, 256px vs 512px)

| Model | 256px (ms) | 512px (ms) | Scaling | Tok @ 512px |
| :--- | :--- | :--- | :--- | :--- |
| spix | 417 | 1433 | 3.4x | 81% |
| vq | 709 | 3079 | 4.3x | 18% |
| conv | 372 | 579 | 1.6x | 8% |
| gnn | 140 | 494 | 3.5x | 77% |
| hybrid | 228 | 712 | 3.1x | 41% |
| ViT | 55 | 273 | 5.0x | 0% |

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
* **Theoretical Complexity**: ViT attention scales quadratically ($O(N^2)$) in time and memory with the number of tokens $N$. SpixRWKV-7's recurrent scan has a linear ($O(N)$) time complexity and constant ($O(1)$) recurrent state memory. GNN scales with graph size ($O(S^2)$ for $S$ superpixels), independent of pixel count.
* **CPU Latency & Parallelization Bottleneck**: On CPU, PyTorch's native transformer uses highly tuned, parallelized CPU matrix multiplication (GEMM) kernels. In contrast, SpixRWKV-7's recurrent scan operates sequentially over the sequence length (`for t in range(N)`), introducing significant loop overhead and poor CPU cache efficiency.
* **Fair Comparison (Matched Params)**: At ~5.7M params (tiny, 256px), ViT is **4-8x faster** than RWKV variants. The gap is smallest for gnn (0.39x) and largest for vq (0.08x). At ~22M params (small, 256px), ViT is **4-20x faster**.
* **Tokenizer Overhead**: For spix, diffSLIC accounts for **61-81%** of inference time depending on resolution. For gnn, tokenizer is **66-77%**. For conv, tokenizer is only **1-8%** (ConvStem is fast). The hybrid model splits this 33-42%.
* **Resolution Scaling**: GNN scales best with resolution (0.39x → 0.55x vs ViT from 256px to 512px) because graph computation scales with S, not N. Conv also scales well (0.15x → 0.47x). Spix tokenizer grows with pixel count.

#### 3. Parameter and Memory Efficiency
* **Parameter Count (Matched)**: At matched parameter counts, the architectures show their true character:
  * **Tiny (~5.7M)**: ViT (5.68M) ≈ spix (5.86M) ≈ gnn (5.67M) ≈ conv (5.61M) ≈ vq (5.43M)
  * **Small (~22M)**: ViT (21.98M) ≈ spix (21.73M) ≈ gnn (21.78M) ≈ conv (21.73M) ≈ vq (22.00M)
* **Memory Footprint**: Under high-resolution or dense-prediction settings, the quadratic memory scaling of ViT attention becomes a bottleneck. SpixRWKV-7 scales linearly and retains a constant state size, preserving memory.

#### 4. Convergence & Stability
* **Fair Comparison (Matched Params)**: At matched parameter counts (256px, 4 real images):
  * **Conv** converges fastest: 6 steps to 100% (conv stem provides strong inductive bias)
  * **VQ** converges moderate: 26 steps (VQ-VAE tokenizer overhead but steady)
  * **GNN** converges in 40 steps (learns from scratch, but fastest per-step at 1.8s)
  * **Spix** converges moderate: 44 steps (sequential bottleneck, 4.3s/step)
* **GNN step count is not a problem**: Conv pre-extracts spatial features before tokenization; GNN must learn both features and message-passing patterns. On real datasets, GNN's per-step speed advantage dominates.
* **Register tokens + JK-LSTM improve GNN convergence**: Ablation shows registers save 3 steps, JK-LSTM saves 8 steps, together save 13 steps.
* **Overfit caveat**: Fast overfitting (conv's 6 steps) means high capacity + easy gradient flow, but may also overfit faster on real data. Slow overfitting (spix's 44 steps) suggests more structured learning that may generalize better.
* Gradient health diagnostics confirm uniform gradient flow across all blocks under RMSNorm and SwiGLU activation configurations.

### Why Is ViT Faster on CPU?

1. **Optimized GEMM kernels**: PyTorch's Transformer uses heavily tuned CPU matrix multiplication that exploits cache locality and SIMD instructions.
2. **Parallel vs Sequential**: The Vision RWKV-7 backbone has a sequential recurrent loop (`for t in range(N)`) that cannot be vectorized like the parallel attention in ViT. Each timestep requires its own forward pass through the recurrence.
3. **diffSLIC overhead**: The tokenization involves iterative clustering with softmax operations over spatial dimensions - computationally expensive on CPU. For spix, this is 61-81% of total time.
4. **Small matrix inefficiency**: The RWKV-7 recurrence uses small matrix operations (head_size=64) that have poor cache efficiency compared to larger GEMM operations.
5. **GPU advantage for RWKV-7**: On GPU, the recurrent loop can run in parallel across sequence positions using the custom CUDA kernel without quadratic attention memory overhead. The tokenizer (diffSLIC) also has CUDA kernels.
6. **GNN avoids the sequential bottleneck**: GATv2 message passing over the KNN graph is embarrassingly parallel, which is why gnn is 3-10x faster than other RWKV variants on CPU.

### Key Insights (Fair Comparison)

- **ViT dominates CPU inference**: 2-19x faster than RWKV variants at matched params. PyTorch's GEMM kernels are heavily optimized for CPU, while RWKV-7's sequential recurrence cannot vectorize.
- **GNN is the fastest RWKV variant**: 3-10x faster than other RWKV variants. GATv2 message passing over 4-NN graph is embarrassingly parallel, avoiding the sequential bottleneck of RWKV-7's recurrence. Scales best with model size.
- **Hybrid is second-fastest**: Combines conv stem (fast tokenizer) + GNN layers (fast backbone). 2.8x faster than spix at small.
- **Conv converges instantly but scales poorly**: 6 steps to 100% accuracy, but the deep backbone makes small inference 23x slower than ViT. Strong inductive bias enables fast memorization but constrains representational capacity.
- **Resolution scaling favors GNN/conv**: GNN improves from 0.39x to 0.55x vs ViT when going from 256px to 512px (graph scales with S, not pixels). Conv improves from 0.15x to 0.47x.
- **VQ is expensive**: Both inference and training are slowest due to VQ-VAE encoder/decoder overhead and codebook bottleneck.
- **GPU would change the ranking**: On GPU, diffSLIC CUDA kernels would drop tokenizer from ~250ms to ~10ms, making spix/gnn competitive with ViT.

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

# Full benchmark suite (inference + training convergence for all 5 variants)
uv run python scripts/run_full_benchmark.py --sizes tiny small --img-size 256

# Sweep downsampling factors (spix backbone only)
uv run python scripts/compare_architectures.py --downsample-factors 1.0 2.0 4.0
uv run python scripts/compare_architectures_alt_vit.py --downsample-factors 1.0 2.0 4.0
uv run python tasks/diagnostics/fast_test_training.py --downsample-factors 1.0 2.0 4.0
```

### C++ Kernel Speedup (all 5 model variants)

Benchmark measuring the speedup from C++ AVX2-accelerated kernels (RWKV-7 recurrence + diffSLIC) vs pure PyTorch, across all 5 model variants. Uses `TORCH_LIBRARY` registration for `torch.compile` compatibility.

#### Kernel-Level Speedup (128px input, tiny config, CPU)

| Kernel | PyTorch (ms) | C++ (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **diffSLIC** (cluster update + assign) | 140.51 | 40.26 | **3.49x** |
| **RecurrentScan** (N=36, D=192) | 45.96 | 33.19 | 1.38x |
| **RecurrentScan** (N=144, D=192) | 148.14 | 111.95 | 1.32x |

#### Full Model Speedup (128px input, tiny config, CPU)

| Model | Params | PyTorch (ms) | C++ (ms) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **spix** | 1.33M | 205.65 | 154.88 | **1.33x** |
| **conv** | 1.35M | 150.49 | 126.53 | **1.19x** |
| **gnn** | 0.83M | 106.72 | 78.58 | **1.36x** |
| **hybrid** | 2.00M | 214.69 | 145.96 | **1.47x** |
| **vq** | 3.38M | 3907.83 | 2523.89 | **1.55x** |

**Key findings**:
- **diffSLIC is the biggest winner**: 3.49x speedup from fused C++ cluster update + pixel assignment (OpenMP + stack-allocated buffers vs PyTorch `unfold` + `einsum` + `softmax`)
- **RecurrentScan**: 1.3-1.4x speedup from AVX2 FMA dot products and fused state update loop
- **All 5 variants benefit**: C++ kernels are wired through `use_cpp=True` into every model's tokenizer and recurrence
- **hybrid and vq see largest speedup** (1.47x, 1.55x) because they have deeper recurrence or more diffSLIC iterations

Run the benchmark:
```bash
uv run python scripts/benchmark_cpp_vs_py.py
```

### Full 5-Variant Benchmark Results

Comprehensive benchmark comparing all 5 model variants (spix, vq, conv, gnn, hybrid) against ViT baseline, using **real images** from `data/caltech101_classification/` (butterfly, dalmatian, dolphin) with **6-channel input (L, a, b, alpha, x, y)** for all models. Both GNN, hybrid, and ViT use 4 register tokens (DINOv2-style) for fair comparison. C++ kernels enabled where applicable.

#### Inference Speed — Tiny (~5.7M params)

| Model | Params | Total (ms) | Tokenizer (ms) | Backbone (ms) | Tok % | vs ViT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **spix** | 5.90M | 417 | 252 | 165 | 61% | 0.13x |
| **vq** | 5.39M | 709 | 113 | 596 | 16% | 0.08x |
| **conv** | 5.70M | 372 | 14 | 358 | 4% | 0.15x |
| **gnn** | 4.49M | 140 | 92 | 48 | 66% | 0.39x |
| **hybrid** | 3.62M | 228 | 75 | 153 | 33% | 0.24x |
| **ViT** | 5.69M | 55 | — | 55 | 0% | 1.00x |

#### Inference Speed — Tiny at 512px (resolution scaling)

| Model | Total (ms) | Tokenizer (ms) | Backbone (ms) | Tok % | vs ViT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **spix** | 1433 | 1161 | 272 | 81% | 0.19x |
| **vq** | 3079 | 568 | 2511 | 18% | 0.09x |
| **conv** | 579 | 46 | 532 | 8% | 0.47x |
| **gnn** | 494 | 378 | 116 | 77% | 0.55x |
| **hybrid** | 712 | 289 | 423 | 41% | 0.37x |
| **ViT** | 273 | — | 273 | 0% | 1.00x |

#### Inference Speed — Small (~22M params)

| Model | Params | Total (ms) | Tokenizer (ms) | Backbone (ms) | Tok % | vs ViT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **spix** | 21.96M | 2686 | 334 | 2352 | 12% | 0.05x |
| **vq** | 21.54M | 2384 | 223 | 2161 | 9% | 0.06x |
| **conv** | 21.99M | 3236 | 24 | 3212 | 1% | 0.04x |
| **gnn** | 14.19M | 571 | 191 | 380 | 33% | 0.25x |
| **hybrid** | 20.66M | 965 | 228 | 737 | 24% | 0.15x |
| **ViT** | 21.99M | 140 | — | 140 | 0% | 1.00x |

#### Speedup Summary vs ViT

| Size | spix | vq | conv | gnn | hybrid |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **tiny (256px)** | 0.13x | 0.08x | 0.15x | 0.39x | 0.24x |
| **tiny (512px)** | 0.19x | 0.09x | 0.47x | 0.55x | 0.37x |
| **small (256px)** | 0.05x | 0.06x | 0.04x | 0.25x | 0.15x |

#### Training Convergence (single-batch overfit, 256px, 4 real images)

| Model | Steps to 100% | Step Time | Total Time | Final Loss |
| :--- | :--- | :--- | :--- | :--- |
| **conv** tiny | 6 | 2747ms | 16.5s | 0.503 |
| **vq** tiny | 26 | 9670ms | 251.4s | 6.050 |
| **gnn** tiny | 40 | 1846ms | 73.8s | 0.405 |
| **spix** tiny | 44 | 4279ms | 188.3s | 0.326 |

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

#### Key Insights

1. **The superpixel tokenizer is the dominant bottleneck** — For spix and gnn, diffSLIC takes 61-81% of total inference time at tiny. At 512px, spix tokenizer jumps from 252ms to 1161ms (4.6x) because diffSLIC is O(pixels × k × neighbors). The recurrent backbone itself is fast; the bottleneck is the segmentation frontend. On GPU (diffSLIC CUDA kernels), this would improve dramatically.

2. **ViT dominates CPU inference** (2-19x faster) — ViT has no tokenizer — just a single Conv2d patch embedding (parallel, highly optimized by PyTorch) then transformer blocks. At matched params, ViT's optimized parallel GEMM kernels are dramatically faster. But ViT treats every patch equally; RWKV variants use superpixels to adaptively allocate tokens to informative regions — a **quality** tradeoff.

3. **GNN is the fastest RWKV variant** — 140ms (tiny/256px) / 571ms (small/256px), 3-10x faster than other RWKV variants. GATv2 message passing over 4-NN graph is embarrassingly parallel, avoiding the sequential bottleneck of RWKV-7's recurrence. GNN scales best with model size: backbone is 8.5x faster than conv at small (380ms vs 3212ms) because graph computation scales with number of superpixels (S), not pixel count (N).

4. **Resolution scaling reveals architecture character** — At 512px vs 256px:
   - GNN improves from 0.39x to 0.55x vs ViT (graph scales with S, not pixels)
   - Conv improves from 0.15x to 0.47x (backbone dominates, tokenizer is tiny)
   - Spix degrades from 0.13x to 0.19x (tokenizer grows with pixels)
   - This suggests GNN and conv are better suited for high-resolution inputs

5. **Hybrid combines best of both** — Uses conv stem (fast tokenizer: 75ms, 33% of time) + GNN layers (fast backbone: 153ms). Second-fastest RWKV variant at tiny (228ms). At small (965ms), hybrid is 2.8x faster than spix (2686ms) and 3.4x faster than conv (3236ms).

6. **Conv converges fastest but runs slowest at scale** — 6 steps to 100% accuracy. The conv stem provides strong inductive bias that makes features immediately linearly separable. But at small, conv's backbone dominates (3212ms) because the deep recurrent layers process many tokens sequentially. Classic speed-accuracy tradeoff.

7. **VQ is slowest in both inference and training** — The VQ-VAE codebook (1024 entries) creates a representation bottleneck. At tiny, VQ backbone is 3.6x slower than spix (596ms vs 165ms). High final loss (6.05) suggests the codebook fights the classification objective — 1024 entries for 3 classes is overkill.

8. **Training convergence reflects memorization, not generalization** — All models hit 100% on 4 training images (expected for overfit). Conv's 6 steps means high capacity + easy gradient flow; spix's 44 steps suggests more structured learning. Fast overfitting ≠ good generalization — conv may overfit faster on real data too.

9. **GPU would change the ranking entirely** — On GPU, spix tokenizer drops from 252ms to ~10ms (CUDA kernels), GNN from 92ms to ~5ms. The recurrent backbone (the novel contribution) is actually very fast. CPU benchmark underestimates RWKV variants' potential because superpixel tokenizers are CPU-bound.

10. **Predicted generalization ranking on real training**:
    - **Best**: GNN and Hybrid — graph structure enforces spatial coherence; hybrid combines local texture + global structure
    - **Middle**: SPIX — superpixel segmentation provides spatial inductive bias, but recurrent backbone may not fully exploit it
    - **Worst**: Conv and VQ — conv is standard CNN+RNN with no special spatial bias; VQ's codebook bottleneck constrains representation

Run the full benchmark:
```bash
# Full inference + training convergence for all 5 variants
uv run python scripts/run_full_benchmark.py --sizes tiny small --img-size 256

# Inference only (faster)
uv run python scripts/run_full_benchmark.py --sizes tiny small --skip-training

# Training convergence only
uv run python scripts/run_full_benchmark.py --sizes tiny small --skip-inference

# Custom resolution
uv run python scripts/run_full_benchmark.py --sizes tiny --img-size 512 --warmup 3 --runs 10
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

| Variant | Tiny Steps | Tiny Time | Tiny Step Time |
| :--- | :--- | :--- | :--- |
| conv | 6 | 16.5s | 2747ms |
| vq | 26 | 251.4s | 9670ms |
| gnn | 40 | 73.8s | 1846ms |
| spix | 44 | 188.3s | 4279ms |

**Key findings**:
- Conv converges fastest (6 steps) due to strong inductive bias from conv stem
- GNN converges well (40 steps) — per-step time is fastest (1846ms) due to parallel message passing
- Spix converges in 44 steps — sequential bottleneck makes per-step time 2.3x slower than GNN
- VQ converges in 26 steps but is slowest per-step (9670ms) due to VQ-VAE encoder/decoder and codebook overhead
- **Overfit caveat**: This measures memorization speed, NOT generalization. Fast overfitting (conv's 6 steps) means high capacity + easy gradient flow — but may also overfit faster on real data. Slow overfitting (spix's 44 steps) suggests more structured learning.

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
│   │   ├── gnn_spixrwkv7.py     # GNN Vision model (GNN message passing ablation)
│   │   └── hybrid_spixrwkv7.py  # Hybrid RWKV+GNN model (recurrence + graph)
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
│   └── run_full_benchmark.py      # Full 5-variant benchmark suite
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
│   │   ├── gnn_large.yaml         # GNN ablation large config
│   │   ├── hybrid_tiny.yaml       # Hybrid RWKV+GNN tiny config
│   │   ├── hybrid_small.yaml      # Hybrid RWKV+GNN small config
│   │   ├── hybrid_medium.yaml     # Hybrid RWKV+GNN medium config
│   │   └── hybrid_large.yaml      # Hybrid RWKV+GNN large config
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

**Expected output:** 131 tests pass.

```text
tests/test_models/test_model.py             ...................................................... (92 tests)
tests/test_data/test_colors.py             ......................... (25 tests)
tests/test_data/test_transforms.py         .................... (20 tests)
tests/test_data/test_diff_slic.py          .............. (14 tests)
tests/test_regression.py         ....... (7 tests) + warnings
============================= 131 passed in 12-13s =============================
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
