# Vision-RWKV-7 with Superpixel Tokenization

A PyTorch implementation of a **Vision-RWKV-7** backbone, enhanced with differentiable superpixel tokenization (`diffSLIC`), Graph-Based Q-Shift, and bidirectional scanning.

This architecture merges the linear-complexity, constant-memory advantages of the **RWKV-7** recurrent state-space model with vision-specific adaptations inspired by **Vision-RWKV** and **AudioRWKV**, while introducing a novel **irregular grid tokenization** pipeline.

> **NOTICE:** This repository is a learning project from a single person who is a beginner on the field, I started from a pytorch implementation of RWKV-7 and adapted it for vision tasks with superpixel tokenization. More things may be added as an way of exploring the design space of RWKV-based vision backbones.

## Key Features

- **Differentiable Superpixel Tokenization**: Replaces rigid patch grids with `diffSLIC`, supporting both **hard** (discrete) and **soft** (continuous, fully differentiable) aggregation modes.
- **Perceptual Color Space Support**: Native, fully differentiable support for the **OkLAB** color space, including sRGB/Linear RGB conversions and robust **Gamut Clipping** methods.
- **Graph-Based Q-Shift**: Adapts the original 2D grid Q-Shift to operate on K-Nearest Neighbor (KNN) graphs, allowing spatial mixing to dynamically adapt to irregular superpixel topologies.
- **Bidirectional Scanning (Bi-WKV)**: Processes the token sequence in both forward and backward directions, fusing them via a dynamic gating mechanism to capture full global context with $O(N)$ complexity.
- **Scatter-Back-to-Grid**: Automatically maps the irregular sequence of superpixel tokens back to a dense `[B, C, H, W]` tensor at the output, ensuring seamless compatibility with downstream dense prediction heads.
- **Robust Data Utilities**: Includes tools for calculating dataset-wide mean/std statistics and stable image loading with resolution interpolation.
- **RWKV-7 Stability**: Inherits RWKV-7's generalized delta rule, flexible decay, bounded exponentials, value residuals, and Layer Scale for robust, scalable training.

## Installation

This repository is optimized for modern Python environments. We recommend using [`uv`](https://github.com/astral-sh/uv) for fast dependency resolution, though standard `pip` works perfectly.

```bash
# Clone the repository
git clone https://github.com/your-username/Visual_RWKV7_Pytorch.git
cd Visual_RWKV7_Pytorch

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using uv (or uv sync)
uv pip install torch torchvision torchaudio
uv pip install pytest numpy scipy scikit-image matplotlib
```

## Quick Start

You can instantiate the backbone and run a forward pass with just a few lines of code. The model automatically handles superpixel generation, graph construction, and grid scattering.

```python
import torch
from VisualRWKV7.model import Vision_RWKV7
from VisualRWKV7.utils.data import load_image_to_tensor

# Initialize the model
model = Vision_RWKV7(
    img_size=224,
    in_chans=3,
    embed_dims=192,
    num_heads=3,
    depth=12,
    num_superpixels=196,      # Target number of superpixels (approx 14x14)
    diff_slic_iters=5,        # Iterations for diffSLIC optimization
    out_indices=[3, 5, 7, 11] # Multi-scale feature extraction
)

# Load and preprocess an image (supports OkLAB conversion)
x = load_image_to_tensor(
    "path/to/image.jpg", 
    target_size=(224, 224), 
    color_space="oklab", 
    normalize=True
)

# Forward pass
outs = model(x)

print(f"Input shape:  {tuple(x.shape)}")
print(f"Output levels: {len(outs)}")
for i, o in enumerate(outs):
    print(f"  Level {i} shape: {tuple(o.shape)}")
```

## Testing

The repository includes a comprehensive test suite covering color conversions, dataset utilities, diffSLIC mechanics, and model invariants.

Run the full test suite using `pytest`:

```bash
uv run pytest -v
```

**Expected Output:**

```text
tests/test_colors.py ....................                                [ 25%]
tests/test_dataload.py ............                                      [ 40%]
tests/test_diffSlic.py .............                                     [ 56%]
tests/test_model.py ...........................                          [ 91%]
tests/test_regression.py .......                                         [100%]
========================= 79 passed in X.XXs =========================
```

## Architecture Overview

1. **Preprocessing**: Optional conversion from sRGB to OkLAB perceptual color space and normalization using dataset-wide statistics.
2. **Tokenization (`diffSLIC`)**: The input image is processed by `DiffSLIC` to generate soft or hard superpixel assignments.
3. **Embedding**: Pixels are aggregated into superpixel tokens via weighted mean pooling (`SuperpixelEmbedding`).
4. **Graph Construction**: Centroids of the generated superpixels are used to build a batched K-NN graph (`build_knn_graph`).
5. **Vision-RWKV-7 Blocks**:
   - **Graph Q-Shift**: Tokens are shifted along graph edges to provide local spatial inductive bias.
   - **Bi-WKV Scan**: Forward and backward recurrent passes compute the generalized delta rule state updates.
   - **Gated Fusion**: Forward and backward outputs are blended using a learned gate.
6. **Scatter Back**: For multi-scale outputs, tokens are scattered back to their original pixel coordinates using `torch.gather` (hard mode) or `torch.einsum` (soft mode), restoring the `[B, C, H, W]` shape.

## Utilities

- **`utils/colors.py`**: Differentiable conversions between sRGB, Linear RGB, and OkLAB.
- **`utils/gamut.py`**: Vectorized OkLAB gamut clipping methods (Chroma preservation, adaptive L0 projection).
- **`utils/data.py`**: Dataset statistics calculation and robust image loading pipelines.
- **`utils/graph.py`**: KNN graph construction and multi-head graph-based token shifting.
- **`utils/drop.py`**: Stochastic depth (DropPath) implementation.

## References & Inspirations

This implementation builds upon several foundational works. Please consider citing them if you use this code in your research:

- **RWKV-7**: Peng, B., et al. "RWKV-7 'Goose' with Expressive Dynamic State Evolution." _arXiv preprint arXiv is ongoing_ (2024/2025).
- **Vision-RWKV**: Duan, Y., et al. "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-like Architectures." _ICLR 2025_.
- **AudioRWKV**: Wang, J., et al. "AudioRWKV: Efficient and Stable Bidirectional RWKV for Audio Pattern Recognition." _arXiv preprint_ (2024).
- **diffSLIC**: (Add specific diffSLIC paper citation here if applicable, or link to the original repository).

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Visual_RWKV7_Pytorch/issues).

## License

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](LICENSE) file for details, aligning with the upstream RWKV project.

---

_Built with ❤️ for learning about efficient, scalable, and adaptive computer vision._
