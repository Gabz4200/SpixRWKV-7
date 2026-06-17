# Vision-RWKV-7 with Superpixel Tokenization

A PyTorch implementation of a **Vision-RWKV-7** backbone, enhanced with differentiable superpixel tokenization (`diffSLIC`), Graph-Based Q-Shift, and bidirectional scanning.

This architecture merges the linear-complexity, constant-memory advantages of the **RWKV-7** recurrent state-space model with vision-specific adaptations inspired by **Vision-RWKV** and **AudioRWKV**, while introducing a novel **irregular grid tokenization** pipeline.

> **NOTICE:** This repository is a learning project from a single person who is a beginner on the field, I started from a pytorch implementation of RWKV-7 and adapted it for vision tasks with superpixel tokenization. More things may be added as an way of exploring the design space of RWKV-based vision backbones.

## Key Features

- **Differentiable Superpixel Tokenization**: Replaces rigid patch grids with `diffSLIC`, supporting both **hard** (discrete) and **soft** (continuous, fully differentiable) aggregation modes.
- **Graph-Based Q-Shift**: Adapts the original 2D grid Q-Shift to operate on K-Nearest Neighbor (KNN) graphs, allowing spatial mixing to dynamically adapt to irregular superpixel topologies.
- **Bidirectional Scanning (Bi-WKV)**: Processes the token sequence in both forward and backward directions, fusing them via a dynamic gating mechanism to capture full global context with $O(N)$ complexity.
- **Scatter-Back-to-Grid**: Automatically maps the irregular sequence of superpixel tokens back to a dense `[B, C, H, W]` tensor at the output, ensuring seamless compatibility with standard downstream dense prediction heads (e.g., UperNet, Mask R-CNN).
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

# Dummy input: [Batch, Channels, Height, Width]
x = torch.randn(2, 3, 224, 224)

# Forward pass
outs = model(x)

print(f"Input shape:  {tuple(x.shape)}")
print(f"Output levels: {len(outs)}")
for i, o in enumerate(outs):
    print(f"  Level {i} shape: {tuple(o.shape)}")
    # Note: Outputs are scattered back to [B, C, H, W]!
```

## Testing

The repository includes a comprehensive test suite to verify the mathematical correctness of the graph shifts, superpixel embedding, and RWKV-7 delta rule mechanics.

Run the tests using `pytest`:

```bash
export PYTHONPATH=.
pytest tests/test_model.py -v
```

**Expected Output:**

```text
========================= 12 passed in X.XXs =========================
```

## Architecture Overview

1. **Tokenization (`diffSLIC`)**: The input image is processed by `DiffSLIC` to generate soft or hard superpixel assignments.
2. **Embedding**: Pixels are aggregated into superpixel tokens via weighted mean pooling (`SuperpixelEmbedding`).
3. **Graph Construction**: Centroids of the generated superpixels are used to build a batched K-NN graph (`build_knn_graph`).
4. **Vision-RWKV-7 Blocks**:
   - **Graph Q-Shift**: Tokens are shifted along graph edges to provide local spatial inductive bias.
   - **Bi-WKV Scan**: Forward and backward recurrent passes compute the generalized delta rule state updates.
   - **Gated Fusion**: Forward and backward outputs are blended using a learned gate.
5. **Scatter Back**: For multi-scale outputs, tokens are scattered back to their original pixel coordinates using `torch.gather` (hard mode) or `torch.einsum` (soft mode), restoring the `[B, C, H, W]` shape.

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
