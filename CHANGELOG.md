# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-06-21

### Added
- **Attention Residuals (AttnRes)**: Implemented depth-wise attention residuals (`block_attn_res`) for both standard and optimized blocks, replacing fixed additive residuals with learned softmax attention over preceding layer/block representations.
- **AttnRes Gating Options**: Supported `"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, and `"learnable_alpha"` gating configurations for the AttnRes mixing step.
- **AttnRes History Modes**: Supported `"block"` (block boundary only) and `"full"` sequence tracking.
- **Depth-Aware Feature Heads**: Adapted `ClassificationHead`, `RegressionHead` (HumorDB), and `SegHead` (ADE20K) to selectively attend to the complete backbone sequence history, resolving data dilution and improving training efficiency.
- **Alternative Superpixel Backends**: Added support for `"grid"`, `"slic"`, `"slico"`, and `"lnsnet"` superpixel tokenization backends in `SuperpixelTokenizer`.
- **LNS-Net Integration**: Implemented learnable superpixel segmentation (LNS-Net, CVPR 2021) with support for automated BSDS checkpoint download and weight loading.
- **Architectural Enhancements**: Configurable normalization layers (`norm_layer="layernorm"|"rmsnorm"`) and activation functions (`act_layer="relu2"|"gelu"|"silu"|"swiglu"`).
- **Register Tokens**: DINOv2-style learnable register tokens (`register_tokens=N`) for global context accumulation.
- **Dynamic Image Scaling**: Support for flexible resolution heights (`img_size=-1` / `img_size>0`).
- **Attention Residuals (AttnRes)**: Implemented depth-wise attention residuals (`block_attn_res`) for both standard and optimized blocks, replacing fixed additive residuals with learned softmax attention over preceding layer/block representations
- **AttnRes Gating Options**: Supported `"bias"`, `"sigmoid_scalar"`, `"sigmoid_vector"`, and `"learnable_alpha"` gating configurations for the AttnRes mixing step
- **AttnRes History Modes**: Supported `"block"` (block boundary only) and `"full"` sequence tracking
- **Depth-Aware Feature Heads**: Adapted `ClassificationHead`, `RegressionHead` (HumorDB), and `SegHead` (ADE20K) to selectively attend to the complete backbone sequence history, resolving data dilution and improving training efficiency
- **Alternative Superpixel Backends**: Added support for `"grid"`, `"slic"`, `"slico"`, and `"lnsnet"` superpixel tokenization backends in `SuperpixelTokenizer`
- **LNS-Net Integration**: Implemented learnable superpixel segmentation (LNS-Net, CVPR 2021) with support for automated BSDS checkpoint download and weight loading
- **Architectural Enhancements**: Configurable normalization layers (`norm_layer="layernorm"|"rmsnorm"`) and activation functions (`act_layer="relu2"|"gelu"|"silu"|"swiglu"`)
- **Register Tokens**: DINOv2-style learnable register tokens (`register_tokens=N`) for global context accumulation
- **Dynamic Image Scaling**: Support for flexible resolution heights (`img_size=-1` / `img_size>0`)

### Changed
- **Modular Refactoring**: Reorganized the architecture into separate modular classes: `RecurrentScan`, `SpatialMixer`, `ChannelMix`, `SuperpixelTokenizer`, `_DynamicOffset`, and `_TimeMixParams`.
- **Project Structure**: Relocated the core backbone definition file from `spixrwkv7/spixrwkv7.py` to `spixrwkv7/models/spixrwkv7.py`.
- **Inference & Demo Updates**: Expose `--use-attnres` option in `scripts/demo.py` and diagnostic training script parameters.
- **Modular Refactoring**: Reorganized the architecture into separate modular classes: `RecurrentScan`, `SpatialMixer`, `ChannelMix`, `SuperpixelTokenizer`, `_DynamicOffset`, and `_TimeMixParams`
- **Project Structure**: Relocated the core backbone definition file from `spixrwkv7/spixrwkv7.py` to `spixrwkv7/models/spixrwkv7.py`
- **Inference & Demo Updates**: Expose `--use-attnres` option in `scripts/demo.py` and diagnostic training script parameters