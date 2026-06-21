# FlashFusion Documentation

Welcome to the FlashFusion documentation. FlashFusion is a multi-model vision fusion framework that enables ensemble, cascade, and multi-task pipelines for detection, classification, segmentation, and OCR models.

## Overview

FlashFusion provides:

- **Ensemble Fusion** — Combine predictions from multiple detection models using Weighted Box Fusion, Voting, or NMS strategies
- **Cascade Pipelines** — Chain models sequentially for progressive refinement
- **Multi-Task Fusion** — Fuse detection + classification + segmentation in a unified pipeline
- **Strategy Registry** — Plug-and-play fusion strategies via the registry pattern
- **LoRA/QLoRA** — Efficient fine-tuning of fusion layers with frozen base models
- **ONNX Export** — Deploy fused models on edge devices

## Quick Links

| Page | Description |
|------|-------------|
| [Installation](Installation.md) | Install FlashFusion and dependencies |
| [Quick Start](Quick-Start.md) | Get running in 5 minutes |
| [Models](Models.md) | FlashFusion model architecture |
| [Fusion Strategies](Fusion-Strategies.md) | WBF, Voting, Cascade, Stacking, NMS |
| [Pipelines](Pipelines.md) | Pre-built multi-task pipelines |
| [Training](Training.md) | Train fusion layers |
| [FAQ](FAQ.md) | Frequently asked questions |

## Architecture

```
FlashFusion
├── models/         # FlashFusion model, LoRA adapters
├── strategies/     # Fusion strategies (WBF, Voting, Cascade, etc.)
├── pipelines/      # Pre-built multi-model pipelines
├── engine/         # Training, validation, prediction, export
├── solutions/      # High-level APIs (EnsembleDetector, MultiModelAnalyzer)
├── analytics/      # Benchmarking, profiling, plotting
├── data/           # Dataset, DataLoader, transforms
├── losses/         # Multi-task and consistency losses
├── utils/          # Checkpoints, logging, metrics, visualization
├── nn/             # Neural network building blocks
└── cfg/            # Configuration management
```

## Citation

If you use FlashFusion in your research, please cite:

```bibtex
@software{flashfusion2024,
  title={FlashFusion: Multi-Model Vision Fusion},
  author={FlashVision},
  year={2024},
  url={https://github.com/FlashVision/FlashFusion}
}
```
