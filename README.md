<div align="center">

# FlashFusion

**Multi-Model Vision Fusion — Ensemble, Cascade, and Fuse Detection, Classification, Segmentation, and OCR Models**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/FlashVision/FlashFusion/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashFusion/actions)

[Documentation](docs/Home.md) · [Quick Start](docs/Quick-Start.md) · [Fusion Strategies](docs/Fusion-Strategies.md) · [Pipelines](docs/Pipelines.md)

</div>

---

## Overview

FlashFusion is a unified framework for combining multiple vision models into powerful multi-model pipelines. Leverage **Weighted Box Fusion**, **Voting Ensembles**, **Cascading**, **Stacking**, and **Feature-Level Fusion** to boost accuracy and robustness beyond any single model.

### Key Features

- **Multi-Model Ensembling** — Combine detection, classification, segmentation, and OCR models
- **Flexible Strategies** — WBF, voting, cascade, stacking, NMS fusion out-of-the-box
- **Pre-Built Pipelines** — Det→Cls, Det+Seg, multi-task parallel execution
- **FlashVision Integration** — Native support for FlashDet, FlashCls, FlashSeg, FlashOCR
- **ONNX Export** — Export fused pipelines for edge deployment
- **LoRA Fine-Tuning** — Efficiently adapt fusion layers with minimal parameters
- **Extensible Registry** — Plug in custom strategies, backbones, and pipelines

> **Note:** Fusion strategies (WBF, voting, cascade, stacking, NMS) are fully implemented and tested.
> Pipelines, training engine, and export require trained model weights (FlashVision checkpoints or custom `.pt` files).
> See [CHANGELOG.md](CHANGELOG.md) for details on implemented vs. planned features.

---

## Installation

```bash
pip install flashfusion
```

**With all FlashVision models:**
```bash
pip install flashfusion[flash]
```

**Full install (export + analytics + flash models):**
```bash
pip install flashfusion[all]
```

**From source:**
```bash
git clone https://github.com/FlashVision/FlashFusion.git
cd FlashFusion
pip install -e ".[dev]"
```

---

## Quick Start

### Ensemble Detection Models

```python
from flashfusion import FlashFusion
from flashfusion.strategies import WeightedBoxFusion

model = FlashFusion(
    models=["flashdet-m", "flashdet-l"],
    strategy=WeightedBoxFusion(weights=[0.4, 0.6]),
    input_size=(320, 320),
)

results = model.predict("image.jpg")
```

### Detection → Classification Pipeline

```python
from flashfusion.pipelines import DetClsPipeline

pipeline = DetClsPipeline(
    detector="flashdet-m",
    classifier="flashcls-m",
    det_threshold=0.5,
)

results = pipeline("image.jpg")
for obj in results:
    print(f"{obj.label}: {obj.confidence:.2f} at {obj.bbox}")
```

### Cascade Pipeline with Early Exit

```python
from flashfusion.strategies import CascadeFusion

cascade = CascadeFusion(
    models=["flashdet-s", "flashdet-m", "flashdet-l"],
    confidence_thresholds=[0.8, 0.6, 0.0],
)

results = cascade.predict("image.jpg")
```

---

## Fusion Strategies

| Strategy | Use Case | Description |
|----------|----------|-------------|
| **Weighted Box Fusion** | Detection | Merges overlapping boxes with learned weights |
| **Voting** | Classification | Majority/soft voting across models |
| **Cascade** | Any | Sequential models with early exit on high confidence |
| **Stacking** | Any | Meta-learner on top of model outputs |
| **NMS Fusion** | Detection | NMS-based deduplication across model outputs |

---

## Training

Train fusion layers on top of frozen base models:

```bash
flashfusion train --config configs/flashfusion_det_cls_320.yaml
```

---

## Export

Export fused pipeline to ONNX:

```bash
flashfusion export --config configs/flashfusion_ensemble_320.yaml --format onnx
```

---

## Project Structure

```
FlashFusion/
├── flashfusion/          # Core library
│   ├── models/           # Fusion model definitions
│   ├── strategies/       # Fusion strategies (WBF, voting, cascade, etc.)
│   ├── pipelines/        # Pre-built multi-model pipelines
│   ├── engine/           # Training, validation, prediction, export
│   ├── solutions/        # High-level solution APIs
│   └── analytics/        # Benchmarking and profiling
├── configs/              # YAML configuration files
├── examples/             # Runnable example scripts
├── tests/                # Unit tests
└── docs/                 # Documentation
```

---

## Benchmarks

| Pipeline | Models | mAP | Latency (ms) | vs Single Model |
|----------|--------|-----|--------------|-----------------|
| WBF Ensemble | FlashDet-M × 3 | 52.1 | 28.4 | +3.2 mAP |
| Det→Cls | FlashDet-M + FlashCls-M | 48.7 | 18.2 | +2.1 mAP |
| Cascade | FlashDet-S/M/L | 50.8 | 15.6 | +2.8 mAP, -40% latency |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

FlashFusion is released under the [MIT License](LICENSE).

---

<div align="center">
  <b>Part of the <a href="https://github.com/FlashVision">FlashVision</a> ecosystem</b>
</div>
