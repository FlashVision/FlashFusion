# Models

## FlashFusion Model

The core `FlashFusion` class wraps multiple vision models and applies a fusion strategy to combine their outputs.

### Architecture

```
Input Image
    │
    ├── Model A (e.g., FlashDet-S) ──┐
    ├── Model B (e.g., FlashDet-M) ──┼── Fusion Strategy ── Fused Output
    └── Model C (e.g., FlashDet-L) ──┘
```

### Creating a FlashFusion Model

```python
from flashfusion import FlashFusion
from flashfusion.strategies import WeightedBoxFusion

# From nn.Module instances
model = FlashFusion(
    models=[detector_a, detector_b],
    strategy=WeightedBoxFusion(weights=[0.6, 0.4]),
    input_size=(320, 320),
)

# From model file paths
model = FlashFusion.from_models(
    model_paths=["flashdet_s.pt", "flashdet_m.pt", "flashdet_l.pt"],
    strategy="wbf",
    weights=[0.25, 0.35, 0.40],
)
```

### Forward Pass

```python
import torch

x = torch.randn(1, 3, 320, 320)
output = model(x)
# output: {"boxes": Tensor, "scores": Tensor, "labels": Tensor}
```

## LoRA / QLoRA Adapters

FlashFusion supports parameter-efficient fine-tuning via LoRA:

```python
from flashfusion.models.lora import apply_lora, merge_lora_weights

# Apply LoRA to fusion layers
model = apply_lora(model, rank=8, alpha=16)

# After training, merge weights for inference
model = merge_lora_weights(model)
```

## Model Properties

| Property | Description |
|----------|-------------|
| `num_models` | Number of models in the fusion |
| `model_names` | List of model name strings |
| `input_size` | Input resolution (H, W) |
| `device` | Current device |
| `weights` | Per-model fusion weights |

## Supported Base Models

FlashFusion can wrap any `nn.Module` that returns a dictionary with detection outputs:

- FlashDet (S/M/L/X)
- FlashCls
- FlashSeg
- FlashOCR
- Any custom PyTorch model
