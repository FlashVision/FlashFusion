# Training

FlashFusion trains fusion layers (heads, necks, learned weights) while keeping base model backbones frozen.

## Quick Start

```bash
flashfusion train --config configs/flashfusion_det_cls_320.yaml
```

## Training Configuration

```yaml
# configs/flashfusion_det_cls_320.yaml
fusion:
  strategy: weighted_box_fusion
  input_size: [320, 320]
  models:
    - type: detection
      source: flashdet
      model_size: m
      frozen: true
    - type: classification
      source: flashcls
      model_size: m
      frozen: true

train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.0005
  scheduler: cosine
  warmup_epochs: 3
  save_dir: workspace/flashfusion_det_cls
```

## Programmatic Training

```python
from flashfusion import Trainer
from flashfusion.cfg import get_config

config = get_config("configs/flashfusion_det_cls_320.yaml")
trainer = Trainer(config, device="cuda")
results = trainer.train()
```

## Resume Training

```bash
flashfusion train --config configs/flashfusion_det_cls_320.yaml --resume workspace/flashfusion_det_cls/last.pt
```

## Dataset Format

FlashFusion expects the following directory structure:

```
dataset_root/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_100.jpg
│       └── ...
├── annotations/
│   ├── train/
│   │   ├── img_001.json
│   │   └── ...
│   └── val/
│       ├── img_100.json
│       └── ...
└── masks/  (optional)
    ├── train/
    └── val/
```

Each annotation JSON:

```json
{
    "boxes": [[x1, y1, x2, y2], ...],
    "labels": [0, 1, ...],
    "class_label": 3
}
```

## Loss Functions

FlashFusion uses a multi-task loss:

```python
from flashfusion.losses import FusionLoss

criterion = FusionLoss(
    det_weight=1.0,          # Detection loss weight
    cls_weight=0.5,          # Classification loss weight
    consistency_weight=0.1,  # Inter-model consistency
)
```

## Checkpoints

Checkpoints are saved automatically:
- `workspace/<name>/last.pt` — Latest checkpoint
- `workspace/<name>/best.pt` — Best validation metric

## LoRA Fine-Tuning

For efficient training with minimal parameters:

```python
from flashfusion.models.lora import apply_lora

model = apply_lora(model, rank=8, alpha=16)
# Only LoRA parameters are trained
```

## Monitoring

Use the logger and metrics utilities:

```python
from flashfusion.utils import setup_logger, AverageMeter

logger = setup_logger("train", log_file="train.log")
loss_meter = AverageMeter("loss")
```
