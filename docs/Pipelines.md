# Pipelines

FlashFusion provides pre-built multi-task pipelines that combine detection, classification, and segmentation models.

## Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `DetClsPipeline` | Detection + Classification |
| `DetSegPipeline` | Detection + Segmentation |
| `MultiTaskPipeline` | Detection + Classification + Segmentation |

## Detection + Classification Pipeline

Combines a detector with a classifier for fine-grained recognition:

```python
from flashfusion.pipelines import DetClsPipeline

pipeline = DetClsPipeline(
    det_model="flashdet_m.pt",
    cls_model="flashcls_m.pt",
    device="cuda",
)

results = pipeline.run("image.jpg")
# Returns: detections with refined class labels
```

## Detection + Segmentation Pipeline

Combines a detector with a segmentation model:

```python
from flashfusion.pipelines import DetSegPipeline

pipeline = DetSegPipeline(
    det_model="flashdet_m.pt",
    seg_model="flashseg_m.pt",
    device="cuda",
)

results = pipeline.run("image.jpg")
# Returns: detections + per-pixel segmentation masks
```

## Multi-Task Pipeline

Full multi-task fusion combining all model types:

```python
from flashfusion.pipelines import MultiTaskPipeline

pipeline = MultiTaskPipeline(
    models={
        "detection": "flashdet_m.pt",
        "classification": "flashcls_m.pt",
        "segmentation": "flashseg_m.pt",
    },
    strategy="wbf",
    device="cuda",
)

results = pipeline.run("image.jpg")
```

## Configuration-Based Pipelines

Use YAML configs to define pipelines:

```bash
flashfusion predict --config configs/flashfusion_det_cls_320.yaml --source image.jpg
flashfusion predict --config configs/flashfusion_det_seg_320.yaml --source image.jpg
```

## Custom Pipelines

Register custom pipelines via the registry:

```python
from flashfusion.registry import PIPELINES

@PIPELINES.register("my_pipeline")
class MyPipeline:
    def __init__(self, **kwargs):
        ...

    def run(self, source):
        ...
```
