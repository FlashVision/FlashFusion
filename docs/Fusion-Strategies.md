# Fusion Strategies

FlashFusion provides multiple fusion strategies for combining predictions from multiple models.

## Available Strategies

| Strategy | Key | Best For |
|----------|-----|----------|
| Weighted Box Fusion | `wbf` | Detection ensembles (best mAP) |
| Voting Ensemble | `voting` | Classification consensus |
| Cascade Fusion | `cascade` | Sequential refinement |
| Stacking Ensemble | `stacking` | Learned combination |
| NMS Fusion | `nms` | Fast duplicate removal |

## Weighted Box Fusion (WBF)

WBF merges overlapping boxes using confidence-weighted averaging. It produces more accurate localizations than NMS by averaging coordinates rather than selecting a single box.

```python
from flashfusion.strategies import WeightedBoxFusion

wbf = WeightedBoxFusion(
    weights=[0.6, 0.4],       # per-model importance
    iou_threshold=0.55,        # clustering threshold
    skip_box_threshold=0.01,   # minimum confidence
    conf_type="avg",           # 'avg', 'max', 'box_and_model_avg'
)

fused = wbf.fuse(model_outputs)
```

### How WBF Works

1. Collect all boxes from all models with associated weights
2. Sort by weighted confidence
3. Cluster overlapping boxes (IoU > threshold) with same class
4. Compute weighted average coordinates for each cluster
5. Return cluster centers as fused detections

## Voting Ensemble

Majority voting for classification or box selection based on agreement across models.

```python
from flashfusion.strategies import VotingEnsemble

voting = VotingEnsemble()
fused = voting.fuse(model_outputs)
```

## Cascade Fusion

Sequential pipeline where each stage refines the previous output.

```python
from flashfusion.strategies import CascadeFusion

cascade = CascadeFusion()
# Stage 1 proposes, Stage 2 refines
fused = cascade.fuse([coarse_output, refined_output])
```

## NMS Fusion

Standard Non-Maximum Suppression applied across all model predictions.

```python
from flashfusion.strategies import NMSFusion

nms = NMSFusion()
fused = nms.fuse(model_outputs)
```

## Stacking Ensemble

Learned meta-model that combines base model outputs.

```python
from flashfusion.strategies import StackingEnsemble

stacking = StackingEnsemble()
fused = stacking.fuse(model_outputs)
```

## Strategy Factory

Use `get_strategy()` to instantiate by name:

```python
from flashfusion.strategies import get_strategy

strategy = get_strategy("wbf", weights=[0.5, 0.5], iou_threshold=0.6)
```

## Custom Strategies

Register your own strategy via the registry:

```python
from flashfusion.registry import STRATEGIES

@STRATEGIES.register("my_fusion")
class MyFusion:
    def fuse(self, predictions, weights=None):
        # Your fusion logic here
        ...
```

## Strategy Comparison

Use the benchmark example to compare strategies:

```bash
python examples/benchmark_fusion.py --device cuda
```
