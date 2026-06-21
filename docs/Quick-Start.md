# Quick Start

Get up and running with FlashFusion in 5 minutes.

## 1. Basic Ensemble

```python
from flashfusion import FlashFusion
from flashfusion.strategies import WeightedBoxFusion

# Create fusion model with two detectors
model = FlashFusion(
    models=["weights/flashdet_s.pt", "weights/flashdet_m.pt"],
    strategy=WeightedBoxFusion(weights=[0.4, 0.6]),
    input_size=(320, 320),
)

# Run prediction
results = model.predict("image.jpg")
```

## 2. Using EnsembleDetector (High-Level API)

```python
from flashfusion import EnsembleDetector

detector = EnsembleDetector(
    models=["model_a.pt", "model_b.pt", "model_c.pt"],
    strategy="wbf",
    weights=[0.5, 0.3, 0.2],
)

results = detector.detect("image.jpg")
for det in results:
    print(f"{det['label']}: {det['score']:.2f} at {det['bbox']}")
```

## 3. CLI Usage

```bash
# Show version and system info
flashfusion version

# Run fusion prediction
flashfusion predict --config configs/flashfusion_ensemble_320.yaml --source image.jpg

# Direct multi-model fusion
flashfusion fuse --models model1.pt model2.pt --strategy wbf --source image.jpg

# Train fusion layers
flashfusion train --config configs/flashfusion_det_cls_320.yaml

# Export to ONNX
flashfusion export --config configs/flashfusion_ensemble_320.yaml --format onnx
```

## 4. Compare Fusion Strategies

```python
from flashfusion.strategies import get_strategy

for name in ["wbf", "voting", "nms", "cascade"]:
    strategy = get_strategy(name)
    print(f"{name}: {strategy}")
```

## 5. Benchmark Performance

```python
from flashfusion.analytics import Benchmark

bench = Benchmark("weights/fusion.pt", device="cuda")
results = bench.run()
print(f"FPS: {results['fps']:.1f}")
print(f"Latency: {results['latency_ms']:.2f} ms")
print(f"Parameters: {results['params']:,}")
```

## 6. Multi-Model Analysis

```python
from flashfusion import MultiModelAnalyzer

analyzer = MultiModelAnalyzer(
    models=["model_a.pt", "model_b.pt", "model_c.pt"],
    device="cuda",
)

report = analyzer.analyze("image.jpg")
print(f"Agreement: {report['agreement_score']:.2%}")
print(f"Total detections: {report['total_detections']}")
```

## Next Steps

- [Models](Models.md) — Learn about the FlashFusion architecture
- [Fusion Strategies](Fusion-Strategies.md) — Deep dive into WBF, Voting, Cascade
- [Training](Training.md) — Train fusion layers on your data
- [Pipelines](Pipelines.md) — Use pre-built multi-task pipelines
