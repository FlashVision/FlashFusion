# FAQ

## General

### What is FlashFusion?

FlashFusion is a multi-model vision fusion framework. It combines predictions from multiple vision models (detection, classification, segmentation) using configurable strategies like Weighted Box Fusion, Voting, and Cascade.

### How is FlashFusion different from a simple ensemble?

FlashFusion provides:
- Multiple fusion strategies beyond simple NMS
- Multi-task support (det + cls + seg in one pipeline)
- Cascade pipelines for sequential refinement
- Consistency losses for training agreement between models
- Built-in benchmarking and profiling tools

### Which fusion strategy should I use?

| Scenario | Recommended Strategy |
|----------|---------------------|
| Multiple detectors, maximize mAP | WBF |
| Speed-critical with multiple detectors | NMS |
| Classification ensemble | Voting |
| Progressive refinement (coarse → fine) | Cascade |
| Learned combination with training data | Stacking |

## Performance

### What FPS can I expect?

FPS depends on:
- Number of models in the ensemble
- Model sizes (S/M/L)
- Input resolution
- Hardware (GPU vs CPU)

Typical ranges on GPU:
- 2-model ensemble: 30-60 FPS
- 3-model ensemble: 20-40 FPS
- Cascade (2-stage): 25-50 FPS

### How much does fusion improve accuracy?

Typical improvements over single-model:
- WBF ensemble (3 models): +2-5% mAP
- Cascade refinement: +1-3% mAP
- Multi-task fusion: task-dependent

## Training

### What gets trained in FlashFusion?

Base model backbones are frozen. Only fusion components are trained:
- Fusion heads/necks
- Learned strategy weights
- LoRA adapters (if enabled)

### How much data do I need?

Since base models are frozen, fusion training is data-efficient:
- Minimum: ~500 annotated images
- Recommended: 2,000+ images
- Fine-tuning converges in 20-50 epochs typically

### Can I use models from different frameworks?

Yes, as long as they're wrapped as PyTorch `nn.Module` instances with a forward method returning the expected dictionary format.

## Deployment

### Can I export to ONNX?

Yes:

```bash
flashfusion export --config configs/flashfusion_ensemble_320.yaml --format onnx
```

Or use the export example:

```bash
python examples/export_fused_model.py --output model.onnx
```

### Does FlashFusion work on edge devices?

Yes, with optimizations:
- Use smaller base models (S variant)
- Reduce input resolution (160x160 or 224x224)
- Export to ONNX + TensorRT
- Use 2-model ensemble instead of 3+

## Troubleshooting

### Import errors

Run `flashfusion check` to verify all dependencies. Missing packages can be installed with:

```bash
pip install -e ".[all]"
```

### Out of memory

- Reduce batch size
- Use smaller input resolution
- Use fewer models in the ensemble
- Enable mixed precision (FP16)

### Fusion results worse than single model

- Check model weights are correct
- Tune `iou_threshold` for your data
- Ensure models are complementary (different architectures/sizes)
- Verify confidence thresholds
