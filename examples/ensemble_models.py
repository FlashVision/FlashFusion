"""Example: Ensemble Multiple Detection Models

Demonstrates how to ensemble multiple detection models of different sizes
using Weighted Box Fusion for improved accuracy.

Usage:
    python examples/ensemble_models.py --source image.jpg
"""

import argparse
import time

import torch
import torch.nn as nn

from flashfusion.models.fusion import FlashFusion
from flashfusion.solutions.ensemble_detector import EnsembleDetector
from flashfusion.strategies import WeightedBoxFusion, get_strategy


class DetectorVariant(nn.Module):
    """Parameterized detector simulating different model sizes."""

    def __init__(self, channels: int = 16, num_detections: int = 10, name: str = "det"):
        super().__init__()
        self.name = name
        self.conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(channels * 2, num_detections * 5)
        self.num_detections = num_detections

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        feat = self.conv(x)
        out = self.head(feat).view(batch_size, self.num_detections, 5)
        boxes = torch.sigmoid(out[:, :, :4]) * x.shape[-1]
        scores = torch.sigmoid(out[:, :, 4])
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": torch.zeros(batch_size, self.num_detections, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Model Ensemble Example")
    parser.add_argument("--source", type=str, default=None, help="Image path")
    parser.add_argument("--strategy", type=str, default="wbf", help="Fusion strategy")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlashFusion: Multi-Model Ensemble Example")
    print("=" * 60)

    # Create model variants (small, medium, large)
    models = [
        DetectorVariant(channels=8, num_detections=15, name="small"),
        DetectorVariant(channels=16, num_detections=12, name="medium"),
        DetectorVariant(channels=32, num_detections=10, name="large"),
    ]

    print("\n[Models]")
    for model in models:
        params = sum(p.numel() for p in model.parameters())
        print(f"  • {model.name}: {params:,} parameters")

    # Create ensemble with WBF
    weights = [0.25, 0.35, 0.40]
    strategy = get_strategy(args.strategy, weights=weights, iou_threshold=0.55)

    fusion = FlashFusion(
        models=models,
        strategy=strategy,
        input_size=(320, 320),
        weights=weights,
        device=args.device,
    )
    print(f"\n[Ensemble] {fusion.num_models} models with '{args.strategy}' strategy")
    print(f"  Weights: {weights}")

    # Run ensemble inference
    dummy_input = torch.randn(1, 3, 320, 320)
    print(f"\n[Inference] Input shape: {dummy_input.shape}")

    start_time = time.perf_counter()
    with torch.no_grad():
        results = fusion(dummy_input)
    elapsed = (time.perf_counter() - start_time) * 1000

    print(f"  Inference time: {elapsed:.2f} ms")
    print(f"\n[Results]")
    if "boxes" in results:
        num_dets = results["boxes"].shape[-2] if results["boxes"].dim() > 1 else results["boxes"].shape[0]
        print(f"  Fused detections: {num_dets}")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Compare strategies
    print(f"\n[Strategy Comparison]")
    for strategy_name in ["wbf", "voting", "nms"]:
        try:
            strat = get_strategy(strategy_name, weights=weights)
            temp_fusion = FlashFusion(
                models=models, strategy=strat,
                input_size=(320, 320), weights=weights, device=args.device,
            )
            start = time.perf_counter()
            with torch.no_grad():
                res = temp_fusion(dummy_input)
            ms = (time.perf_counter() - start) * 1000
            n_boxes = res["boxes"].shape[-2] if res["boxes"].dim() > 1 else res["boxes"].shape[0]
            print(f"  {strategy_name:>8}: {n_boxes} detections in {ms:.2f} ms")
        except Exception as e:
            print(f"  {strategy_name:>8}: Error — {e}")

    print("\n" + "=" * 60)
    print("  Ensemble inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
