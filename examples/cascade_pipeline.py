"""Example: Cascade Pipeline

Demonstrates a cascade fusion pipeline where a fast detector proposes regions
and a refined detector/classifier processes them sequentially.

Usage:
    python examples/cascade_pipeline.py --source image.jpg
"""

import argparse

import torch
import torch.nn as nn

from flashfusion.models.fusion import FlashFusion
from flashfusion.strategies import CascadeFusion


class FastDetector(nn.Module):
    """Lightweight detector for the first cascade stage."""

    def __init__(self, num_detections: int = 20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(16, num_detections * 5)
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


class RefinedDetector(nn.Module):
    """Higher-capacity detector for the second cascade stage."""

    def __init__(self, num_classes: int = 80, num_detections: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, num_detections * (5 + num_classes))
        self.num_classes = num_classes
        self.num_detections = num_detections

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        feat = self.conv(x)
        out = self.head(feat).view(batch_size, self.num_detections, -1)
        boxes = torch.sigmoid(out[:, :, :4]) * x.shape[-1]
        scores = torch.sigmoid(out[:, :, 4])
        labels = out[:, :, 5:].argmax(dim=-1)
        return {"boxes": boxes, "scores": scores, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Cascade Pipeline Example")
    parser.add_argument("--source", type=str, default=None, help="Image path")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlashFusion: Cascade Pipeline Example")
    print("=" * 60)

    # Stage 1: Fast coarse detector
    fast_model = FastDetector(num_detections=20)
    # Stage 2: Refined detector
    refined_model = RefinedDetector(num_classes=80, num_detections=10)

    print(f"\n[1] Stage 1 — Fast detector: {sum(p.numel() for p in fast_model.parameters())} params")
    print(f"[2] Stage 2 — Refined detector: {sum(p.numel() for p in refined_model.parameters())} params")

    # Create cascade strategy
    cascade = CascadeFusion()
    print(f"[3] Strategy: CascadeFusion")

    # Build fusion model
    fusion = FlashFusion(
        models=[fast_model, refined_model],
        strategy=cascade,
        input_size=(320, 320),
        device=args.device,
    )
    print(f"[4] Cascade pipeline with {fusion.num_models} stages")

    # Run inference
    dummy_input = torch.randn(1, 3, 320, 320)
    print(f"\n[5] Running cascade inference...")

    with torch.no_grad():
        results = fusion(dummy_input)

    print(f"[6] Cascade output:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: shape={value.shape}")
        else:
            print(f"     {key}: {type(value).__name__}")

    print("\n" + "=" * 60)
    print("  Cascade pipeline inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
