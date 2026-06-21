"""Example: Detection + Classification Fusion

Demonstrates how to combine a detection model with a classification model
using FlashFusion's weighted fusion strategy.

Usage:
    python examples/fuse_det_cls.py --det-model weights/flashdet_m.pt \
                                     --cls-model weights/flashcls_m.pt \
                                     --source image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flashfusion.models.fusion import FlashFusion
from flashfusion.strategies import WeightedBoxFusion


class MockDetector(nn.Module):
    """Mock detection model for demonstration."""

    def __init__(self, num_classes: int = 80, num_detections: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, num_detections * (4 + 1 + num_classes))
        self.num_classes = num_classes
        self.num_detections = num_detections

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        feat = self.backbone(x).flatten(1)
        out = self.head(feat).view(batch_size, self.num_detections, -1)
        boxes = torch.sigmoid(out[:, :, :4]) * x.shape[-1]
        scores = torch.sigmoid(out[:, :, 4])
        labels = out[:, :, 5:].argmax(dim=-1)
        return {"boxes": boxes, "scores": scores, "labels": labels}


class MockClassifier(nn.Module):
    """Mock classification model for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return {
            "class_logits": logits,
            "class_probs": torch.softmax(logits, dim=-1),
            "boxes": torch.zeros(x.shape[0], 0, 4),
            "scores": torch.zeros(x.shape[0], 0),
            "labels": torch.zeros(x.shape[0], 0, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Detection + Classification Fusion Example")
    parser.add_argument("--det-model", type=str, default=None, help="Detection model path")
    parser.add_argument("--cls-model", type=str, default=None, help="Classification model path")
    parser.add_argument("--source", type=str, default=None, help="Image path")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlashFusion: Detection + Classification Fusion Example")
    print("=" * 60)

    # Create models (use mock models for demonstration)
    det_model = MockDetector(num_classes=80, num_detections=10)
    cls_model = MockClassifier(num_classes=10)
    print(f"\n[1] Created detection model: {det_model.__class__.__name__}")
    print(f"[2] Created classification model: {cls_model.__class__.__name__}")

    # Create fusion strategy
    strategy = WeightedBoxFusion(weights=[0.6, 0.4], iou_threshold=0.55)
    print(f"[3] Fusion strategy: {strategy}")

    # Build FlashFusion model
    fusion = FlashFusion(
        models=[det_model, cls_model],
        strategy=strategy,
        input_size=(320, 320),
        weights=[0.6, 0.4],
        device=args.device,
    )
    print(f"[4] FlashFusion model created with {fusion.num_models} models")

    # Run inference on dummy image
    dummy_image = torch.randn(1, 3, 320, 320)
    print(f"\n[5] Running inference on input shape: {dummy_image.shape}")

    with torch.no_grad():
        results = fusion(dummy_image)

    print(f"[6] Fusion results:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"     {key}: {type(value)}")

    print("\n" + "=" * 60)
    print("  Detection + Classification fusion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
