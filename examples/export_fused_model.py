"""Example: Export Fused Model to ONNX

Demonstrates exporting a FlashFusion ensemble model to ONNX format
for deployment on edge devices or inference servers.

Usage:
    python examples/export_fused_model.py --output fusion_model.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class ExportableDetector(nn.Module):
    """A simple detector suitable for ONNX export."""

    def __init__(self, channels: int = 16, num_detections: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.box_head = nn.Linear(channels * 2, num_detections * 4)
        self.score_head = nn.Linear(channels * 2, num_detections)
        self.num_detections = num_detections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        boxes = self.box_head(feat).view(-1, self.num_detections, 4)
        scores = self.score_head(feat).view(-1, self.num_detections, 1)
        return torch.cat([boxes, scores], dim=-1)


class FusedEnsemble(nn.Module):
    """Wrapper that combines multiple models for ONNX-compatible export.

    Concatenates outputs from multiple detectors and applies simple
    score-weighted averaging suitable for static graph export.
    """

    def __init__(self, models: list, weights: list):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, model in enumerate(self.models):
            out = model(x)
            out[:, :, 4] *= self.weights[i]
            outputs.append(out)

        combined = torch.cat(outputs, dim=1)

        # Sort by score descending and take top-K
        scores = combined[:, :, 4]
        top_k = min(50, combined.shape[1])
        _, indices = scores.topk(top_k, dim=1)
        indices = indices.unsqueeze(-1).expand(-1, -1, 5)
        result = torch.gather(combined, 1, indices)

        return result


def main():
    parser = argparse.ArgumentParser(description="Export Fused Model to ONNX")
    parser.add_argument("--output", type=str, default="workspace/fusion_model.onnx", help="Output ONNX path")
    parser.add_argument("--num-models", type=int, default=3, help="Number of models to ensemble")
    parser.add_argument("--input-size", type=int, default=320, help="Input image size")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlashFusion: Export to ONNX")
    print("=" * 60)

    # Create models
    models = [ExportableDetector(channels=16) for _ in range(args.num_models)]
    weights = [1.0 / args.num_models] * args.num_models

    print(f"\n[1] Created {args.num_models} detector models")
    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f"    Total parameters: {total_params:,}")

    # Wrap in exportable ensemble
    ensemble = FusedEnsemble(models, weights)
    ensemble.eval()
    print(f"[2] Created exportable FusedEnsemble wrapper")

    # Prepare dummy input
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    print(f"[3] Input shape: {dummy_input.shape}")

    # Verify forward pass works
    with torch.no_grad():
        test_output = ensemble(dummy_input)
    print(f"[4] Test forward pass: output shape = {test_output.shape}")

    # Export to ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            ensemble,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["detections"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "detections": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[5] Exported ONNX model to: {output_path}")
        print(f"    File size: {output_path.stat().st_size / 1024:.1f} KB")

        # Optionally simplify
        if args.simplify:
            try:
                import onnx
                from onnxsim import simplify

                model_onnx = onnx.load(str(output_path))
                model_simplified, check = simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print(f"[6] Simplified ONNX model saved")
                else:
                    print(f"[6] Simplification check failed, keeping original")
            except ImportError:
                print(f"[6] onnxsim not installed, skipping simplification")
                print(f"    Install with: pip install onnxsim")

    except Exception as e:
        print(f"[5] Export failed: {e}")
        print(f"    Make sure 'onnx' is installed: pip install onnx")

    print("\n" + "=" * 60)
    print("  Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
