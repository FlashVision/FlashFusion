"""Example: Benchmark Different Fusion Strategies

Demonstrates how to benchmark and compare performance of different fusion
strategies using FlashFusion's analytics module.

Usage:
    python examples/benchmark_fusion.py --device cpu
"""

import argparse
import time

import torch
import torch.nn as nn

from flashfusion.models.fusion import FlashFusion
from flashfusion.strategies import get_strategy


class BenchmarkModel(nn.Module):
    """Model used for benchmarking fusion overhead."""

    def __init__(self, channels: int = 16, num_detections: int = 10):
        super().__init__()
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


def benchmark_strategy(
    fusion_model: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Benchmark a fusion model's throughput and latency."""
    fusion_model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            fusion_model(input_tensor)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            fusion_model(input_tensor)
            latencies.append(time.perf_counter() - start)

    latencies_ms = [l * 1000 for l in latencies]
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    return {
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "fps": fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fusion Strategies")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--iterations", type=int, default=50, help="Timing iterations")
    parser.add_argument("--num-models", type=int, default=3, help="Number of models in ensemble")
    args = parser.parse_args()

    print("=" * 60)
    print("  FlashFusion: Strategy Benchmark")
    print("=" * 60)

    # Create ensemble models
    models = [BenchmarkModel(channels=16) for _ in range(args.num_models)]
    total_params = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f"\n  Models: {args.num_models} x BenchmarkModel")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {args.device}")
    print(f"  Iterations: {args.iterations}")

    dummy_input = torch.randn(1, 3, 320, 320)
    strategies_to_test = ["wbf", "voting", "nms", "cascade"]
    weights = [1.0 / args.num_models] * args.num_models

    print(f"\n  {'Strategy':<15} {'FPS':>8} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    results = {}
    for strategy_name in strategies_to_test:
        try:
            strategy = get_strategy(strategy_name, weights=weights)
            fusion = FlashFusion(
                models=models,
                strategy=strategy,
                input_size=(320, 320),
                weights=weights,
                device=args.device,
            )

            metrics = benchmark_strategy(fusion, dummy_input, iterations=args.iterations)
            results[strategy_name] = metrics

            print(
                f"  {strategy_name:<15} "
                f"{metrics['fps']:>8.1f} "
                f"{metrics['avg_latency_ms']:>10.2f} "
                f"{metrics['min_latency_ms']:>10.2f} "
                f"{metrics['max_latency_ms']:>10.2f}"
            )
        except Exception as e:
            print(f"  {strategy_name:<15} {'ERROR':>8} — {e}")

    # Summary
    if results:
        fastest = max(results.items(), key=lambda x: x[1]["fps"])
        print(f"\n  Fastest strategy: {fastest[0]} ({fastest[1]['fps']:.1f} FPS)")

    print("\n" + "=" * 60)
    print("  Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
