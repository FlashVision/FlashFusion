"""FlashFusion Benchmark — Measure throughput, latency, and parameter counts.

Provides a simple API for benchmarking fusion models and comparing different
fusion strategies on the same inputs.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class Benchmark:
    """Benchmark fusion model performance.

    Measures FPS, latency, parameter count, and memory usage for a given
    model or configuration.

    Args:
        model_path: Path to model weights or a config YAML.
        device: Target device for benchmarking.
        warmup: Number of warmup iterations before timing.
        iterations: Number of timed iterations.
        input_size: Input image size as (height, width).

    Example:
        >>> bench = Benchmark("weights/fusion_model.pt", device="cuda")
        >>> results = bench.run()
        >>> print(f"FPS: {results['fps']:.1f}, Latency: {results['latency_ms']:.2f}ms")
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: str = "auto",
        warmup: int = 10,
        iterations: int = 100,
        input_size: tuple = (320, 320),
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = self._resolve_device(device)
        self.warmup = warmup
        self.iterations = iterations
        self.input_size = input_size
        self._model: Optional[nn.Module] = None

    def run(self, model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Run the benchmark and return performance metrics.

        Args:
            model: Optional nn.Module to benchmark directly.
                   If None, loads from model_path.

        Returns:
            Dictionary with keys:
                - fps: frames per second (float)
                - latency_ms: average latency in milliseconds (float)
                - latency_std_ms: standard deviation of latency (float)
                - params: total parameter count (int)
                - trainable_params: trainable parameter count (int)
                - flops_estimate: estimated FLOPs (int or None)
                - memory_mb: peak GPU memory in MB (float or None)
        """
        target_model = model or self._load_model()
        if target_model is None:
            return self._empty_results("No model available for benchmarking")

        target_model.eval()
        target_model.to(self.device)

        dummy_input = torch.randn(1, 3, self.input_size[0], self.input_size[1], device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                target_model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        with torch.no_grad():
            for _ in range(self.iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                target_model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

        latencies_ms = np.array(latencies) * 1000.0
        avg_latency = float(np.mean(latencies_ms))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        total_params = sum(p.numel() for p in target_model.parameters())
        trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)

        memory_mb = None
        if self.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(self.device)

        return {
            "fps": fps,
            "latency_ms": avg_latency,
            "latency_std_ms": float(np.std(latencies_ms)),
            "params": total_params,
            "trainable_params": trainable_params,
            "flops_estimate": self._estimate_flops(target_model, dummy_input),
            "memory_mb": memory_mb,
        }

    def compare_strategies(
        self,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different fusion strategies.

        Args:
            strategies: List of strategy names to compare.
                        Defaults to ['wbf', 'voting', 'nms'].

        Returns:
            Dictionary mapping strategy name to benchmark results.
        """
        from flashfusion.strategies import get_strategy
        from flashfusion.models.fusion import FlashFusion

        if strategies is None:
            strategies = ["wbf", "voting", "nms"]

        results = {}
        for strategy_name in strategies:
            try:
                strategy = get_strategy(strategy_name)
                if self.model_path and self.model_path.exists():
                    model = FlashFusion(
                        models=[str(self.model_path)],
                        strategy=strategy,
                        input_size=self.input_size,
                        device=str(self.device),
                    )
                    results[strategy_name] = self.run(model)
                else:
                    results[strategy_name] = self._empty_results(
                        f"Model path not available for strategy '{strategy_name}'"
                    )
            except Exception as e:
                results[strategy_name] = self._empty_results(str(e))

        return results

    def _load_model(self) -> Optional[nn.Module]:
        """Load model from model_path."""
        if self.model_path is None or not self.model_path.exists():
            return None

        if self.model_path.suffix in (".pt", ".pth"):
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            if isinstance(checkpoint, nn.Module):
                return checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                return checkpoint["model"]

        return None

    @staticmethod
    def _estimate_flops(model: nn.Module, dummy_input: torch.Tensor) -> Optional[int]:
        """Estimate FLOPs using hook-based counting or thop if available."""
        try:
            from thop import profile

            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            return int(flops)
        except ImportError:
            pass

        total_flops = 0
        hooks = []

        def _count_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Linear):
                in_feat = module.in_features
                out_feat = module.out_features
                batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
                total_flops += batch_size * in_feat * out_feat * 2
            elif isinstance(module, nn.Conv2d):
                batch_size = input[0].shape[0]
                out_h, out_w = output.shape[2], output.shape[3]
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
                total_flops += batch_size * module.out_channels * out_h * out_w * kernel_ops * 2
            elif isinstance(module, nn.BatchNorm2d):
                total_flops += input[0].numel() * 2

        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                hooks.append(m.register_forward_hook(_count_hook))

        try:
            with torch.no_grad():
                model(dummy_input)
        finally:
            for h in hooks:
                h.remove()

        return total_flops if total_flops > 0 else None

    @staticmethod
    def _empty_results(reason: str) -> Dict[str, Any]:
        """Return empty results with an error reason."""
        return {
            "fps": 0.0,
            "latency_ms": 0.0,
            "latency_std_ms": 0.0,
            "params": 0,
            "trainable_params": 0,
            "flops_estimate": None,
            "memory_mb": None,
            "error": reason,
        }

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return f"Benchmark(model_path={self.model_path}, device={self.device}, iterations={self.iterations})"
