"""FlashFusion Profiler — Per-layer timing and resource profiling.

Profiles each layer/module in a fusion model to identify bottlenecks
and optimize inference latency.
"""

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class Profiler:
    """Profile per-layer inference timing for fusion models.

    Hooks into each module to measure forward-pass duration and reports
    a sorted breakdown of time spent in each layer.

    Args:
        model_path: Path to model weights file.
        device: Target device for profiling.
        input_size: Input tensor size as (height, width).

    Example:
        >>> profiler = Profiler("weights/fusion.pt")
        >>> report = profiler.run()
        >>> for layer, ms in report["layers"][:5]:
        ...     print(f"{layer}: {ms:.3f} ms")
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: str = "auto",
        input_size: tuple = (320, 320),
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = self._resolve_device(device)
        self.input_size = input_size
        self._timings: OrderedDict = OrderedDict()
        self._hooks: List[Any] = []

    def run(self, model: Optional[nn.Module] = None, iterations: int = 50) -> Dict[str, Any]:
        """Profile the model and print per-layer timing report.

        Args:
            model: Optional model to profile. If None, loads from model_path.
            iterations: Number of forward passes to average over.

        Returns:
            Dictionary with:
                - total_ms: total inference time in milliseconds
                - layers: list of (layer_name, time_ms) tuples sorted by time descending
                - num_layers: number of profiled layers
        """
        target_model = model or self._load_model()
        if target_model is None:
            print("[Profiler] No model available for profiling.")
            return {"total_ms": 0.0, "layers": [], "num_layers": 0}

        target_model.eval()
        target_model.to(self.device)

        self._timings.clear()
        self._register_hooks(target_model)

        dummy_input = torch.randn(1, 3, self.input_size[0], self.input_size[1], device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                target_model(dummy_input)

        # Reset timings after warmup
        self._timings.clear()

        with torch.no_grad():
            for _ in range(iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                target_model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        self._remove_hooks()

        # Compute average per-layer timing
        layer_timings = []
        for name, times in self._timings.items():
            avg_ms = (sum(times) / len(times)) * 1000.0 if times else 0.0
            layer_timings.append((name, avg_ms))

        layer_timings.sort(key=lambda x: -x[1])
        total_ms = sum(ms for _, ms in layer_timings)

        # Print report
        print(f"\n{'=' * 60}")
        print(f"  FlashFusion Profiler Report ({iterations} iterations)")
        print(f"{'=' * 60}")
        print(f"  {'Layer':<40} {'Time (ms)':>10} {'%':>6}")
        print(f"  {'-' * 40} {'-' * 10} {'-' * 6}")

        for name, ms in layer_timings[:20]:
            pct = (ms / total_ms * 100) if total_ms > 0 else 0
            display_name = name[:40] if len(name) > 40 else name
            print(f"  {display_name:<40} {ms:>10.3f} {pct:>5.1f}%")

        if len(layer_timings) > 20:
            print(f"  ... and {len(layer_timings) - 20} more layers")

        print(f"  {'-' * 40} {'-' * 10} {'-' * 6}")
        print(f"  {'TOTAL':<40} {total_ms:>10.3f}")
        print(f"{'=' * 60}\n")

        return {
            "total_ms": total_ms,
            "layers": layer_timings,
            "num_layers": len(layer_timings),
        }

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward pre-hooks and post-hooks on all leaf modules."""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                layer_name = name or module.__class__.__name__
                pre_hook = module.register_forward_pre_hook(self._make_pre_hook(layer_name))
                post_hook = module.register_forward_hook(self._make_post_hook(layer_name))
                self._hooks.append(pre_hook)
                self._hooks.append(post_hook)

    def _make_pre_hook(self, layer_name: str):
        """Create a pre-forward hook that records start time."""

        def hook_fn(module, input):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            if not hasattr(module, "_profiler_start_time"):
                module._profiler_start_time = {}
            module._profiler_start_time[layer_name] = time.perf_counter()

        return hook_fn

    def _make_post_hook(self, layer_name: str):
        """Create a post-forward hook that records elapsed time."""

        def hook_fn(module, input, output):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = getattr(module, "_profiler_start_time", {}).get(layer_name, None)
            if start is not None:
                elapsed = time.perf_counter() - start
                if layer_name not in self._timings:
                    self._timings[layer_name] = []
                self._timings[layer_name].append(elapsed)

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _load_model(self) -> Optional[nn.Module]:
        """Load model from model_path."""
        if self.model_path is None or not self.model_path.exists():
            return None

        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            return checkpoint["model"]
        return None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return f"Profiler(model_path={self.model_path}, device={self.device})"
