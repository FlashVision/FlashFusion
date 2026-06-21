"""FlashFusion analytics — benchmarking, profiling, and visualization."""

from flashfusion.analytics.benchmark import Benchmark
from flashfusion.analytics.profiler import Profiler
from flashfusion.analytics.plots import (
    plot_training_curves,
    plot_fusion_comparison,
    plot_strategy_comparison,
)

__all__ = [
    "Benchmark",
    "Profiler",
    "plot_training_curves",
    "plot_fusion_comparison",
    "plot_strategy_comparison",
]
