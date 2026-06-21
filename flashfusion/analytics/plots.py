"""FlashFusion plotting utilities for training curves and strategy comparisons.

Provides publication-ready visualizations for fusion model training, strategy
benchmarking, and multi-model comparison results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def plot_training_curves(
    log_path: Union[str, Path, None] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "FlashFusion Training Curves",
) -> None:
    """Plot training loss and validation metric curves.

    Args:
        log_path: Path to a CSV/JSON training log file. If provided, metrics
                  are loaded from the file.
        metrics: Dictionary mapping metric name to list of per-epoch values.
                 Example: {"train_loss": [...], "val_map": [...]}
        save_path: If provided, saves the figure to this path. Otherwise shows interactively.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if metrics is None and log_path is not None:
        metrics = _load_training_log(log_path)

    if not metrics:
        print("[plot_training_curves] No metrics data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    loss_keys = [k for k in metrics if "loss" in k.lower()]
    metric_keys = [k for k in metrics if "loss" not in k.lower()]

    ax_loss = axes[0]
    for key in loss_keys:
        epochs = range(1, len(metrics[key]) + 1)
        ax_loss.plot(epochs, metrics[key], label=key, linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_metric = axes[1]
    for key in metric_keys:
        epochs = range(1, len(metrics[key]) + 1)
        ax_metric.plot(epochs, metrics[key], label=key, linewidth=2)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Metric")
    ax_metric.set_title("Validation Metrics")
    ax_metric.legend()
    ax_metric.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[plot_training_curves] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_fusion_comparison(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Fusion Strategy Comparison",
) -> None:
    """Plot comparison of different fusion strategies.

    Args:
        results: Dictionary mapping strategy name to metrics dict.
                 Each metrics dict should have 'fps', 'latency_ms', and optionally 'map'.
        save_path: If provided, saves figure. Otherwise shows interactively.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if not results:
        print("[plot_fusion_comparison] No results data provided.")
        return

    strategies = list(results.keys())
    fps_values = [results[s].get("fps", 0) for s in strategies]
    latency_values = [results[s].get("latency_ms", 0) for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    ax_fps = axes[0]
    bars = ax_fps.bar(strategies, fps_values, color=colors)
    ax_fps.set_ylabel("FPS")
    ax_fps.set_title("Throughput (FPS)")
    ax_fps.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, fps_values):
        ax_fps.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}", ha="center", va="bottom", fontsize=10,
        )

    ax_lat = axes[1]
    bars = ax_lat.bar(strategies, latency_values, color=colors)
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Latency (ms)")
    ax_lat.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, latency_values):
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2f}", ha="center", va="bottom", fontsize=10,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[plot_fusion_comparison] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_strategy_comparison(
    strategy_results: Dict[str, Dict[str, float]],
    metric_name: str = "mAP",
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Strategy Comparison",
) -> None:
    """Plot a radar/bar comparison of strategies across multiple metrics.

    Args:
        strategy_results: Mapping of strategy name to metrics dict.
            Example: {"wbf": {"mAP": 0.82, "fps": 45}, "voting": {"mAP": 0.79, "fps": 60}}
        metric_name: Primary metric to highlight.
        save_path: If provided, saves figure.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if not strategy_results:
        print("[plot_strategy_comparison] No strategy results provided.")
        return

    strategies = list(strategy_results.keys())
    all_metrics = set()
    for v in strategy_results.values():
        all_metrics.update(v.keys())
    metric_names = sorted(all_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.8 / len(strategies)
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for i, strategy in enumerate(strategies):
        values = [strategy_results[strategy].get(m, 0) for m in metric_names]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=strategy, color=colors[i])

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[plot_strategy_comparison] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _load_training_log(log_path: Union[str, Path]) -> Dict[str, List[float]]:
    """Load training log from CSV or JSON file."""
    log_path = Path(log_path)
    metrics: Dict[str, List[float]] = {}

    if not log_path.exists():
        return metrics

    if log_path.suffix == ".json":
        import json
        with open(log_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                for key, val in entry.items():
                    if isinstance(val, (int, float)):
                        metrics.setdefault(key, []).append(float(val))
        elif isinstance(data, dict):
            metrics = {k: v for k, v in data.items() if isinstance(v, list)}

    elif log_path.suffix == ".csv":
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    try:
                        metrics.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        continue

    return metrics
