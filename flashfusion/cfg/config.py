"""FlashFusion configuration management.

Provides a dataclass-based configuration system with YAML loading support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class ModelSource:
    """Configuration for a single model source in the fusion pipeline."""

    type: str = "detection"
    source: str = "flashdet"
    model_size: str = "m"
    weight: float = 1.0
    checkpoint: Optional[str] = None
    frozen: bool = True


@dataclass
class FusionConfig:
    """Main configuration dataclass for FlashFusion.

    Attributes:
        strategy: Fusion strategy name (weighted_box_fusion, voting, cascade, stacking, nms_fusion).
        models: List of model source configurations.
        input_size: Input image size as (height, width).
        weights: Optional per-model weights for fusion.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for fusion layer training.
        save_dir: Directory for saving outputs.
        device: Target device (auto, cpu, cuda).
        num_workers: Number of data loading workers.
        conf_threshold: Confidence threshold for predictions.
        iou_threshold: IoU threshold for NMS/WBF.
        max_detections: Maximum number of detections per image.
        export_format: Export format (onnx, torchscript).
        data_path: Path to dataset root directory.
        val_split: Validation split ratio when no explicit val set.
        augment: Augmentation configuration dictionary.
        warmup_epochs: Number of warmup epochs for scheduler.
        scheduler: Learning rate scheduler name.
        early_stopping: Early stopping configuration dictionary.
        weight_decay: Weight decay for optimizer.
    """

    strategy: str = "weighted_box_fusion"
    models: List[ModelSource] = field(default_factory=list)
    input_size: Tuple[int, int] = (320, 320)
    weights: Optional[List[float]] = None
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.0005
    save_dir: str = "workspace/flashfusion"
    device: str = "auto"
    num_workers: int = 4
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    max_detections: int = 300
    export_format: str = "onnx"
    data_path: Optional[str] = None
    val_split: float = 0.2
    augment: Dict[str, Any] = field(default_factory=dict)
    warmup_epochs: int = 0
    scheduler: str = "cosine"
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {"patience": 10, "min_delta": 0.001})
    weight_decay: float = 1e-4


def _parse_model_sources(models_data: List[Dict[str, Any]]) -> List[ModelSource]:
    """Parse model source dictionaries into ModelSource objects."""
    sources = []
    for m in models_data:
        sources.append(ModelSource(
            type=m.get("type", "detection"),
            source=m.get("source", "flashdet"),
            model_size=m.get("model_size", "m"),
            weight=m.get("weight", 1.0),
            checkpoint=m.get("checkpoint"),
            frozen=m.get("frozen", True),
        ))
    return sources


def get_config(config_path: Union[str, Path, None] = None, overrides: Optional[Dict[str, Any]] = None) -> FusionConfig:
    """Load configuration from a YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file. If None, returns default config.
        overrides: Dictionary of overrides to apply on top of file config.

    Returns:
        FusionConfig instance with all settings resolved.
    """
    config = FusionConfig()

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            return config

        fusion_data = data.get("fusion", {})
        train_data = data.get("train", {})
        data_section = data.get("data", {})
        augment_section = data.get("augment", {})

        if "strategy" in fusion_data:
            config.strategy = fusion_data["strategy"]
        if "models" in fusion_data:
            config.models = _parse_model_sources(fusion_data["models"])
        if "input_size" in fusion_data:
            size = fusion_data["input_size"]
            config.input_size = tuple(size) if isinstance(size, list) else (size, size)
        if "weights" in fusion_data:
            config.weights = fusion_data["weights"]

        if "epochs" in train_data:
            config.epochs = train_data["epochs"]
        if "batch_size" in train_data:
            config.batch_size = train_data["batch_size"]
        if "learning_rate" in train_data:
            config.learning_rate = train_data["learning_rate"]
        if "save_dir" in train_data:
            config.save_dir = train_data["save_dir"]
        if "device" in train_data:
            config.device = train_data["device"]
        if "num_workers" in train_data:
            config.num_workers = train_data["num_workers"]
        if "warmup_epochs" in train_data:
            config.warmup_epochs = train_data["warmup_epochs"]
        if "scheduler" in train_data:
            config.scheduler = train_data["scheduler"]
        if "weight_decay" in train_data:
            config.weight_decay = train_data["weight_decay"]
        if "early_stopping" in train_data:
            es = train_data["early_stopping"]
            if isinstance(es, dict):
                config.early_stopping = es
            elif isinstance(es, bool) and not es:
                config.early_stopping = {}

        # Data section
        if "train" in data_section:
            config.data_path = data_section["train"]
        elif "path" in data_section:
            config.data_path = data_section["path"]
        if "val_split" in data_section:
            config.val_split = data_section["val_split"]

        # Augmentation section
        if augment_section:
            config.augment = augment_section

    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return config
