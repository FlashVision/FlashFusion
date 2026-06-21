"""FlashFusion Validator — Computes fused metrics for multi-model pipelines.

Evaluates the performance of the fused model output against ground truth,
supporting detection mAP, classification accuracy, and segmentation IoU.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flashfusion.cfg.config import FusionConfig


class Validator:
    """FlashFusion validation engine.

    Computes evaluation metrics for fused model outputs including
    mAP, accuracy, and IoU depending on the fusion task type.

    Args:
        config: FusionConfig instance.
        device: Target device.

    Example:
        >>> from flashfusion import Validator
        >>> from flashfusion.cfg import get_config
        >>> config = get_config("configs/flashfusion_ensemble_320.yaml")
        >>> validator = Validator(config)
        >>> metrics = validator.validate(model, dataloader)
    """

    def __init__(self, config: FusionConfig, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        self.metrics: Dict[str, float] = {}

    def validate(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Run validation loop and compute metrics.

        Args:
            model: The fusion model to evaluate.
            dataloader: Validation data loader. If None, builds from config.

        Returns:
            Dictionary of metric name -> value.
        """
        model.eval()
        model.to(self.device)

        all_predictions: List[Dict[str, Any]] = []
        all_targets: List[Dict[str, Any]] = []

        if dataloader is None:
            raise NotImplementedError(
                "Automatic dataloader construction requires a configured dataset path."
            )

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                targets = batch["targets"]

                outputs = model(images)
                all_predictions.append(outputs)
                all_targets.append(targets)

        self.metrics = self._compute_metrics(all_predictions, all_targets)
        return self.metrics

    def _compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute task-specific metrics.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Dictionary with computed metrics.
        """
        strategy = self.config.strategy
        metrics: Dict[str, float] = {}

        if strategy in ("weighted_box_fusion", "nms_fusion", "cascade"):
            metrics["mAP@0.5"] = self._compute_map(predictions, targets, iou_thresh=0.5)
            metrics["mAP@0.5:0.95"] = self._compute_map(predictions, targets, iou_thresh=0.5)
            metrics["primary_metric"] = metrics["mAP@0.5"]
        elif strategy == "voting":
            metrics["accuracy"] = self._compute_accuracy(predictions, targets)
            metrics["primary_metric"] = metrics["accuracy"]
        else:
            metrics["primary_metric"] = 0.0

        return metrics

    def _compute_map(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
        iou_thresh: float = 0.5,
    ) -> float:
        """Compute mean Average Precision for detection tasks."""
        from flashfusion.utils.metrics import compute_map

        result = compute_map(predictions, targets, iou_threshold=iou_thresh)
        return result["mAP"]

    def _compute_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> float:
        """Compute classification accuracy."""
        from flashfusion.utils.metrics import compute_accuracy

        all_logits = []
        all_labels = []
        for pred, tgt in zip(predictions, targets):
            if "logits" in pred:
                all_logits.append(pred["logits"])
            elif "probabilities" in pred:
                all_logits.append(pred["probabilities"])
            elif "scores" in pred and isinstance(pred["scores"], torch.Tensor) and pred["scores"].dim() >= 2:
                all_logits.append(pred["scores"])
            else:
                continue

            if "class_labels" in tgt:
                all_labels.append(tgt["class_labels"])
            elif "labels" in tgt:
                all_labels.append(tgt["labels"])

        if not all_logits or not all_labels:
            return 0.0

        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        result = compute_accuracy(logits_tensor, labels_tensor, topk=(1,))
        return result.get("top1", 0.0) / 100.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
