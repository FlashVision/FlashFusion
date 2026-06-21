"""FusionLoss — Combined multi-task loss for detection + classification + consistency.

Combines detection regression/classification losses with an image-level
classification loss and optional inter-model consistency regularization.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLoss(nn.Module):
    """Multi-task fusion loss combining detection, classification, and consistency terms.

    The total loss is a weighted sum:
        L = w_det * L_detection + w_cls * L_classification + w_con * L_consistency

    Args:
        det_weight: Weight for detection loss component.
        cls_weight: Weight for classification loss component.
        consistency_weight: Weight for consistency loss between model outputs.
        num_classes: Number of object detection classes.
        cls_num_classes: Number of image-level classification classes.
        label_smoothing: Label smoothing factor for classification.

    Example:
        >>> criterion = FusionLoss(det_weight=1.0, cls_weight=0.5, consistency_weight=0.1)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        det_weight: float = 1.0,
        cls_weight: float = 0.5,
        consistency_weight: float = 0.1,
        num_classes: int = 80,
        cls_num_classes: int = 10,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.det_weight = det_weight
        self.cls_weight = cls_weight
        self.consistency_weight = consistency_weight
        self.num_classes = num_classes
        self.cls_num_classes = cls_num_classes

        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.box_criterion = nn.SmoothL1Loss(reduction="mean")

        from flashfusion.losses.consistency_loss import ConsistencyLoss
        self.consistency_criterion = ConsistencyLoss()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        model_outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Compute the combined fusion loss.

        Args:
            predictions: Fused model predictions containing:
                - 'boxes': predicted boxes (B, N, 4)
                - 'scores': predicted class scores (B, N, num_classes)
                - 'class_logits': image classification logits (B, cls_num_classes)
            targets: Ground truth containing:
                - 'boxes': list of target boxes per sample
                - 'labels': list of target labels per sample
                - 'class_labels': image-level class labels (B,)
            model_outputs: Optional list of per-model output dicts for consistency loss.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))

        # Detection loss
        det_loss = self._compute_detection_loss(predictions, targets)
        total_loss = total_loss + self.det_weight * det_loss

        # Classification loss
        if "class_logits" in predictions and "class_labels" in targets:
            cls_loss = self._compute_classification_loss(predictions, targets)
            total_loss = total_loss + self.cls_weight * cls_loss

        # Consistency loss
        if model_outputs is not None and len(model_outputs) > 1:
            con_loss = self.consistency_criterion(model_outputs)
            total_loss = total_loss + self.consistency_weight * con_loss

        return total_loss

    def _compute_detection_loss(
        self, predictions: Dict[str, Any], targets: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute detection loss (box regression + objectness).

        Uses smooth L1 for box regression and focal-style BCE for objectness.
        """
        pred_boxes = predictions.get("boxes")
        target_boxes = targets.get("boxes", [])

        if pred_boxes is None or not target_boxes:
            return torch.tensor(0.0, requires_grad=True)

        # Flatten target boxes for batch loss computation
        all_pred = []
        all_target = []
        if isinstance(pred_boxes, torch.Tensor) and pred_boxes.dim() == 3:
            batch_size = pred_boxes.shape[0]
            for i in range(batch_size):
                tgt = target_boxes[i] if i < len(target_boxes) else torch.zeros((0, 4))
                if isinstance(tgt, torch.Tensor) and tgt.numel() > 0:
                    num_targets = tgt.shape[0]
                    num_preds = min(pred_boxes.shape[1], num_targets)
                    all_pred.append(pred_boxes[i, :num_preds])
                    all_target.append(tgt[:num_preds])
        else:
            return torch.tensor(0.0, requires_grad=True)

        if not all_pred:
            return torch.tensor(0.0, requires_grad=True)

        pred_cat = torch.cat(all_pred, dim=0)
        target_cat = torch.cat(all_target, dim=0)

        box_loss = self.box_criterion(pred_cat, target_cat)

        # Objectness loss via focal BCE
        if "scores" in predictions:
            scores = predictions["scores"]
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                obj_targets = torch.ones_like(scores[:, :1]) if scores.dim() > 1 else torch.ones_like(scores)
                obj_loss = F.binary_cross_entropy_with_logits(
                    scores.view(-1)[:obj_targets.numel()],
                    obj_targets.view(-1),
                    reduction="mean",
                )
                box_loss = box_loss + obj_loss

        return box_loss

    def _compute_classification_loss(
        self, predictions: Dict[str, Any], targets: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute image-level classification loss."""
        logits = predictions["class_logits"]
        labels = targets["class_labels"]

        if isinstance(labels, (list, tuple)):
            labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)

        labels = labels.to(logits.device)
        return self.cls_criterion(logits, labels)

    @staticmethod
    def _get_device(predictions: Dict[str, Any]) -> torch.device:
        """Extract device from predictions dict."""
        for v in predictions.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def __repr__(self) -> str:
        return (
            f"FusionLoss(det_weight={self.det_weight}, "
            f"cls_weight={self.cls_weight}, "
            f"consistency_weight={self.consistency_weight})"
        )
