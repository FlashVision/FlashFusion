"""NMS-based fusion for overlapping detections from multiple models.

Applies Non-Maximum Suppression across predictions from multiple detection
models to deduplicate overlapping boxes while preserving unique detections.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("nms_fusion")
class NMSFusion:
    """NMS-based fusion for multi-model detection deduplication.

    Collects all detections from multiple models and applies NMS
    to remove duplicate/overlapping predictions.

    Args:
        iou_threshold: IoU threshold for NMS suppression.
        score_threshold: Minimum confidence score to keep.
        max_detections: Maximum number of detections to return.
        class_agnostic: Whether to apply NMS across all classes or per-class.

    Example:
        >>> nms = NMSFusion(iou_threshold=0.5, score_threshold=0.3)
        >>> fused = nms.fuse(model_outputs)
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.01,
        max_detections: int = 300,
        class_agnostic: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.class_agnostic = class_agnostic

    def fuse(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fuse detection predictions using NMS.

        Args:
            predictions: List of prediction dicts from each model with
                'boxes' (N, 4), 'scores' (N,), 'labels' (N,).
            weights: Per-model confidence scaling weights.

        Returns:
            Fused predictions after NMS deduplication.
        """
        model_weights = weights or [1.0] * len(predictions)

        all_boxes = []
        all_scores = []
        all_labels = []

        for pred, w in zip(predictions, model_weights):
            boxes = self._to_tensor(pred.get("boxes", torch.zeros(0, 4)))
            scores = self._to_tensor(pred.get("scores", torch.zeros(0)))
            labels = self._to_tensor(pred.get("labels", torch.zeros(0))).long()

            if boxes.numel() == 0:
                continue

            scores = scores * w
            mask = scores > self.score_threshold
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_labels.append(labels[mask])

        if not all_boxes:
            return {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            }

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if self.class_agnostic:
            keep = self._nms(boxes, scores)
        else:
            keep = self._batched_nms(boxes, scores, labels)

        keep = keep[: self.max_detections]

        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply class-agnostic NMS.

        Args:
            boxes: (N, 4) boxes in [x1, y1, x2, y2] format.
            scores: (N,) confidence scores.

        Returns:
            Indices of kept boxes.
        """
        try:
            from torchvision.ops import nms
            return nms(boxes, scores, self.iou_threshold)
        except ImportError:
            return self._nms_numpy(boxes, scores)

    def _batched_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply per-class NMS.

        Args:
            boxes: (N, 4) boxes.
            scores: (N,) scores.
            labels: (N,) class labels.

        Returns:
            Indices of kept boxes.
        """
        try:
            from torchvision.ops import batched_nms
            return batched_nms(boxes, scores, labels, self.iou_threshold)
        except ImportError:
            return self._nms(boxes, scores)

    def _nms_numpy(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Fallback NMS implementation using NumPy."""
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()

        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores_np.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.long)

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        """Convert data to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(np.array(data), dtype=torch.float32)

    def __repr__(self) -> str:
        return (
            f"NMSFusion(iou_threshold={self.iou_threshold}, "
            f"max_detections={self.max_detections})"
        )
