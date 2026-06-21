"""Weighted Box Fusion (WBF) strategy for detection ensembles.

WBF merges overlapping bounding boxes from multiple detection models
using confidence-weighted averaging, producing more accurate localizations
than standard NMS.

Reference:
    Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different
    object detection models", Image and Vision Computing, 2021.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("weighted_box_fusion")
class WeightedBoxFusion:
    """Weighted Box Fusion for detection ensemble.

    Combines bounding box predictions from multiple models by clustering
    overlapping boxes and computing weighted averages of their coordinates.

    Args:
        weights: Per-model weights for fusion. If None, uniform weights are used.
        iou_threshold: IoU threshold for matching boxes across models.
        skip_box_threshold: Minimum confidence to consider a box.
        conf_type: How to compute fused confidence ('avg', 'max', 'box_and_model_avg').

    Example:
        >>> wbf = WeightedBoxFusion(weights=[0.6, 0.4], iou_threshold=0.55)
        >>> fused = wbf.fuse(model_outputs)
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.01,
        conf_type: str = "avg",
    ):
        self.weights = weights
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type

    def fuse(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fuse detection predictions from multiple models using WBF.

        Args:
            predictions: List of prediction dicts from each model, each containing
                'boxes' (N, 4), 'scores' (N,), 'labels' (N,).
            weights: Override per-model weights.

        Returns:
            Fused prediction dictionary with merged boxes, scores, and labels.
        """
        model_weights = weights or self.weights
        if model_weights is None:
            model_weights = [1.0] * len(predictions)

        all_boxes = []
        all_scores = []
        all_labels = []

        for pred in predictions:
            boxes = self._to_numpy(pred.get("boxes", []))
            scores = self._to_numpy(pred.get("scores", []))
            labels = self._to_numpy(pred.get("labels", []))
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        fused_boxes, fused_scores, fused_labels = self._wbf(all_boxes, all_scores, all_labels, model_weights)

        return {
            "boxes": torch.from_numpy(fused_boxes).float(),
            "scores": torch.from_numpy(fused_scores).float(),
            "labels": torch.from_numpy(fused_labels).long(),
        }

    def _wbf(
        self,
        boxes_list: List[np.ndarray],
        scores_list: List[np.ndarray],
        labels_list: List[np.ndarray],
        weights: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core WBF algorithm.

        Args:
            boxes_list: List of box arrays per model (each N_i x 4).
            scores_list: List of score arrays per model.
            labels_list: List of label arrays per model.
            weights: Per-model weights.

        Returns:
            Tuple of (fused_boxes, fused_scores, fused_labels).
        """
        if not any(len(b) > 0 for b in boxes_list):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)

        # Collect all boxes with model index and weight
        weighted_boxes = []
        for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
            for box_idx in range(len(boxes)):
                if scores[box_idx] < self.skip_box_threshold:
                    continue
                weighted_boxes.append(
                    {
                        "box": boxes[box_idx],
                        "score": scores[box_idx] * weights[model_idx],
                        "label": labels[box_idx] if len(labels) > box_idx else 0,
                        "model_idx": model_idx,
                    }
                )

        if not weighted_boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)

        # Sort by score descending
        weighted_boxes.sort(key=lambda x: -x["score"])

        # Cluster overlapping boxes
        clusters: List[List[Dict]] = []
        for wb in weighted_boxes:
            matched = False
            for cluster in clusters:
                if cluster[0]["label"] != wb["label"]:
                    continue
                cluster_box = self._compute_cluster_box(cluster)
                if self._compute_iou(cluster_box, wb["box"]) > self.iou_threshold:
                    cluster.append(wb)
                    matched = True
                    break
            if not matched:
                clusters.append([wb])

        # Compute fused boxes from clusters
        fused_boxes = []
        fused_scores = []
        fused_labels = []

        num_models = len(boxes_list)
        for cluster in clusters:
            fused_box = self._compute_cluster_box(cluster)
            fused_boxes.append(fused_box)
            fused_labels.append(cluster[0]["label"])

            if self.conf_type == "max":
                fused_scores.append(max(wb["score"] for wb in cluster))
            elif self.conf_type == "box_and_model_avg":
                fused_scores.append(sum(wb["score"] for wb in cluster) / num_models)
            else:
                fused_scores.append(sum(wb["score"] for wb in cluster) / len(cluster))

        return (
            np.array(fused_boxes),
            np.array(fused_scores),
            np.array(fused_labels, dtype=np.int64),
        )

    def _compute_cluster_box(self, cluster: List[Dict]) -> np.ndarray:
        """Compute weighted average box for a cluster."""
        total_weight = sum(wb["score"] for wb in cluster)
        if total_weight == 0:
            return cluster[0]["box"]

        box = np.zeros(4)
        for wb in cluster:
            box += wb["box"] * wb["score"]
        return box / total_weight

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data) if data else np.zeros((0,))
        return np.array(data)

    def __repr__(self) -> str:
        return (
            f"WeightedBoxFusion(weights={self.weights}, "
            f"iou_threshold={self.iou_threshold}, conf_type='{self.conf_type}')"
        )
