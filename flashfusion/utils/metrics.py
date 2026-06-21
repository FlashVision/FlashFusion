"""FlashFusion metrics — mAP, accuracy, and fusion-specific evaluation metrics."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute mean Average Precision (mAP) for object detection.

    Args:
        predictions: List of prediction dicts, each with:
            - 'boxes': np.ndarray (N, 4) in [x1, y1, x2, y2]
            - 'scores': np.ndarray (N,)
            - 'labels': np.ndarray (N,)
        ground_truths: List of ground truth dicts, each with:
            - 'boxes': np.ndarray (M, 4)
            - 'labels': np.ndarray (M,)
        iou_threshold: IoU threshold for true positive matching.
        num_classes: Total number of classes. Auto-detected if None.

    Returns:
        Dictionary with 'mAP', 'mAP_per_class' (dict), and 'precision'/'recall'.
    """
    if not predictions or not ground_truths:
        return {"mAP": 0.0, "mAP_per_class": {}, "precision": 0.0, "recall": 0.0}

    all_labels = set()
    for gt in ground_truths:
        labels = gt.get("labels", [])
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        all_labels.update(labels.tolist() if hasattr(labels, "tolist") else list(labels))

    if num_classes is None:
        num_classes = max(all_labels) + 1 if all_labels else 1

    ap_per_class = {}
    for cls_id in range(num_classes):
        ap = _compute_ap_single_class(predictions, ground_truths, cls_id, iou_threshold)
        ap_per_class[cls_id] = ap

    valid_aps = [v for v in ap_per_class.values() if v > 0]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

    total_tp, total_fp, total_fn = _count_matches(predictions, ground_truths, iou_threshold)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    return {
        "mAP": mean_ap,
        "mAP_per_class": ap_per_class,
        "precision": precision,
        "recall": recall,
    }


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> Dict[str, float]:
    """Compute top-k accuracy for classification.

    Args:
        predictions: Model logits of shape (B, C).
        targets: Ground truth labels of shape (B,).
        topk: Tuple of k values to compute accuracy for.

    Returns:
        Dictionary mapping 'top1', 'top5', etc. to accuracy percentages.
    """
    if predictions.numel() == 0:
        return {f"top{k}": 0.0 for k in topk}

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred_indices = predictions.topk(maxk, dim=1, largest=True, sorted=True)
    pred_indices = pred_indices.t()
    correct = pred_indices.eq(targets.view(1, -1).expand_as(pred_indices))

    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = float(correct_k * 100.0 / batch_size)

    return results


def compute_fusion_metrics(
    fused_predictions: List[Dict[str, Any]],
    individual_predictions: List[List[Dict[str, Any]]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute fusion-specific metrics comparing fused vs individual model performance.

    Args:
        fused_predictions: Predictions from the fused ensemble.
        individual_predictions: List of predictions from each individual model.
        ground_truths: Ground truth annotations.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary with:
            - 'fused_map': mAP of the fused predictions
            - 'individual_maps': list of per-model mAPs
            - 'improvement': relative improvement over best individual model
            - 'agreement_rate': fraction of detections agreed upon by majority
            - 'complementarity': fraction of correct detections unique to fusion
    """
    fused_map_result = compute_map(fused_predictions, ground_truths, iou_threshold)
    fused_map = fused_map_result["mAP"]

    individual_maps = []
    for model_preds in individual_predictions:
        result = compute_map(model_preds, ground_truths, iou_threshold)
        individual_maps.append(result["mAP"])

    best_individual = max(individual_maps) if individual_maps else 0.0
    improvement = (fused_map - best_individual) / max(best_individual, 1e-6) * 100

    agreement_rate = _compute_agreement_rate(individual_predictions, iou_threshold)
    complementarity = _compute_complementarity(
        fused_predictions, individual_predictions, ground_truths, iou_threshold
    )

    return {
        "fused_map": fused_map,
        "individual_maps": individual_maps,
        "improvement": improvement,
        "agreement_rate": agreement_rate,
        "complementarity": complementarity,
    }


def _compute_ap_single_class(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    cls_id: int,
    iou_threshold: float,
) -> float:
    """Compute AP for a single class using 11-point interpolation."""
    all_scores = []
    all_matches = []
    total_gt = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = _to_numpy(pred.get("boxes", []))
        pred_scores = _to_numpy(pred.get("scores", []))
        pred_labels = _to_numpy(pred.get("labels", []))

        gt_boxes = _to_numpy(gt.get("boxes", []))
        gt_labels = _to_numpy(gt.get("labels", []))

        cls_mask_pred = pred_labels == cls_id
        cls_mask_gt = gt_labels == cls_id

        cls_pred_boxes = pred_boxes[cls_mask_pred] if len(pred_boxes) > 0 and cls_mask_pred.any() else np.zeros((0, 4))
        cls_pred_scores = pred_scores[cls_mask_pred] if len(pred_scores) > 0 and cls_mask_pred.any() else np.zeros(0)
        cls_gt_boxes = gt_boxes[cls_mask_gt] if len(gt_boxes) > 0 and cls_mask_gt.any() else np.zeros((0, 4))

        total_gt += len(cls_gt_boxes)
        matched_gt = set()

        sorted_idx = np.argsort(-cls_pred_scores) if len(cls_pred_scores) > 0 else []
        for idx in sorted_idx:
            all_scores.append(cls_pred_scores[idx])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in range(len(cls_gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                iou = _compute_iou(cls_pred_boxes[idx], cls_gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_matches.append(1)
                matched_gt.add(best_gt_idx)
            else:
                all_matches.append(0)

    if total_gt == 0:
        return 0.0

    sorted_indices = np.argsort(-np.array(all_scores)) if all_scores else []
    tp_cumsum = np.cumsum([all_matches[i] for i in sorted_indices]) if len(sorted_indices) > 0 else np.array([])
    fp_cumsum = np.cumsum([1 - all_matches[i] for i in sorted_indices]) if len(sorted_indices) > 0 else np.array([])

    if len(tp_cumsum) == 0:
        return 0.0

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall) / 11.0

    return float(ap)


def _count_matches(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Count total TP, FP, FN across all images."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = _to_numpy(pred.get("boxes", []))
        gt_boxes = _to_numpy(gt.get("boxes", []))

        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_threshold and best_idx >= 0:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    return total_tp, total_fp, total_fn


def _compute_agreement_rate(
    individual_predictions: List[List[Dict[str, Any]]],
    iou_threshold: float,
) -> float:
    """Compute the fraction of predictions agreed upon by majority of models."""
    if len(individual_predictions) < 2:
        return 1.0

    num_models = len(individual_predictions)
    num_images = len(individual_predictions[0]) if individual_predictions else 0
    majority = num_models // 2 + 1

    total_dets = 0
    agreed_dets = 0

    for img_idx in range(num_images):
        all_boxes = []
        for model_preds in individual_predictions:
            if img_idx < len(model_preds):
                boxes = _to_numpy(model_preds[img_idx].get("boxes", []))
                all_boxes.append(boxes)

        if not all_boxes:
            continue

        for model_idx, boxes in enumerate(all_boxes):
            for box in boxes:
                total_dets += 1
                support = 1
                for other_idx, other_boxes in enumerate(all_boxes):
                    if other_idx == model_idx:
                        continue
                    if any(_compute_iou(box, ob) >= iou_threshold for ob in other_boxes):
                        support += 1
                if support >= majority:
                    agreed_dets += 1

    return agreed_dets / max(total_dets, 1)


def _compute_complementarity(
    fused_predictions: List[Dict[str, Any]],
    individual_predictions: List[List[Dict[str, Any]]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float,
) -> float:
    """Compute fraction of correct fused detections not found by any individual model."""
    unique_correct = 0
    total_correct = 0

    for img_idx, (fused, gt) in enumerate(zip(fused_predictions, ground_truths)):
        fused_boxes = _to_numpy(fused.get("boxes", []))
        gt_boxes = _to_numpy(gt.get("boxes", []))

        for fb in fused_boxes:
            is_correct = any(_compute_iou(fb, gb) >= iou_threshold for gb in gt_boxes)
            if not is_correct:
                continue

            total_correct += 1
            found_by_individual = False
            for model_preds in individual_predictions:
                if img_idx < len(model_preds):
                    indiv_boxes = _to_numpy(model_preds[img_idx].get("boxes", []))
                    if any(_compute_iou(fb, ib) >= iou_threshold for ib in indiv_boxes):
                        found_by_individual = True
                        break

            if not found_by_individual:
                unique_correct += 1

    return unique_correct / max(total_correct, 1)


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

    return float(intersection / max(union, 1e-6))


def _to_numpy(data) -> np.ndarray:
    """Convert data to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data) if data else np.zeros((0,))
    return np.array(data)
