"""FlashFusion visualization — Draw detections, fusion results, and overlays."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

COLORS = [
    (255, 56, 56),    # Red
    (255, 157, 56),   # Orange
    (255, 224, 56),   # Yellow
    (56, 255, 56),    # Green
    (56, 255, 224),   # Cyan
    (56, 157, 255),   # Blue
    (157, 56, 255),   # Purple
    (255, 56, 224),   # Pink
    (56, 56, 255),    # Deep Blue
    (224, 255, 56),   # Lime
    (255, 112, 56),   # Dark Orange
    (56, 224, 255),   # Light Cyan
    (168, 56, 255),   # Violet
    (255, 56, 112),   # Rose
    (56, 255, 112),   # Mint
    (112, 56, 255),   # Indigo
    (255, 168, 56),   # Amber
    (56, 112, 255),   # Royal Blue
    (255, 56, 168),   # Magenta
    (112, 255, 56),   # Chartreuse
]


def draw_detections(
    image: Union[str, Path, np.ndarray],
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
    line_width: int = 2,
    font_scale: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Draw detection boxes on an image.

    Args:
        image: Image as numpy array (H, W, C) in BGR or path to image file.
        boxes: Detection boxes of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Confidence scores of shape (N,). If None, all boxes are drawn.
        labels: Class labels of shape (N,). Used for color coding.
        class_names: List of class name strings for label display.
        conf_threshold: Minimum confidence to draw a box.
        line_width: Box line width in pixels.
        font_scale: Font scale for text labels.
        save_path: If provided, saves result image to this path.

    Returns:
        Annotated image as numpy array (BGR).
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        image = image.copy()

    if len(boxes) == 0:
        return image

    for i in range(len(boxes)):
        if scores is not None and scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        label_id = int(labels[i]) if labels is not None else 0
        color = COLORS[label_id % len(COLORS)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

        # Build label text
        label_parts = []
        if class_names and label_id < len(class_names):
            label_parts.append(class_names[label_id])
        elif labels is not None:
            label_parts.append(f"cls{label_id}")

        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label_text = " ".join(label_parts)
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color, -1,
            )
            cv2.putText(
                image, label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)

    return image


def draw_fusion_results(
    image: Union[str, Path, np.ndarray],
    fused_results: Dict[str, Any],
    individual_results: Optional[List[Dict[str, Any]]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_individual: bool = True,
) -> np.ndarray:
    """Draw fusion results with optional individual model overlays.

    Creates a visualization showing the fused detections prominently,
    with optional transparent overlays of each individual model's predictions.

    Args:
        image: Input image (BGR numpy array or file path).
        fused_results: Fused prediction dict with 'boxes', 'scores', 'labels'.
        individual_results: Optional list of per-model prediction dicts.
        class_names: Class name strings for labels.
        save_path: If provided, saves the visualization.
        show_individual: Whether to overlay individual model predictions.

    Returns:
        Annotated image as numpy array (BGR).
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        image = image.copy()

    # Draw individual model results with transparency
    if show_individual and individual_results:
        overlay = image.copy()
        for model_idx, model_result in enumerate(individual_results):
            boxes = _to_numpy(model_result.get("boxes", []))
            if len(boxes) == 0:
                continue
            color = COLORS[(model_idx + 10) % len(COLORS)]
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    # Draw fused results prominently
    fused_boxes = _to_numpy(fused_results.get("boxes", []))
    fused_scores = _to_numpy(fused_results.get("scores", []))
    fused_labels = _to_numpy(fused_results.get("labels", []))

    image = draw_detections(
        image, fused_boxes, fused_scores, fused_labels,
        class_names=class_names, line_width=3, font_scale=0.6,
    )

    # Add legend
    legend_y = 20
    cv2.putText(
        image, "FUSED (bold)", (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if show_individual and individual_results:
        for i in range(len(individual_results)):
            legend_y += 20
            color = COLORS[(i + 10) % len(COLORS)]
            cv2.putText(
                image, f"Model {i + 1} (faint)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)

    return image


def _to_numpy(data) -> np.ndarray:
    """Convert data to numpy array."""
    import torch
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data) if data else np.zeros((0,))
    return np.array(data)
