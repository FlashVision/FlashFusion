"""EnsembleDetector — High-level API for multi-model detection ensemble.

A turnkey solution that wraps model loading, fusion strategy configuration,
and inference into a simple interface for detection ensemble.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class EnsembleDetector:
    """High-level ensemble detection solution.

    Combines multiple detection models with configurable fusion strategies
    for improved accuracy and robustness.

    Args:
        models: List of model paths or identifiers.
        strategy: Fusion strategy ('wbf', 'nms', 'cascade').
        weights: Per-model weights.
        iou_threshold: IoU threshold for fusion.
        conf_threshold: Confidence threshold for filtering.
        device: Target device.

    Example:
        >>> detector = EnsembleDetector(
        ...     models=["model_a.pt", "model_b.pt", "model_c.pt"],
        ...     strategy="wbf",
        ...     weights=[0.5, 0.3, 0.2],
        ... )
        >>> results = detector.detect("image.jpg")
        >>> for det in results:
        ...     print(f"{det['label']}: {det['score']:.2f}")
    """

    def __init__(
        self,
        models: List[str],
        strategy: str = "wbf",
        weights: Optional[List[float]] = None,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        device: str = "auto",
    ):
        self.model_paths = models
        self.strategy_name = strategy
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)

        self._strategy = self._build_strategy()

    def detect(self, source: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Run ensemble detection on an image.

        Args:
            source: Image path or numpy array.

        Returns:
            List of detection results with 'bbox', 'score', 'label'.
        """
        predictions = self._run_all_models(source)
        fused = self._strategy.fuse(predictions, weights=self.weights)

        results = []
        boxes = fused.get("boxes", torch.zeros(0, 4))
        scores = fused.get("scores", torch.zeros(0))
        labels = fused.get("labels", torch.zeros(0))

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

        for i in range(len(scores)):
            if scores[i] >= self.conf_threshold:
                results.append(
                    {
                        "bbox": boxes[i].tolist(),
                        "score": float(scores[i]),
                        "label": int(labels[i]),
                    }
                )

        return results

    def _run_all_models(self, source: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Run all detection models on the source."""
        import cv2

        if isinstance(source, np.ndarray):
            image = source
        else:
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {source}")

        h, w = image.shape[:2]
        resized = cv2.resize(image, (320, 320))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        if not hasattr(self, "_loaded_models"):
            self._loaded_models = []
            for model_path in self.model_paths:
                model = self._load_model(model_path)
                self._loaded_models.append(model)

        predictions = []
        scale_x, scale_y = w / 320.0, h / 320.0

        for model in self._loaded_models:
            with torch.no_grad():
                output = model(tensor)

            if isinstance(output, dict):
                boxes = output.get("boxes", torch.zeros(0, 4))
                scores = output.get("scores", torch.zeros(0))
                labels = output.get("labels", torch.zeros(0, dtype=torch.long))

                if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                    scaled_boxes = boxes.clone()
                    scaled_boxes[:, 0] *= scale_x
                    scaled_boxes[:, 1] *= scale_y
                    scaled_boxes[:, 2] *= scale_x
                    scaled_boxes[:, 3] *= scale_y
                    predictions.append(
                        {
                            "boxes": scaled_boxes,
                            "scores": scores,
                            "labels": labels,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "boxes": torch.zeros(0, 4),
                            "scores": torch.zeros(0),
                            "labels": torch.zeros(0, dtype=torch.long),
                        }
                    )
            else:
                predictions.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )

        return predictions

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load a single detection model from path."""
        from pathlib import Path as _Path

        path = _Path(model_path)
        if path.exists() and path.suffix in (".pt", ".pth"):
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                checkpoint.to(self.device)
                checkpoint.eval()
                return checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model = checkpoint["model"]
                model.to(self.device)
                model.eval()
                return model

        raise ValueError(f"Cannot load model from '{model_path}'. Provide a valid .pt/.pth checkpoint.")

    def _build_strategy(self):
        """Build the fusion strategy instance."""
        from flashfusion.strategies import get_strategy

        return get_strategy(
            self.strategy_name,
            weights=self.weights,
            iou_threshold=self.iou_threshold,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return (
            f"EnsembleDetector(models={len(self.model_paths)}, strategy='{self.strategy_name}', weights={self.weights})"
        )
