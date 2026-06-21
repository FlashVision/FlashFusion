"""Detection → Classification pipeline.

Runs detection first to locate objects, then classifies each detected
region for fine-grained recognition (e.g., detect vehicles then classify make/model).
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

from flashfusion.registry import PIPELINES


@PIPELINES.register("det_cls")
class DetClsPipeline:
    """Detection followed by Classification pipeline.

    First detects objects in the image, then crops each detection and
    runs classification for fine-grained category recognition.

    Args:
        detector: Detection model identifier or path.
        classifier: Classification model identifier or path.
        det_threshold: Detection confidence threshold.
        cls_threshold: Classification confidence threshold.
        crop_padding: Padding ratio when cropping detections for classification.
        device: Target device.

    Example:
        >>> pipeline = DetClsPipeline(
        ...     detector="flashdet-m",
        ...     classifier="flashcls-m",
        ...     det_threshold=0.5,
        ... )
        >>> results = pipeline("image.jpg")
    """

    def __init__(
        self,
        detector: Union[str, Any] = "flashdet-m",
        classifier: Union[str, Any] = "flashcls-m",
        det_threshold: float = 0.5,
        cls_threshold: float = 0.3,
        crop_padding: float = 0.1,
        device: str = "auto",
    ):
        self.detector = detector
        self.classifier = classifier
        self.det_threshold = det_threshold
        self.cls_threshold = cls_threshold
        self.crop_padding = crop_padding
        self.device = self._resolve_device(device)

        self._detector_model = None
        self._classifier_model = None

    def __call__(self, source: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Run the detection → classification pipeline.

        Args:
            source: Input image path or array.

        Returns:
            List of result dicts with keys: 'bbox', 'det_score', 'label',
            'cls_score', 'confidence'.
        """
        return self.predict(source)

    def predict(self, source: Union[str, Path, np.ndarray]) -> List[Dict[str, Any]]:
        """Run full pipeline prediction.

        Args:
            source: Input image.

        Returns:
            Combined detection + classification results.
        """
        image = self._load_image(source)
        detections = self._run_detection(image)
        results = []

        for det in detections:
            if det["score"] < self.det_threshold:
                continue

            crop = self._crop_detection(image, det["bbox"])
            cls_result = self._run_classification(crop)

            results.append(
                {
                    "bbox": det["bbox"],
                    "det_score": det["score"],
                    "label": cls_result["label"],
                    "cls_score": cls_result["score"],
                    "confidence": det["score"] * cls_result["score"],
                }
            )

        return results

    def _run_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run the detection model.

        Args:
            image: Input image array.

        Returns:
            List of detection dicts with 'bbox' and 'score'.
        """
        import cv2

        if self._detector_model is None:
            self._detector_model = self._load_torch_model(self.detector)

        h, w = image.shape[:2]
        resized = cv2.resize(image, (320, 320))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._detector_model(tensor)

        detections = []
        if isinstance(output, dict):
            boxes = output.get("boxes", torch.zeros(0, 4))
            scores = output.get("scores", torch.zeros(0))
            labels = output.get("labels", torch.zeros(0))

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

            scale_x, scale_y = w / 320.0, h / 320.0
            for i in range(len(scores)):
                box = boxes[i].copy()
                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y
                detections.append(
                    {
                        "bbox": box.tolist(),
                        "score": float(scores[i]),
                        "label": int(labels[i]) if i < len(labels) else 0,
                    }
                )

        return detections

    def _run_classification(self, crop: np.ndarray) -> Dict[str, Any]:
        """Run the classification model on a cropped region.

        Args:
            crop: Cropped image region.

        Returns:
            Classification result with 'label' and 'score'.
        """
        import cv2

        if self._classifier_model is None:
            self._classifier_model = self._load_torch_model(self.classifier)

        resized = cv2.resize(crop, (224, 224))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._classifier_model(tensor)

        if isinstance(output, dict):
            if "logits" in output:
                logits = output["logits"]
            elif "probabilities" in output:
                logits = output["probabilities"]
            else:
                logits = torch.zeros(1, 1)
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return {"label": 0, "score": 0.0}

        probs = torch.softmax(logits, dim=-1)
        score, label = torch.max(probs, dim=-1)
        return {"label": int(label.item()), "score": float(score.item())}

    def _load_torch_model(self, model_id) -> torch.nn.Module:
        """Load a torch model from path or return if already a module."""
        if isinstance(model_id, torch.nn.Module):
            model_id.to(self.device)
            model_id.eval()
            return model_id

        from pathlib import Path as _Path

        path = _Path(str(model_id))
        if path.exists() and path.suffix in (".pt", ".pth"):
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                checkpoint.eval()
                return checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model = checkpoint["model"]
                model.to(self.device)
                model.eval()
                return model

        raise ValueError(f"Cannot load model '{model_id}'. Provide a valid .pt/.pth path or nn.Module instance.")

    def _crop_detection(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop a detection region from the image with padding.

        Args:
            image: Full image array (H, W, C).
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Cropped image region.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        pad_w = (x2 - x1) * self.crop_padding
        pad_h = (y2 - y1) * self.crop_padding

        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))

        return image[y1:y2, x1:x2]

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or return array directly."""
        if isinstance(source, np.ndarray):
            return source

        import cv2

        path = Path(source)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return (
            f"DetClsPipeline(detector='{self.detector}', "
            f"classifier='{self.classifier}', "
            f"det_threshold={self.det_threshold})"
        )
