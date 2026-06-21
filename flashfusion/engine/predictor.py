"""FlashFusion Predictor — Runs multi-model inference and fuses results.

Handles image loading, preprocessing, multi-model forward passes,
fusion strategy application, and result post-processing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from flashfusion.cfg.config import FusionConfig


class Predictor:
    """FlashFusion prediction engine.

    Runs inference through multiple models and applies the configured
    fusion strategy to produce unified predictions.

    Args:
        config: FusionConfig instance.
        device: Target device.

    Example:
        >>> from flashfusion import Predictor
        >>> from flashfusion.cfg import get_config
        >>> config = get_config("configs/flashfusion_ensemble_320.yaml")
        >>> predictor = Predictor(config)
        >>> results = predictor.predict(source="image.jpg")
    """

    def __init__(self, config: FusionConfig, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        self.model: Optional[nn.Module] = None
        self._setup()

    def _setup(self) -> None:
        """Initialize the fusion model from config."""
        from flashfusion.models.fusion import FlashFusion

        model_sources = [m.source for m in self.config.models]
        weights = [m.weight for m in self.config.models]

        self.model = FlashFusion(
            models=model_sources,
            input_size=self.config.input_size,
            weights=weights,
            device=str(self.device),
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        source: Union[str, Path, np.ndarray, torch.Tensor],
        save_dir: Optional[str] = None,
        conf_threshold: float = 0.25,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run prediction on input source.

        Args:
            source: Image path, directory path, video path, numpy array, tensor,
                    or integer (webcam device index).
            save_dir: Directory to save visualization results.
            conf_threshold: Confidence threshold for filtering predictions.

        Returns:
            List of prediction result dictionaries with keys:
                - 'boxes': Detected bounding boxes (if applicable).
                - 'scores': Confidence scores.
                - 'labels': Class labels.
                - 'masks': Segmentation masks (if applicable).
        """
        if isinstance(source, int) or (
            isinstance(source, str) and source.isdigit()
        ):
            return self._predict_webcam(int(source) if isinstance(source, str) else source, conf_threshold, save_dir)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                return self._predict_video(str(path), conf_threshold, save_dir)

        images = self._load_source(source)
        results = []

        for idx, image in enumerate(images):
            tensor = self._preprocess(image)
            with torch.no_grad():
                output = self.model(tensor)
            result = self._postprocess(output, conf_threshold)
            result["_source_image"] = image
            result["_index"] = idx
            results.append(result)

        if save_dir:
            self._save_results(results, save_dir)

        return results

    def _predict_video(self, video_path: str, conf_threshold: float, save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run inference on a video file frame by frame."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        results = []
        frame_idx = 0
        writer = None

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = str(Path(save_dir) / "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tensor = self._preprocess(frame)
                with torch.no_grad():
                    output = self.model(tensor)
                result = self._postprocess(output, conf_threshold)
                result["frame_idx"] = frame_idx
                results.append(result)

                if writer is not None:
                    annotated = self._draw_detections(frame, result)
                    writer.write(annotated)

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        return results

    def _predict_webcam(self, device_id: int, conf_threshold: float, save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run inference on webcam stream. Press 'q' to stop."""
        import cv2

        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam device: {device_id}")

        results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tensor = self._preprocess(frame)
                with torch.no_grad():
                    output = self.model(tensor)
                result = self._postprocess(output, conf_threshold)
                result["frame_idx"] = frame_idx
                results.append(result)
                frame_idx += 1

                if frame_idx > 300:
                    break
        finally:
            cap.release()

        return results

    def _load_source(self, source: Union[str, Path, np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Load images from various source types."""
        import cv2

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Failed to load image: {path}")
                return [img]
            elif path.is_dir():
                extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                images = []
                for ext in extensions:
                    images.extend(sorted(path.glob(f"*{ext}")))
                return [cv2.imread(str(p)) for p in images]
            else:
                raise FileNotFoundError(f"Source not found: {path}")
        elif isinstance(source, np.ndarray):
            return [source]
        elif isinstance(source, torch.Tensor):
            return [source.cpu().numpy()]
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        import cv2

        h, w = self.config.input_size
        resized = cv2.resize(image, (w, h))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess(self, output: Dict[str, Any], conf_threshold: float) -> Dict[str, Any]:
        """Post-process model output and apply confidence filtering."""
        result = {
            "boxes": [],
            "scores": [],
            "labels": [],
        }

        if isinstance(output, dict):
            if "scores" in output:
                scores = output["scores"]
                if isinstance(scores, torch.Tensor):
                    mask = scores > conf_threshold
                    result["scores"] = scores[mask].cpu().numpy()
                    if "boxes" in output:
                        result["boxes"] = output["boxes"][mask].cpu().numpy()
                    if "labels" in output:
                        result["labels"] = output["labels"][mask].cpu().numpy()

        return result

    def _save_results(self, results: List[Dict[str, Any]], save_dir: str) -> None:
        """Save prediction results and annotated visualizations."""
        import cv2
        import json

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        all_serializable = []
        for idx, result in enumerate(results):
            source_image = result.pop("_source_image", None)
            result.pop("_index", None)

            if source_image is not None:
                annotated = self._draw_detections(source_image, result)
                img_filename = f"result_{idx:04d}.jpg"
                cv2.imwrite(str(save_path / img_filename), annotated)

            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, torch.Tensor):
                    serializable_result[k] = v.cpu().numpy().tolist()
                else:
                    serializable_result[k] = v
            all_serializable.append(serializable_result)

        with open(save_path / "results.json", "w") as f:
            json.dump(all_serializable, f, indent=2)

    def _draw_detections(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw detection boxes on image."""
        import cv2

        annotated = image.copy()
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        if isinstance(boxes, np.ndarray) and len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                score = float(scores[i]) if i < len(scores) else 0.0
                label = int(labels[i]) if i < len(labels) else 0
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {score:.2f}"
                cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
