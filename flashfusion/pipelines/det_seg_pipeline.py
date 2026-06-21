"""Detection + Segmentation fusion pipeline.

Combines object detection with instance/semantic segmentation for
complete scene understanding with both bounding boxes and pixel masks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from flashfusion.registry import PIPELINES


@PIPELINES.register("det_seg")
class DetSegPipeline:
    """Detection and Segmentation fusion pipeline.

    Runs detection and segmentation in parallel (or sequentially),
    then fuses results to associate masks with detected objects.

    Args:
        detector: Detection model identifier or path.
        segmentor: Segmentation model identifier or path.
        det_threshold: Detection confidence threshold.
        mask_threshold: Segmentation mask threshold.
        fusion_mode: How to fuse results ('parallel', 'det_guided').
        device: Target device.

    Example:
        >>> pipeline = DetSegPipeline(
        ...     detector="flashdet-m",
        ...     segmentor="flashseg-m",
        ... )
        >>> results = pipeline("image.jpg")
    """

    def __init__(
        self,
        detector: Union[str, Any] = "flashdet-m",
        segmentor: Union[str, Any] = "flashseg-m",
        det_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        fusion_mode: str = "parallel",
        device: str = "auto",
    ):
        self.detector = detector
        self.segmentor = segmentor
        self.det_threshold = det_threshold
        self.mask_threshold = mask_threshold
        self.fusion_mode = fusion_mode
        self.device = self._resolve_device(device)

    def __call__(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Run the detection + segmentation pipeline.

        Args:
            source: Input image path or array.

        Returns:
            Dictionary with 'detections' (boxes + labels) and 'masks'.
        """
        return self.predict(source)

    def predict(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Run full pipeline prediction.

        Args:
            source: Input image.

        Returns:
            Combined detection + segmentation results.
        """
        image = self._load_image(source)

        if self.fusion_mode == "parallel":
            return self._parallel_fusion(image)
        elif self.fusion_mode == "det_guided":
            return self._det_guided_fusion(image)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def _parallel_fusion(self, image: np.ndarray) -> Dict[str, Any]:
        """Run detection and segmentation in parallel, then fuse.

        Args:
            image: Input image array.

        Returns:
            Fused results with detections and associated masks.
        """
        detections = self._run_detection(image)
        seg_masks = self._run_segmentation(image)

        results = {
            "detections": [d for d in detections if d["score"] >= self.det_threshold],
            "masks": seg_masks,
            "fused_objects": self._associate_masks(detections, seg_masks),
        }
        return results

    def _det_guided_fusion(self, image: np.ndarray) -> Dict[str, Any]:
        """Use detection boxes to guide segmentation.

        Args:
            image: Input image array.

        Returns:
            Results with detection-guided instance masks.
        """
        detections = self._run_detection(image)
        filtered_dets = [d for d in detections if d["score"] >= self.det_threshold]

        instance_masks = []
        for det in filtered_dets:
            crop = self._crop_region(image, det["bbox"])
            mask = self._run_segmentation(crop)
            instance_masks.append({
                "bbox": det["bbox"],
                "label": det.get("label", 0),
                "score": det["score"],
                "mask": mask,
            })

        return {"detections": filtered_dets, "instance_masks": instance_masks}

    def _associate_masks(
        self,
        detections: List[Dict[str, Any]],
        masks: Any,
    ) -> List[Dict[str, Any]]:
        """Associate segmentation masks with detected objects."""
        fused = []
        for det in detections:
            if det["score"] < self.det_threshold:
                continue
            fused.append({
                "bbox": det["bbox"],
                "label": det.get("label", 0),
                "score": det["score"],
                "has_mask": masks is not None,
            })
        return fused

    def _run_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection model."""
        import cv2

        if not hasattr(self, "_detector_model") or self._detector_model is None:
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
                detections.append({
                    "bbox": box.tolist(),
                    "score": float(scores[i]),
                    "label": int(labels[i]) if i < len(labels) else 0,
                })

        return detections

    def _run_segmentation(self, image: np.ndarray) -> Any:
        """Run segmentation model."""
        import cv2

        if not hasattr(self, "_segmentor_model") or self._segmentor_model is None:
            self._segmentor_model = self._load_torch_model(self.segmentor)

        h, w = image.shape[:2]
        resized = cv2.resize(image, (320, 320))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._segmentor_model(tensor)

        if isinstance(output, dict) and "masks" in output:
            masks = output["masks"]
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            return masks
        elif isinstance(output, torch.Tensor):
            mask = (output.squeeze(0) > self.mask_threshold).cpu().numpy()
            return mask

        return None

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

        raise ValueError(
            f"Cannot load model '{model_id}'. Provide a valid .pt/.pth path or nn.Module instance."
        )

    def _crop_region(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop image region from bounding box."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from source."""
        if isinstance(source, np.ndarray):
            return source
        import cv2
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {source}")
        return img

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return (
            f"DetSegPipeline(detector='{self.detector}', "
            f"segmentor='{self.segmentor}', "
            f"fusion_mode='{self.fusion_mode}')"
        )
