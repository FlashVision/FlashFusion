"""FlashFusion — Main multi-model fusion model.

Wraps multiple base vision models and applies a fusion strategy to combine
their outputs into unified predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class FlashFusion(nn.Module):
    """Multi-model fusion module that combines outputs from multiple vision models.

    Args:
        models: List of model identifiers or nn.Module instances.
        strategy: Fusion strategy instance or name string.
        input_size: Input image size as (height, width).
        weights: Optional per-model weights for fusion.
        device: Target device for inference.

    Example:
        >>> from flashfusion import FlashFusion
        >>> from flashfusion.strategies import WeightedBoxFusion
        >>> model = FlashFusion(
        ...     models=["flashdet-m", "flashdet-l"],
        ...     strategy=WeightedBoxFusion(weights=[0.4, 0.6]),
        ...     input_size=(320, 320),
        ... )
        >>> results = model.predict("image.jpg")
    """

    def __init__(
        self,
        models: Union[List[str], List[nn.Module]],
        strategy: Any = None,
        input_size: Tuple[int, int] = (320, 320),
        weights: Optional[List[float]] = None,
        device: str = "auto",
    ):
        super().__init__()
        self.input_size = input_size
        self.device = self._resolve_device(device)
        self.weights = weights

        self._models = nn.ModuleList()
        self._model_names: List[str] = []
        self._strategy = strategy

        for model in models:
            if isinstance(model, nn.Module):
                self._models.append(model)
                self._model_names.append(model.__class__.__name__)
            elif isinstance(model, str):
                loaded = self._load_model(model)
                self._models.append(loaded)
                self._model_names.append(model)

    @classmethod
    def from_models(
        cls,
        model_paths: List[str],
        strategy: str = "wbf",
        weights: Optional[List[float]] = None,
        device: str = "auto",
        **kwargs,
    ) -> "FlashFusion":
        """Create FlashFusion from model file paths.

        Args:
            model_paths: List of paths to model weight files.
            strategy: Strategy name (wbf, voting, cascade).
            weights: Optional per-model weights.
            device: Target device.

        Returns:
            Configured FlashFusion instance.
        """
        from flashfusion.strategies import get_strategy

        strategy_instance = get_strategy(strategy, weights=weights)
        return cls(
            models=model_paths,
            strategy=strategy_instance,
            weights=weights,
            device=device,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run forward pass through all models and fuse results.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Dictionary with fused predictions.
        """
        all_outputs = []
        for model in self._models:
            with torch.no_grad():
                output = model(x)
            all_outputs.append(output)

        if self._strategy is not None:
            fused = self._strategy.fuse(all_outputs, weights=self.weights)
        else:
            fused = all_outputs[0]

        return fused

    def predict(self, source: Union[str, Path, torch.Tensor], **kwargs) -> List[Dict[str, Any]]:
        """Run prediction on an image source.

        Args:
            source: Image path, directory, or tensor.
            **kwargs: Additional prediction options.

        Returns:
            List of prediction result dictionaries.
        """
        import cv2

        self.eval()
        conf_threshold = kwargs.get("conf_threshold", 0.25)

        if isinstance(source, torch.Tensor):
            images_tensor = source if source.dim() == 4 else source.unsqueeze(0)
        else:
            path = Path(source)
            image_paths = []
            if path.is_file():
                image_paths = [path]
            elif path.is_dir():
                extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                for ext in extensions:
                    image_paths.extend(sorted(path.glob(f"*{ext}")))
            else:
                raise FileNotFoundError(f"Source not found: {path}")

            tensors = []
            h, w = self.input_size
            for img_path in image_paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                resized = cv2.resize(img, (w, h))
                t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                tensors.append(t)

            if not tensors:
                return []
            images_tensor = torch.stack(tensors)

        images_tensor = images_tensor.to(self.device)
        results = []
        with torch.no_grad():
            for i in range(images_tensor.shape[0]):
                output = self.forward(images_tensor[i : i + 1])
                result = {"raw_output": output}
                if isinstance(output, dict):
                    if "scores" in output:
                        scores = output["scores"]
                        mask = scores > conf_threshold if isinstance(scores, torch.Tensor) else None
                        if mask is not None:
                            result["scores"] = scores[mask].cpu()
                            if "boxes" in output:
                                result["boxes"] = output["boxes"][mask].cpu()
                            if "labels" in output:
                                result["labels"] = output["labels"][mask].cpu()
                        else:
                            result.update(output)
                    else:
                        result.update(output)
                results.append(result)

        return results

    def _load_model(self, model_id: str) -> nn.Module:
        """Load a model from identifier string or path.

        Args:
            model_id: Model identifier (e.g., 'flashdet-m') or file path.

        Returns:
            Loaded nn.Module instance.
        """
        path = Path(model_id)
        if path.exists() and path.suffix in (".pt", ".pth"):
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, nn.Module):
                return checkpoint
            if isinstance(checkpoint, dict):
                if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                    return checkpoint["model"]
                if "model_state_dict" in checkpoint:
                    model_cls = checkpoint.get("model_class", None)
                    if model_cls is not None and isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
                        model = model_cls()
                        model.load_state_dict(checkpoint["model_state_dict"])
                        return model
                    raise ValueError(
                        f"Checkpoint at {path} contains 'model_state_dict' but no "
                        "'model_class' to reconstruct the architecture. "
                        "Pass a pre-built nn.Module instance instead."
                    )
                if "state_dict" in checkpoint:
                    raise ValueError(
                        f"Checkpoint at {path} contains 'state_dict' but no model architecture. "
                        "Pass a pre-built nn.Module instance instead."
                    )
            raise ValueError(f"Checkpoint at {path} is not a valid nn.Module or recognized format")

        raise ValueError(
            f"Model '{model_id}' not found as a file path. "
            "Provide a valid .pt/.pth checkpoint path or pass nn.Module instances directly."
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def num_models(self) -> int:
        """Return the number of models in the fusion."""
        return len(self._models)

    @property
    def model_names(self) -> List[str]:
        """Return names of all models in the fusion."""
        return self._model_names

    def __repr__(self) -> str:
        return (
            f"FlashFusion(\n"
            f"  models={self._model_names},\n"
            f"  strategy={self._strategy},\n"
            f"  input_size={self.input_size},\n"
            f"  device={self.device}\n"
            f")"
        )
