"""Cascade fusion strategy with early exit.

Runs models sequentially from lightweight to heavyweight, exiting early
when prediction confidence exceeds the threshold, saving compute on
easy samples while maintaining accuracy on hard ones.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("cascade")
class CascadeFusion:
    """Cascade fusion with sequential model execution and early exit.

    Models are run in order from fastest/smallest to slowest/largest.
    If a model produces a prediction with sufficient confidence, the
    cascade exits early without running remaining models.

    Args:
        models: List of model identifiers or nn.Module instances (ordered light→heavy).
        confidence_thresholds: Per-model confidence thresholds for early exit.
            The last model always runs if reached.
        fallback_strategy: Strategy to combine results if no early exit occurs.
        device: Target device.

    Example:
        >>> cascade = CascadeFusion(
        ...     models=["flashdet-s", "flashdet-m", "flashdet-l"],
        ...     confidence_thresholds=[0.8, 0.6, 0.0],
        ... )
        >>> results = cascade.predict(image)
    """

    def __init__(
        self,
        models: Optional[List[Any]] = None,
        confidence_thresholds: Optional[List[float]] = None,
        fallback_strategy: str = "last",
        device: str = "auto",
    ):
        self.models = models or []
        self.confidence_thresholds = confidence_thresholds or [0.0] * len(self.models)
        self.fallback_strategy = fallback_strategy
        self.device = self._resolve_device(device)

        if len(self.confidence_thresholds) != len(self.models):
            raise ValueError(
                f"Number of thresholds ({len(self.confidence_thresholds)}) must match "
                f"number of models ({len(self.models)})"
            )

    def fuse(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fuse pre-computed predictions using cascade logic.

        Selects the first prediction that meets its confidence threshold.

        Args:
            predictions: List of prediction dicts from each model.
            weights: Unused (kept for API consistency).

        Returns:
            Selected prediction dictionary.
        """
        for i, (pred, threshold) in enumerate(zip(predictions, self.confidence_thresholds)):
            confidence = self._get_confidence(pred)
            if confidence >= threshold or i == len(predictions) - 1:
                pred["cascade_exit_stage"] = i
                pred["cascade_confidence"] = confidence
                return pred

        return predictions[-1]

    def predict(self, image: Any) -> Dict[str, Any]:
        """Run cascade prediction with early exit.

        Args:
            image: Input image (tensor, numpy array, or path).

        Returns:
            Prediction from the first model exceeding its confidence threshold.
        """
        for i, (model, threshold) in enumerate(
            zip(self.models, self.confidence_thresholds)
        ):
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    prediction = model(image)
            else:
                raise NotImplementedError(
                    f"String model IDs require model loading. Got: {model}"
                )

            confidence = self._get_confidence(prediction)
            if confidence >= threshold or i == len(self.models) - 1:
                prediction["cascade_exit_stage"] = i
                prediction["cascade_confidence"] = confidence
                return prediction

        return {"cascade_exit_stage": -1, "cascade_confidence": 0.0}

    def _get_confidence(self, prediction: Dict[str, Any]) -> float:
        """Extract confidence score from a prediction."""
        if "scores" in prediction:
            scores = prediction["scores"]
            if isinstance(scores, torch.Tensor):
                return scores.max().item() if scores.numel() > 0 else 0.0
            elif isinstance(scores, (list, tuple)) and scores:
                return max(scores)
        if "confidence" in prediction:
            return float(prediction["confidence"])
        return 0.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return (
            f"CascadeFusion(num_models={len(self.models)}, "
            f"thresholds={self.confidence_thresholds})"
        )
