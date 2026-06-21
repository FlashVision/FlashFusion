"""Deep Ensembles for uncertainty estimation.

Combines predictions from independently trained models to estimate
both aleatoric and epistemic uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepEnsemble:
    """Deep Ensemble uncertainty estimation.

    Uses disagreement between independently trained ensemble members
    to quantify predictive uncertainty.

    Args:
        models: List of trained ensemble member models.
        temperature: Temperature for softmax scaling.

    Example:
        >>> ensemble = DeepEnsemble(models=[model_1, model_2, model_3])
        >>> result = ensemble.estimate(input_tensor)
        >>> print(result['epistemic_uncertainty'])
    """

    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        temperature: float = 1.0,
    ) -> None:
        self.models = models or []
        self.temperature = temperature

    def add_model(self, model: nn.Module) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)

    def estimate(
        self,
        inputs: torch.Tensor,
        return_individual: bool = False,
    ) -> Dict[str, Any]:
        """Estimate uncertainty from ensemble disagreement.

        Args:
            inputs: Input tensor.
            return_individual: Whether to return individual model predictions.

        Returns:
            Dictionary with:
                - 'mean': Mean prediction (B, C).
                - 'variance': Predictive variance (B, C).
                - 'epistemic_uncertainty': Epistemic uncertainty (B,).
                - 'aleatoric_uncertainty': Aleatoric uncertainty (B,).
                - 'total_uncertainty': Total predictive entropy (B,).
                - 'probabilities': Mean calibrated probabilities (B, C).
                - 'labels': Predicted labels (B,).
                - 'scores': Prediction confidence (B,).
        """
        if len(self.models) < 2:
            raise ValueError("Deep ensemble requires at least 2 models")

        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(inputs)
            predictions.append(output)

        stacked = torch.stack(predictions, dim=0)  # (M, B, C)
        mean_logits = stacked.mean(dim=0)  # (B, C)

        all_probs = F.softmax(stacked / self.temperature, dim=-1)  # (M, B, C)
        mean_probs = all_probs.mean(dim=0)  # (B, C)

        predictive_entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)

        per_model_entropy = -(all_probs * (all_probs + 1e-10).log()).sum(dim=-1)
        expected_entropy = per_model_entropy.mean(dim=0)

        epistemic = predictive_entropy - expected_entropy
        aleatoric = expected_entropy

        variance = stacked.var(dim=0)

        scores, labels = mean_probs.max(dim=-1)

        result = {
            "mean": mean_logits,
            "variance": variance,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": predictive_entropy,
            "probabilities": mean_probs,
            "labels": labels,
            "scores": scores,
        }

        if return_individual:
            result["individual_predictions"] = stacked

        return result

    def predict(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simple prediction without full uncertainty decomposition."""
        result = self.estimate(inputs)
        return {
            "probabilities": result["probabilities"],
            "labels": result["labels"],
            "scores": result["scores"],
            "uncertainty": result["total_uncertainty"],
        }

    @staticmethod
    def diversity_score(predictions: List[torch.Tensor]) -> float:
        """Compute ensemble diversity (pairwise disagreement).

        Higher diversity indicates more complementary ensemble members.

        Args:
            predictions: List of prediction tensors from each model.

        Returns:
            Average pairwise disagreement score.
        """
        n = len(predictions)
        if n < 2:
            return 0.0

        labels = [p.argmax(dim=-1) for p in predictions]
        disagreement = 0.0
        pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                disagreement += (labels[i] != labels[j]).float().mean().item()
                pairs += 1

        return disagreement / pairs if pairs > 0 else 0.0

    def __repr__(self) -> str:
        return f"DeepEnsemble(n_models={len(self.models)}, temperature={self.temperature})"
