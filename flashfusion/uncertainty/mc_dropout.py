"""Monte Carlo Dropout for uncertainty estimation.

Enables dropout at inference time and runs multiple forward passes
to approximate Bayesian posterior predictive uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout:
    """Monte Carlo Dropout uncertainty estimation.

    Performs multiple stochastic forward passes with dropout active
    to estimate predictive uncertainty via variance of outputs.

    Args:
        n_samples: Number of MC forward passes.
        dropout_rate: Override dropout probability (None = use model's existing rate).

    Example:
        >>> mc = MCDropout(n_samples=30)
        >>> result = mc.estimate(model, input_tensor)
        >>> print(result['uncertainty'])  # Per-sample uncertainty
    """

    def __init__(
        self,
        n_samples: int = 30,
        dropout_rate: Optional[float] = None,
    ) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 for uncertainty estimation")
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def estimate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run MC Dropout and compute uncertainty metrics.

        Args:
            model: Model with dropout layers.
            inputs: Input tensor.

        Returns:
            Dictionary with:
                - 'mean': Mean prediction across samples (B, C).
                - 'variance': Predictive variance (B, C).
                - 'uncertainty': Scalar uncertainty per sample (B,).
                - 'predictions': All stochastic predictions (n_samples, B, C).
                - 'entropy': Predictive entropy (B,).
                - 'mutual_information': Epistemic uncertainty via MI (B,).
        """
        self._enable_dropout(model)

        if self.dropout_rate is not None:
            self._set_dropout_rate(model, self.dropout_rate)

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = model(inputs)
                predictions.append(output)

        stacked = torch.stack(predictions, dim=0)  # (n_samples, B, ...)

        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        uncertainty = variance.mean(dim=-1) if variance.dim() > 1 else variance

        mean_probs = F.softmax(mean_pred, dim=-1)
        predictive_entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)

        per_sample_probs = F.softmax(stacked, dim=-1)
        per_sample_entropy = -(per_sample_probs * (per_sample_probs + 1e-10).log()).sum(dim=-1)
        expected_entropy = per_sample_entropy.mean(dim=0)
        mutual_information = predictive_entropy - expected_entropy

        return {
            "mean": mean_pred,
            "variance": variance,
            "uncertainty": uncertainty,
            "predictions": stacked,
            "entropy": predictive_entropy,
            "mutual_information": mutual_information,
        }

    @staticmethod
    def _enable_dropout(model: nn.Module) -> None:
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    @staticmethod
    def _set_dropout_rate(model: nn.Module, rate: float) -> None:
        """Override dropout probability in all dropout layers."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = rate

    @staticmethod
    def get_confidence_interval(
        predictions: torch.Tensor,
        confidence: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute confidence intervals from MC samples.

        Args:
            predictions: Stacked predictions (n_samples, B, ...).
            confidence: Confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound) tensors.
        """
        alpha = 1.0 - confidence
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        sorted_preds, _ = predictions.sort(dim=0)
        n = predictions.shape[0]

        lower_idx = max(0, int(n * lower_q))
        upper_idx = min(n - 1, int(n * upper_q))

        return sorted_preds[lower_idx], sorted_preds[upper_idx]

    def __repr__(self) -> str:
        return f"MCDropout(n_samples={self.n_samples}, dropout_rate={self.dropout_rate})"
