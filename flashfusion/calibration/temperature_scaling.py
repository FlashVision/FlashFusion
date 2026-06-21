"""Temperature Scaling for post-hoc model calibration.

Learns a single scalar temperature parameter that rescales logits to
produce well-calibrated probability estimates. Minimizes NLL on a
held-out validation set.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """Temperature Scaling calibration.

    Learns a single temperature T such that softmax(logits / T) produces
    calibrated probabilities. Optimized via NLL loss on validation data.

    Args:
        init_temperature: Initial temperature value.
        max_iter: Maximum LBFGS optimization iterations.
        lr: Learning rate for optimization.

    Example:
        >>> ts = TemperatureScaling()
        >>> ts.fit(val_logits, val_labels)
        >>> calibrated_probs = ts.calibrate(test_logits)
        >>> ece = ts.expected_calibration_error(calibrated_probs, test_labels)
    """

    def __init__(
        self,
        init_temperature: float = 1.5,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.max_iter = max_iter
        self.lr = lr
        self._fitted = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Learn temperature by minimizing NLL on validation data.

        Args:
            logits: Validation logits of shape (N, C).
            labels: Validation labels of shape (N,).

        Returns:
            Dictionary with 'temperature', 'nll_before', 'nll_after', 'ece_before', 'ece_after'.
        """
        logits = logits.detach().float()
        labels = labels.detach().long()

        nll_before = F.cross_entropy(logits, labels).item()
        probs_before = F.softmax(logits, dim=-1)
        ece_before = self._compute_ece(probs_before, labels).item()

        self.temperature.data = torch.tensor(1.5)

        optimizer = LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature.data = self.temperature.data.clamp(min=0.01, max=100.0)
        self._fitted = True

        with torch.no_grad():
            scaled = logits / self.temperature
            nll_after = F.cross_entropy(scaled, labels).item()
            probs_after = F.softmax(scaled, dim=-1)
            ece_after = self._compute_ece(probs_after, labels).item()

        return {
            "temperature": self.temperature.item(),
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply learned temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw model logits of shape (N, C).

        Returns:
            Calibrated probability distribution of shape (N, C).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        with torch.no_grad():
            return F.softmax(logits / self.temperature, dim=-1)

    @staticmethod
    def expected_calibration_error(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> torch.Tensor:
        """Compute Expected Calibration Error (ECE).

        Args:
            probs: Predicted probabilities of shape (N, C).
            labels: True labels of shape (N,).
            n_bins: Number of confidence bins.

        Returns:
            Scalar ECE value.
        """
        return TemperatureScaling._compute_ece(probs, labels, n_bins)

    @staticmethod
    def _compute_ece(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> torch.Tensor:
        """Internal ECE computation."""
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * (avg_confidence - avg_accuracy).abs()

        return ece.squeeze()

    @staticmethod
    def reliability_diagram(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """Compute data for a reliability diagram.

        Args:
            probs: Predicted probabilities of shape (N, C).
            labels: True labels of shape (N,).
            n_bins: Number of confidence bins.

        Returns:
            Dictionary with 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'.
        """
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = torch.zeros(n_bins)
        bin_confidences = torch.zeros(n_bins)
        bin_counts = torch.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_counts[i] = in_bin.sum()
            if bin_counts[i] > 0:
                bin_accuracies[i] = accuracies[in_bin].mean()
                bin_confidences[i] = confidences[in_bin].mean()

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }

    def __repr__(self) -> str:
        return f"TemperatureScaling(T={self.temperature.item():.4f}, fitted={self._fitted})"
