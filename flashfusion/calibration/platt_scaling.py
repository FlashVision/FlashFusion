"""Platt Scaling for post-hoc calibration using logistic regression.

Fits a logistic regression model on validation logits to produce
calibrated probability estimates. Extends naturally to multi-class
via per-class binary calibration or matrix scaling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlattScaling(nn.Module):
    """Platt Scaling calibration via logistic regression on logits.

    Learns affine parameters (weight, bias) per class to transform
    logits into calibrated probabilities: p = sigmoid(a * logit + b).

    Args:
        num_classes: Number of output classes. None = auto-detect from data.
        max_iter: Maximum optimization iterations.
        lr: Learning rate for SGD optimizer.
        reg_lambda: L2 regularization strength.

    Example:
        >>> platt = PlattScaling()
        >>> platt.fit(val_logits, val_labels)
        >>> calibrated = platt.calibrate(test_logits)
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        max_iter: int = 100,
        lr: float = 0.01,
        reg_lambda: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self._fitted = False

        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None

    def _init_parameters(self, num_classes: int) -> None:
        """Initialize learnable affine parameters."""
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling transform to logits."""
        if self.weight is None or self.bias is None:
            raise RuntimeError("Must call fit() before forward()")
        return logits * self.weight + self.bias

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Fit Platt scaling parameters on validation data.

        Uses cross-entropy loss with L2 regularization to learn
        per-class affine transformations of the logits.

        Args:
            logits: Validation logits of shape (N, C).
            labels: Validation labels of shape (N,).

        Returns:
            Dictionary with training metrics.
        """
        logits = logits.detach().float()
        labels = labels.detach().long()
        num_classes = logits.shape[1]

        self._init_parameters(num_classes)

        nll_before = F.cross_entropy(logits, labels).item()

        optimizer = torch.optim.Adam(
            [self.weight, self.bias], lr=self.lr, weight_decay=0.0
        )

        best_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            scaled_logits = logits * self.weight + self.bias
            loss = F.cross_entropy(scaled_logits, labels)
            reg = self.reg_lambda * (
                (self.weight - 1.0).pow(2).sum() + self.bias.pow(2).sum()
            )
            total_loss = loss + reg
            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss - 1e-6:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self._fitted = True

        with torch.no_grad():
            scaled = logits * self.weight + self.bias
            nll_after = F.cross_entropy(scaled, labels).item()

        return {
            "nll_before": nll_before,
            "nll_after": nll_after,
            "weight_mean": self.weight.data.mean().item(),
            "bias_mean": self.bias.data.mean().item(),
            "epochs": epoch + 1,
        }

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply fitted Platt scaling and return calibrated probabilities.

        Args:
            logits: Raw model logits of shape (N, C).

        Returns:
            Calibrated probability distribution of shape (N, C).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        with torch.no_grad():
            scaled = logits * self.weight + self.bias
            return F.softmax(scaled, dim=-1)

    def calibrate_binary(self, logits: torch.Tensor) -> torch.Tensor:
        """Binary calibration for single-column logits.

        Args:
            logits: Raw scores of shape (N,) or (N, 1).

        Returns:
            Calibrated probabilities of shape (N,).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate_binary()")
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        with torch.no_grad():
            scaled = logits * self.weight[0] + self.bias[0]
            return torch.sigmoid(scaled.squeeze(1))

    def __repr__(self) -> str:
        w_str = f"{self.weight.data.mean().item():.4f}" if self.weight is not None else "N/A"
        b_str = f"{self.bias.data.mean().item():.4f}" if self.bias is not None else "N/A"
        return f"PlattScaling(weight_mean={w_str}, bias_mean={b_str}, fitted={self._fitted})"
