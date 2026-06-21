"""ConsistencyLoss — Enforce agreement between multiple model outputs.

Penalizes disagreement between models in an ensemble to encourage consistent
predictions across the fusion pipeline.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Consistency loss for multi-model fusion.

    Measures pairwise divergence between model outputs and penalizes
    disagreement. Supports both detection score distributions and
    feature-level alignment.

    Args:
        method: Divergence measure ('kl', 'mse', 'cosine').
        temperature: Temperature for softening score distributions.
        reduction: Reduction method ('mean', 'sum').

    Example:
        >>> loss_fn = ConsistencyLoss(method="kl", temperature=2.0)
        >>> outputs = [model_a(x), model_b(x), model_c(x)]
        >>> loss = loss_fn(outputs)
    """

    def __init__(
        self,
        method: str = "kl",
        temperature: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.method = method
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, model_outputs: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute consistency loss across model outputs.

        Args:
            model_outputs: List of output dicts from each model. Each should
                           contain 'scores' or 'features' tensors.

        Returns:
            Scalar consistency loss tensor.
        """
        if len(model_outputs) < 2:
            return torch.tensor(0.0, requires_grad=True)

        scores_list = self._extract_scores(model_outputs)
        if not scores_list:
            features_list = self._extract_features(model_outputs)
            if features_list:
                return self._feature_consistency(features_list)
            return torch.tensor(0.0, requires_grad=True)

        return self._score_consistency(scores_list)

    def _score_consistency(self, scores_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute pairwise consistency loss on score distributions."""
        losses = []
        for i in range(len(scores_list)):
            for j in range(i + 1, len(scores_list)):
                loss_ij = self._pairwise_divergence(scores_list[i], scores_list[j])
                losses.append(loss_ij)

        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        total = torch.stack(losses)
        return total.mean() if self.reduction == "mean" else total.sum()

    def _feature_consistency(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss on intermediate features."""
        losses = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                fi = features_list[i]
                fj = features_list[j]

                min_size = min(fi.shape[-1], fj.shape[-1])
                fi_flat = fi.view(fi.shape[0], -1)[:, :min_size]
                fj_flat = fj.view(fj.shape[0], -1)[:, :min_size]

                if self.method == "cosine":
                    sim = F.cosine_similarity(fi_flat, fj_flat, dim=-1)
                    losses.append((1.0 - sim).mean())
                else:
                    losses.append(F.mse_loss(fi_flat, fj_flat))

        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        total = torch.stack(losses)
        return total.mean() if self.reduction == "mean" else total.sum()

    def _pairwise_divergence(self, scores_a: torch.Tensor, scores_b: torch.Tensor) -> torch.Tensor:
        """Compute divergence between two score tensors."""
        # Align shapes
        min_len = min(scores_a.shape[0], scores_b.shape[0])
        a = scores_a[:min_len]
        b = scores_b[:min_len]

        if a.dim() == 1:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)

        if self.method == "kl":
            log_p = F.log_softmax(a / self.temperature, dim=-1)
            q = F.softmax(b / self.temperature, dim=-1)
            kl_ab = F.kl_div(log_p, q, reduction="batchmean")

            log_q = F.log_softmax(b / self.temperature, dim=-1)
            p = F.softmax(a / self.temperature, dim=-1)
            kl_ba = F.kl_div(log_q, p, reduction="batchmean")

            return (kl_ab + kl_ba) / 2.0 * (self.temperature ** 2)

        elif self.method == "mse":
            return F.mse_loss(a, b)

        elif self.method == "cosine":
            sim = F.cosine_similarity(a, b, dim=-1)
            return (1.0 - sim).mean()

        else:
            raise ValueError(f"Unknown consistency method: {self.method}")

    @staticmethod
    def _extract_scores(model_outputs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Extract score tensors from model outputs."""
        scores = []
        for out in model_outputs:
            s = out.get("scores")
            if isinstance(s, torch.Tensor) and s.numel() > 0:
                scores.append(s)
        return scores

    @staticmethod
    def _extract_features(model_outputs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Extract feature tensors from model outputs."""
        features = []
        for out in model_outputs:
            f = out.get("features")
            if isinstance(f, torch.Tensor) and f.numel() > 0:
                features.append(f)
        return features

    def __repr__(self) -> str:
        return (
            f"ConsistencyLoss(method='{self.method}', "
            f"temperature={self.temperature}, reduction='{self.reduction}')"
        )
