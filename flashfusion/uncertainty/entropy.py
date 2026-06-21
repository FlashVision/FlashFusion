"""Entropy-based uncertainty estimation.

Provides various information-theoretic measures for quantifying
prediction confidence and uncertainty from model outputs.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class EntropyEstimator:
    """Entropy-based uncertainty quantification.

    Computes predictive entropy, max probability, margin, and other
    information-theoretic uncertainty metrics from model outputs.

    Args:
        threshold: Uncertainty threshold for flagging low-confidence predictions.
        normalize: Whether to normalize entropy to [0, 1] range.

    Example:
        >>> estimator = EntropyEstimator(threshold=0.5)
        >>> result = estimator.estimate(logits)
        >>> uncertain_mask = result['is_uncertain']
    """

    def __init__(
        self,
        threshold: float = 0.5,
        normalize: bool = True,
    ) -> None:
        self.threshold = threshold
        self.normalize = normalize

    def estimate(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all entropy-based uncertainty metrics.

        Args:
            logits: Model output logits of shape (B, C).

        Returns:
            Dictionary with:
                - 'entropy': Shannon entropy of predictive distribution (B,).
                - 'max_prob': Maximum probability (B,).
                - 'margin': Difference between top-2 probabilities (B,).
                - 'least_confidence': 1 - max_prob (B,).
                - 'ratio': top1_prob / top2_prob (B,).
                - 'is_uncertain': Boolean mask for uncertain predictions (B,).
                - 'labels': Predicted labels (B,).
                - 'scores': Prediction confidence scores (B,).
        """
        probs = F.softmax(logits, dim=-1)

        entropy = self.predictive_entropy(probs)
        max_prob, labels = probs.max(dim=-1)
        margin = self.prediction_margin(probs)
        least_conf = 1.0 - max_prob

        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        ratio = sorted_probs[:, 0] / (sorted_probs[:, 1] + 1e-10)

        if self.normalize:
            num_classes = logits.shape[-1]
            max_entropy = torch.log(torch.tensor(float(num_classes)))
            normalized_entropy = entropy / max_entropy
            is_uncertain = normalized_entropy > self.threshold
        else:
            is_uncertain = entropy > self.threshold

        return {
            "entropy": entropy,
            "max_prob": max_prob,
            "margin": margin,
            "least_confidence": least_conf,
            "ratio": ratio,
            "is_uncertain": is_uncertain,
            "labels": labels,
            "scores": max_prob,
        }

    @staticmethod
    def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of probability distribution.

        Args:
            probs: Probability distribution of shape (B, C).

        Returns:
            Per-sample entropy of shape (B,).
        """
        return -(probs * (probs + 1e-10).log()).sum(dim=-1)

    @staticmethod
    def prediction_margin(probs: torch.Tensor) -> torch.Tensor:
        """Compute margin between top-2 class probabilities.

        Larger margin = more confident prediction.

        Args:
            probs: Probability distribution of shape (B, C).

        Returns:
            Per-sample margin of shape (B,).
        """
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        if sorted_probs.shape[-1] < 2:
            return sorted_probs[:, 0]
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    @staticmethod
    def conditional_entropy(
        probs_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional entropy (average entropy of individual predictions).

        Used for decomposing uncertainty into aleatoric component.

        Args:
            probs_samples: Probabilities from multiple samples (n_samples, B, C).

        Returns:
            Per-sample conditional entropy of shape (B,).
        """
        per_sample_entropy = -(probs_samples * (probs_samples + 1e-10).log()).sum(dim=-1)
        return per_sample_entropy.mean(dim=0)

    @staticmethod
    def mutual_information(
        probs_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mutual information (epistemic uncertainty).

        MI = H[y|x] - E_theta[H[y|x, theta]]
        Higher MI indicates more model disagreement / epistemic uncertainty.

        Args:
            probs_samples: Probabilities from multiple forward passes (n_samples, B, C).

        Returns:
            Per-sample mutual information of shape (B,).
        """
        mean_probs = probs_samples.mean(dim=0)
        total_entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)
        cond_entropy = EntropyEstimator.conditional_entropy(probs_samples)
        return total_entropy - cond_entropy

    def filter_uncertain(
        self,
        logits: torch.Tensor,
        return_indices: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Separate predictions into confident and uncertain sets.

        Args:
            logits: Model logits of shape (B, C).
            return_indices: Whether to return indices of each group.

        Returns:
            Dictionary with 'confident' and 'uncertain' predictions.
        """
        result = self.estimate(logits)
        mask = result["is_uncertain"]

        output = {
            "confident_logits": logits[~mask],
            "uncertain_logits": logits[mask],
            "confident_labels": result["labels"][~mask],
            "uncertain_labels": result["labels"][mask],
            "n_confident": (~mask).sum().item(),
            "n_uncertain": mask.sum().item(),
        }

        if return_indices:
            output["confident_indices"] = torch.where(~mask)[0]
            output["uncertain_indices"] = torch.where(mask)[0]

        return output

    def __repr__(self) -> str:
        return f"EntropyEstimator(threshold={self.threshold}, normalize={self.normalize})"
