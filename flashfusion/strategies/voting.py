"""Voting ensemble strategy for classification fusion.

Supports majority (hard) voting and soft voting (probability averaging)
across multiple classification models.
"""

from typing import Any, Dict, List, Optional

import torch

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("voting")
class VotingEnsemble:
    """Voting ensemble for classification tasks.

    Combines classification outputs from multiple models using either
    hard voting (majority) or soft voting (probability averaging).

    Args:
        weights: Per-model weights for soft voting. None for uniform.
        mode: Voting mode ('soft' or 'hard').
        temperature: Temperature scaling for softmax in soft voting.

    Example:
        >>> voting = VotingEnsemble(weights=[0.5, 0.3, 0.2], mode="soft")
        >>> fused = voting.fuse(model_outputs)
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        mode: str = "soft",
        temperature: float = 1.0,
    ):
        self.weights = weights
        self.mode = mode
        self.temperature = temperature

    def fuse(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fuse classification predictions using voting.

        Args:
            predictions: List of prediction dicts from each model, containing
                'logits' or 'probabilities' of shape (B, num_classes).
            weights: Override per-model weights.

        Returns:
            Fused prediction with 'probabilities', 'labels', and 'scores'.
        """
        model_weights = weights or self.weights
        if model_weights is None:
            model_weights = [1.0] * len(predictions)

        if self.mode == "soft":
            return self._soft_vote(predictions, model_weights)
        else:
            return self._hard_vote(predictions, model_weights)

    def _soft_vote(
        self,
        predictions: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Soft voting: weighted average of probability distributions."""
        probs_list = []
        for pred in predictions:
            if "probabilities" in pred:
                probs = self._to_tensor(pred["probabilities"])
            elif "logits" in pred:
                logits = self._to_tensor(pred["logits"])
                probs = torch.softmax(logits / self.temperature, dim=-1)
            else:
                raise KeyError("Predictions must contain 'logits' or 'probabilities'")
            probs_list.append(probs)

        # Weighted average
        total_weight = sum(weights)
        fused_probs = torch.zeros_like(probs_list[0])
        for prob, w in zip(probs_list, weights):
            fused_probs += prob * (w / total_weight)

        scores, labels = torch.max(fused_probs, dim=-1)

        return {
            "probabilities": fused_probs,
            "labels": labels,
            "scores": scores,
        }

    def _hard_vote(
        self,
        predictions: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Hard voting: weighted majority vote on predicted classes."""
        votes_list = []
        for pred in predictions:
            if "labels" in pred:
                labels = self._to_tensor(pred["labels"]).long()
            elif "logits" in pred:
                logits = self._to_tensor(pred["logits"])
                labels = torch.argmax(logits, dim=-1)
            elif "probabilities" in pred:
                probs = self._to_tensor(pred["probabilities"])
                labels = torch.argmax(probs, dim=-1)
            else:
                raise KeyError("Predictions must contain 'labels', 'logits', or 'probabilities'")
            votes_list.append(labels)

        # Count weighted votes
        num_samples = votes_list[0].shape[0]
        num_classes = max(v.max().item() for v in votes_list) + 1

        vote_counts = torch.zeros(num_samples, num_classes)
        for vote, w in zip(votes_list, weights):
            for i in range(num_samples):
                vote_counts[i, vote[i].item()] += w

        scores, labels = torch.max(vote_counts, dim=-1)
        scores = scores / sum(weights)

        return {
            "labels": labels,
            "scores": scores,
            "vote_counts": vote_counts,
        }

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        """Convert data to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(data)

    def __repr__(self) -> str:
        return f"VotingEnsemble(mode='{self.mode}', weights={self.weights})"
