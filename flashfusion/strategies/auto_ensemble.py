"""Auto-ensemble selection: automatically choose the best model subset.

Implements greedy forward selection and cross-validation-based model
selection for building optimal ensembles.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("auto_ensemble")
class AutoEnsembleSelection:
    """Automatic ensemble selection using greedy search.

    Selects an optimal subset of models from a candidate pool
    that maximizes a given performance metric.

    Args:
        max_models: Maximum number of models in the ensemble.
        metric_fn: Callable that evaluates ensemble performance.
            Signature: metric_fn(predictions, targets) -> float (higher=better).
        with_replacement: Allow selecting the same model multiple times.
        n_folds: Number of cross-validation folds for model selection.
        patience: Stop adding models after this many rounds without improvement.

    Example:
        >>> selector = AutoEnsembleSelection(max_models=5, metric_fn=accuracy_fn)
        >>> selected_indices, weights = selector.select(candidate_predictions, targets)
    """

    def __init__(
        self,
        max_models: int = 10,
        metric_fn: Optional[Callable] = None,
        with_replacement: bool = True,
        n_folds: int = 5,
        patience: int = 3,
    ) -> None:
        self.max_models = max_models
        self.metric_fn = metric_fn or self._default_accuracy
        self.with_replacement = with_replacement
        self.n_folds = n_folds
        self.patience = patience
        self._selected_indices: List[int] = []
        self._best_score: float = -float("inf")

    def select(
        self,
        candidate_predictions: List[torch.Tensor],
        targets: torch.Tensor,
        val_predictions: Optional[List[torch.Tensor]] = None,
        val_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[float]]:
        """Greedily select models to maximize ensemble performance.

        Args:
            candidate_predictions: List of prediction tensors (B, C) from each candidate.
            targets: Ground truth labels of shape (B,).
            val_predictions: Optional validation predictions for early stopping.
            val_targets: Optional validation targets.

        Returns:
            Tuple of (selected_model_indices, per-model_weights).
        """
        n_candidates = len(candidate_predictions)
        selected: List[int] = []
        best_score = -float("inf")
        no_improve_count = 0

        for _ in range(self.max_models):
            best_candidate = -1
            best_candidate_score = -float("inf")

            for idx in range(n_candidates):
                if not self.with_replacement and idx in selected:
                    continue

                trial_ensemble = selected + [idx]
                ensemble_pred = self._compute_ensemble_prediction(candidate_predictions, trial_ensemble)
                score = self.metric_fn(ensemble_pred, targets)

                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = idx

            if best_candidate == -1:
                break

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                break

            selected.append(best_candidate)

        self._selected_indices = selected
        self._best_score = best_score

        weights = self._compute_weights(selected)
        return selected, weights

    def select_with_cv(
        self,
        candidate_predictions: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[List[int], List[float]]:
        """Select models using cross-validation for robust selection.

        Splits data into folds, runs greedy selection on each fold's
        training split, and combines results.

        Args:
            candidate_predictions: List of predictions from each candidate.
            targets: Ground truth labels.

        Returns:
            Tuple of (selected_model_indices, per-model_weights).
        """
        n_samples = targets.shape[0]
        indices = torch.randperm(n_samples)
        fold_size = n_samples // self.n_folds

        model_counts: Dict[int, int] = {}

        for fold in range(self.n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n_samples
            indices[val_start:val_end]
            train_idx = torch.cat([indices[:val_start], indices[val_end:]])

            fold_preds = [p[train_idx] for p in candidate_predictions]
            fold_targets = targets[train_idx]

            selected, _ = self.select(fold_preds, fold_targets)

            for model_idx in selected:
                model_counts[model_idx] = model_counts.get(model_idx, 0) + 1

        sorted_models = sorted(model_counts.items(), key=lambda x: -x[1])
        threshold = max(1, self.n_folds // 2)
        final_selected = [idx for idx, count in sorted_models if count >= threshold]

        if not final_selected:
            final_selected = [sorted_models[0][0]] if sorted_models else [0]

        final_selected = final_selected[: self.max_models]
        weights = self._compute_weights(final_selected)

        self._selected_indices = final_selected
        return final_selected, weights

    def _compute_ensemble_prediction(
        self,
        predictions: List[torch.Tensor],
        indices: List[int],
    ) -> torch.Tensor:
        """Compute averaged ensemble prediction from selected models."""
        selected_preds = torch.stack([predictions[i] for i in indices], dim=0)
        return selected_preds.mean(dim=0)

    def _compute_weights(self, selected: List[int]) -> List[float]:
        """Compute uniform weights for selected models."""
        if not selected:
            return []
        from collections import Counter

        counts = Counter(selected)
        total = len(selected)
        unique_models = sorted(set(selected))
        return [counts[idx] / total for idx in unique_models]

    @staticmethod
    def _default_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Default metric: classification accuracy."""
        if predictions.dim() == 1:
            pred_labels = (predictions > 0.5).long()
        else:
            pred_labels = predictions.argmax(dim=-1)
        return (pred_labels == targets).float().mean().item()

    def fuse(self, predictions: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Strategy interface: fuse predictions using selected ensemble.

        If selection hasn't been run, uses all models equally.
        """
        logits_list = []
        for pred in predictions:
            if "logits" in pred:
                logits_list.append(pred["logits"])
            elif "probabilities" in pred:
                logits_list.append(pred["probabilities"])

        if not logits_list:
            return {"predictions": torch.tensor([])}

        if self._selected_indices:
            valid_indices = [i for i in self._selected_indices if i < len(logits_list)]
            if valid_indices:
                selected = [logits_list[i] for i in valid_indices]
            else:
                selected = logits_list
        else:
            selected = logits_list

        stacked = torch.stack(selected, dim=0)
        fused = stacked.mean(dim=0)
        probs = torch.softmax(fused, dim=-1)
        scores, labels = torch.max(probs, dim=-1)

        return {
            "probabilities": probs,
            "labels": labels,
            "scores": scores,
            "selected_models": self._selected_indices,
        }

    @property
    def selected_indices(self) -> List[int]:
        """Return indices of selected models."""
        return self._selected_indices

    @property
    def best_score(self) -> float:
        """Return best ensemble score achieved."""
        return self._best_score

    def __repr__(self) -> str:
        return (
            f"AutoEnsembleSelection(max_models={self.max_models}, "
            f"with_replacement={self.with_replacement}, n_folds={self.n_folds})"
        )
