"""Stacking ensemble strategy with meta-learner.

Uses a trained meta-learner (e.g., a small MLP or logistic regression)
to combine outputs from multiple base models into a final prediction.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flashfusion.registry import STRATEGIES


class MetaLearner(nn.Module):
    """Simple MLP meta-learner for stacking.

    Args:
        input_dim: Total input dimension (sum of base model output dims).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (number of classes or regression targets).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 80,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through meta-learner."""
        return self.network(x)


@STRATEGIES.register("stacking")
class StackingEnsemble:
    """Stacking ensemble with a trainable meta-learner.

    Base model outputs are concatenated and fed to a meta-learner
    that produces the final prediction.

    Args:
        meta_learner: Trained MetaLearner instance. If None, uses averaging.
        num_classes: Number of output classes.
        device: Target device.

    Example:
        >>> stacking = StackingEnsemble(num_classes=80)
        >>> stacking.fit(train_predictions, train_targets)
        >>> fused = stacking.fuse(model_outputs)
    """

    def __init__(
        self,
        meta_learner: Optional[nn.Module] = None,
        num_classes: int = 80,
        device: str = "auto",
    ):
        self.meta_learner = meta_learner
        self.num_classes = num_classes
        self.device = self._resolve_device(device)
        self._fitted = meta_learner is not None

    def fuse(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fuse predictions using the meta-learner.

        Args:
            predictions: List of prediction dicts from base models.
            weights: Unused (meta-learner handles weighting internally).

        Returns:
            Fused prediction dictionary.
        """
        features = self._extract_features(predictions)

        if self._fitted and self.meta_learner is not None:
            self.meta_learner.eval()
            self.meta_learner.to(self.device)
            with torch.no_grad():
                output = self.meta_learner(features.to(self.device))
            probs = torch.softmax(output, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            return {
                "probabilities": probs,
                "labels": labels,
                "scores": scores,
            }
        else:
            return self._average_fallback(predictions)

    def fit(
        self,
        train_predictions: List[List[Dict[str, Any]]],
        train_targets: List[Any],
        epochs: int = 50,
        learning_rate: float = 0.001,
    ) -> None:
        """Train the meta-learner on base model predictions.

        Args:
            train_predictions: For each sample, list of predictions from each model.
            train_targets: Ground truth targets.
            epochs: Number of training epochs.
            learning_rate: Learning rate for training.
        """
        features_list = []
        for sample_preds in train_predictions:
            feat = self._extract_features(sample_preds)
            features_list.append(feat)

        X = torch.cat(features_list, dim=0).to(self.device)
        y = torch.tensor(train_targets, dtype=torch.long).to(self.device)

        input_dim = X.shape[1]
        self.meta_learner = MetaLearner(
            input_dim=input_dim,
            output_dim=self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.meta_learner.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.meta_learner(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def _extract_features(self, predictions: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract and concatenate features from model predictions."""
        features = []
        for pred in predictions:
            if "logits" in pred:
                feat = self._to_tensor(pred["logits"])
            elif "probabilities" in pred:
                feat = self._to_tensor(pred["probabilities"])
            elif "scores" in pred:
                feat = self._to_tensor(pred["scores"])
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
            else:
                raise KeyError("Predictions must contain 'logits', 'probabilities', or 'scores'")

            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            features.append(feat)

        return torch.cat(features, dim=-1)

    def _average_fallback(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to simple averaging when meta-learner is not trained."""
        probs_list = []
        for pred in predictions:
            if "probabilities" in pred:
                probs_list.append(self._to_tensor(pred["probabilities"]))
            elif "logits" in pred:
                probs_list.append(torch.softmax(self._to_tensor(pred["logits"]), dim=-1))

        if probs_list:
            avg_probs = torch.stack(probs_list).mean(dim=0)
            scores, labels = torch.max(avg_probs, dim=-1)
            return {"probabilities": avg_probs, "labels": labels, "scores": scores}

        return {"labels": torch.tensor([0]), "scores": torch.tensor([0.0])}

    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        """Convert data to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(np.array(data), dtype=torch.float32)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return f"StackingEnsemble(fitted={self._fitted}, num_classes={self.num_classes})"
