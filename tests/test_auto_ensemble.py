"""Tests for auto-ensemble selection."""

import torch

from flashfusion.strategies.auto_ensemble import AutoEnsembleSelection


def _make_candidate_predictions(n_candidates=5, n_samples=50, n_classes=10):
    """Generate synthetic candidate model predictions."""
    predictions = []
    for _ in range(n_candidates):
        predictions.append(torch.randn(n_samples, n_classes))
    return predictions


class TestAutoEnsembleSelection:
    """Tests for AutoEnsembleSelection strategy."""

    def test_basic_selection(self):
        selector = AutoEnsembleSelection(max_models=3)
        preds = _make_candidate_predictions(5, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selected, weights = selector.select(preds, targets)
        assert len(selected) > 0
        assert len(selected) <= 3
        assert len(weights) > 0

    def test_respects_max_models(self):
        selector = AutoEnsembleSelection(max_models=2, patience=100)
        preds = _make_candidate_predictions(10, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selected, weights = selector.select(preds, targets)
        assert len(selected) <= 2

    def test_with_replacement(self):
        selector = AutoEnsembleSelection(max_models=5, with_replacement=True, patience=100)
        preds = _make_candidate_predictions(3, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selected, _ = selector.select(preds, targets)
        # With replacement, same model can appear multiple times
        assert len(selected) <= 5

    def test_without_replacement(self):
        selector = AutoEnsembleSelection(max_models=10, with_replacement=False, patience=100)
        preds = _make_candidate_predictions(5, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selected, _ = selector.select(preds, targets)
        assert len(selected) == len(set(selected))

    def test_cv_selection(self):
        selector = AutoEnsembleSelection(max_models=3, n_folds=3)
        preds = _make_candidate_predictions(5, 60, 10)
        targets = torch.randint(0, 10, (60,))
        selected, weights = selector.select_with_cv(preds, targets)
        assert len(selected) > 0
        assert len(weights) > 0

    def test_custom_metric(self):
        def top5_accuracy(preds, targets):
            _, top5 = preds.topk(5, dim=-1)
            correct = top5.eq(targets.unsqueeze(-1)).any(dim=-1)
            return correct.float().mean().item()

        selector = AutoEnsembleSelection(max_models=3, metric_fn=top5_accuracy)
        preds = _make_candidate_predictions(5, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selected, weights = selector.select(preds, targets)
        assert len(selected) > 0

    def test_best_score_property(self):
        selector = AutoEnsembleSelection(max_models=3)
        preds = _make_candidate_predictions(5, 50, 10)
        targets = torch.randint(0, 10, (50,))
        selector.select(preds, targets)
        assert selector.best_score > -float("inf")

    def test_fuse_interface(self):
        selector = AutoEnsembleSelection(max_models=3)
        predictions = [
            {"logits": torch.randn(4, 10)},
            {"logits": torch.randn(4, 10)},
            {"logits": torch.randn(4, 10)},
        ]
        result = selector.fuse(predictions)
        assert "probabilities" in result
        assert "labels" in result
        assert "scores" in result

    def test_repr(self):
        selector = AutoEnsembleSelection(max_models=5)
        r = repr(selector)
        assert "AutoEnsembleSelection" in r
        assert "5" in r
