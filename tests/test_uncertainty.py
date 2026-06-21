"""Tests for uncertainty estimation methods."""

import pytest
import torch
import torch.nn as nn

from flashfusion.uncertainty import MCDropout, DeepEnsemble, EntropyEstimator


def _make_dropout_model():
    """Create a model with dropout layers."""
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(50, 10),
    )


def _make_ensemble_models(n_models=3):
    """Create multiple independent models for deep ensemble."""
    models = []
    for _ in range(n_models):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        models.append(model)
    return models


class TestMCDropout:
    """Tests for MC Dropout uncertainty estimation."""

    def test_basic_estimate(self):
        model = _make_dropout_model()
        mc = MCDropout(n_samples=10)
        inputs = torch.randn(4, 10)
        result = mc.estimate(model, inputs)
        assert "mean" in result
        assert "variance" in result
        assert "uncertainty" in result
        assert "entropy" in result
        assert "mutual_information" in result
        assert result["mean"].shape == (4, 10)
        assert result["uncertainty"].shape == (4,)

    def test_predictions_shape(self):
        model = _make_dropout_model()
        mc = MCDropout(n_samples=20)
        inputs = torch.randn(2, 10)
        result = mc.estimate(model, inputs)
        assert result["predictions"].shape == (20, 2, 10)

    def test_variance_nonzero(self):
        model = _make_dropout_model()
        mc = MCDropout(n_samples=30)
        inputs = torch.randn(4, 10)
        result = mc.estimate(model, inputs)
        assert result["variance"].sum() > 0

    def test_confidence_interval(self):
        model = _make_dropout_model()
        mc = MCDropout(n_samples=50)
        inputs = torch.randn(4, 10)
        result = mc.estimate(model, inputs)
        lower, upper = MCDropout.get_confidence_interval(result["predictions"])
        assert lower.shape == upper.shape
        assert (upper >= lower).all()

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be >= 2"):
            MCDropout(n_samples=1)

    def test_custom_dropout_rate(self):
        model = _make_dropout_model()
        mc = MCDropout(n_samples=10, dropout_rate=0.5)
        inputs = torch.randn(2, 10)
        result = mc.estimate(model, inputs)
        assert result["mean"].shape == (2, 10)

    def test_repr(self):
        mc = MCDropout(n_samples=30)
        r = repr(mc)
        assert "MCDropout" in r
        assert "30" in r


class TestDeepEnsemble:
    """Tests for Deep Ensemble uncertainty estimation."""

    def test_basic_estimate(self):
        models = _make_ensemble_models(3)
        ensemble = DeepEnsemble(models=models)
        inputs = torch.randn(4, 10)
        result = ensemble.estimate(inputs)
        assert "mean" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert "total_uncertainty" in result
        assert "probabilities" in result
        assert "labels" in result
        assert "scores" in result

    def test_output_shapes(self):
        models = _make_ensemble_models(5)
        ensemble = DeepEnsemble(models=models)
        inputs = torch.randn(4, 10)
        result = ensemble.estimate(inputs)
        assert result["mean"].shape == (4, 10)
        assert result["epistemic_uncertainty"].shape == (4,)
        assert result["labels"].shape == (4,)

    def test_add_model(self):
        ensemble = DeepEnsemble()
        models = _make_ensemble_models(3)
        for m in models:
            ensemble.add_model(m)
        assert len(ensemble.models) == 3

    def test_requires_two_models(self):
        models = _make_ensemble_models(1)
        ensemble = DeepEnsemble(models=models)
        with pytest.raises(ValueError, match="at least 2"):
            ensemble.estimate(torch.randn(2, 10))

    def test_diversity_score(self):
        models = _make_ensemble_models(3)
        inputs = torch.randn(10, 10)
        predictions = []
        for m in models:
            m.eval()
            with torch.no_grad():
                predictions.append(m(inputs))
        diversity = DeepEnsemble.diversity_score(predictions)
        assert 0.0 <= diversity <= 1.0

    def test_predict(self):
        models = _make_ensemble_models(3)
        ensemble = DeepEnsemble(models=models)
        inputs = torch.randn(4, 10)
        result = ensemble.predict(inputs)
        assert "probabilities" in result
        assert "uncertainty" in result

    def test_repr(self):
        models = _make_ensemble_models(3)
        ensemble = DeepEnsemble(models=models)
        r = repr(ensemble)
        assert "DeepEnsemble" in r
        assert "3" in r


class TestEntropyEstimator:
    """Tests for entropy-based uncertainty estimation."""

    def test_basic_estimate(self):
        estimator = EntropyEstimator()
        logits = torch.randn(8, 10)
        result = estimator.estimate(logits)
        assert "entropy" in result
        assert "max_prob" in result
        assert "margin" in result
        assert "is_uncertain" in result
        assert "labels" in result

    def test_entropy_nonnegative(self):
        estimator = EntropyEstimator()
        logits = torch.randn(8, 10)
        result = estimator.estimate(logits)
        assert (result["entropy"] >= 0).all()

    def test_confident_prediction(self):
        estimator = EntropyEstimator(threshold=0.5)
        logits = torch.zeros(4, 10)
        logits[:, 0] = 100.0  # Very confident
        result = estimator.estimate(logits)
        assert not result["is_uncertain"].any()

    def test_uncertain_prediction(self):
        estimator = EntropyEstimator(threshold=0.1)
        logits = torch.ones(4, 10)  # Uniform → maximum entropy
        result = estimator.estimate(logits)
        assert result["is_uncertain"].all()

    def test_filter_uncertain(self):
        estimator = EntropyEstimator(threshold=0.5)
        logits = torch.randn(20, 10)
        result = estimator.filter_uncertain(logits, return_indices=True)
        assert "confident_logits" in result
        assert "uncertain_logits" in result
        total = result["n_confident"] + result["n_uncertain"]
        assert total == 20

    def test_mutual_information(self):
        probs_samples = torch.softmax(torch.randn(10, 8, 5), dim=-1)
        mi = EntropyEstimator.mutual_information(probs_samples)
        assert mi.shape == (8,)
        assert (mi >= -1e-5).all()

    def test_repr(self):
        estimator = EntropyEstimator(threshold=0.3)
        r = repr(estimator)
        assert "EntropyEstimator" in r
        assert "0.3" in r
