"""Tests for calibration methods (Temperature Scaling, Platt Scaling)."""

import pytest
import torch

from flashfusion.calibration import TemperatureScaling, PlattScaling


def _make_logits_and_labels(n_samples=100, n_classes=10):
    """Generate synthetic logits and labels for calibration testing."""
    logits = torch.randn(n_samples, n_classes)
    labels = torch.randint(0, n_classes, (n_samples,))
    return logits, labels


class TestTemperatureScaling:
    """Tests for TemperatureScaling calibration."""

    def test_fit(self):
        ts = TemperatureScaling()
        logits, labels = _make_logits_and_labels()
        result = ts.fit(logits, labels)
        assert "temperature" in result
        assert "nll_before" in result
        assert "nll_after" in result
        assert "ece_before" in result
        assert "ece_after" in result
        assert result["temperature"] > 0

    def test_calibrate(self):
        ts = TemperatureScaling()
        logits, labels = _make_logits_and_labels()
        ts.fit(logits, labels)
        test_logits = torch.randn(20, 10)
        probs = ts.calibrate(test_logits)
        assert probs.shape == test_logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(20), atol=1e-5)

    def test_calibrate_before_fit_raises(self):
        ts = TemperatureScaling()
        with pytest.raises(RuntimeError, match="Must call fit"):
            ts.calibrate(torch.randn(5, 10))

    def test_ece_computation(self):
        logits, labels = _make_logits_and_labels()
        probs = torch.softmax(logits, dim=-1)
        ece = TemperatureScaling.expected_calibration_error(probs, labels)
        assert ece.item() >= 0
        assert ece.item() <= 1.0

    def test_reliability_diagram(self):
        logits, labels = _make_logits_and_labels()
        probs = torch.softmax(logits, dim=-1)
        diagram = TemperatureScaling.reliability_diagram(probs, labels)
        assert "bin_centers" in diagram
        assert "bin_accuracies" in diagram
        assert "bin_confidences" in diagram
        assert "bin_counts" in diagram
        assert diagram["bin_centers"].shape[0] == 15

    def test_nll_improves(self):
        ts = TemperatureScaling()
        logits, labels = _make_logits_and_labels(500)
        result = ts.fit(logits, labels)
        assert result["nll_after"] <= result["nll_before"] + 0.1

    def test_repr(self):
        ts = TemperatureScaling()
        r = repr(ts)
        assert "TemperatureScaling" in r


class TestPlattScaling:
    """Tests for PlattScaling calibration."""

    def test_fit(self):
        platt = PlattScaling()
        logits, labels = _make_logits_and_labels()
        result = platt.fit(logits, labels)
        assert "nll_before" in result
        assert "nll_after" in result
        assert "weight_mean" in result
        assert "bias_mean" in result

    def test_calibrate(self):
        platt = PlattScaling()
        logits, labels = _make_logits_and_labels()
        platt.fit(logits, labels)
        test_logits = torch.randn(20, 10)
        probs = platt.calibrate(test_logits)
        assert probs.shape == test_logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(20), atol=1e-5)

    def test_calibrate_before_fit_raises(self):
        platt = PlattScaling()
        with pytest.raises(RuntimeError, match="Must call fit"):
            platt.calibrate(torch.randn(5, 10))

    def test_nll_improves(self):
        platt = PlattScaling(max_iter=200)
        logits, labels = _make_logits_and_labels(500)
        result = platt.fit(logits, labels)
        assert result["nll_after"] <= result["nll_before"] + 0.1

    def test_repr(self):
        platt = PlattScaling()
        r = repr(platt)
        assert "PlattScaling" in r
