"""Tests for FlashFusion fusion strategies."""

import numpy as np
import pytest
import torch

from flashfusion.strategies import (
    WeightedBoxFusion,
    VotingEnsemble,
    CascadeFusion,
    StackingEnsemble,
    NMSFusion,
    get_strategy,
)


def _make_predictions(num_models: int = 3, num_boxes: int = 5) -> list:
    """Create dummy predictions from multiple models."""
    predictions = []
    for _ in range(num_models):
        boxes = np.random.rand(num_boxes, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        predictions.append({
            "boxes": torch.from_numpy(boxes).float(),
            "scores": torch.rand(num_boxes),
            "labels": torch.zeros(num_boxes, dtype=torch.long),
        })
    return predictions


class TestWeightedBoxFusion:
    """Test suite for Weighted Box Fusion strategy."""

    def test_basic_fusion(self):
        """Test WBF produces valid output structure."""
        wbf = WeightedBoxFusion(iou_threshold=0.5)
        predictions = _make_predictions(num_models=3, num_boxes=5)

        result = wbf.fuse(predictions)

        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert isinstance(result["boxes"], torch.Tensor)
        assert isinstance(result["scores"], torch.Tensor)

    def test_empty_predictions(self):
        """Test WBF handles empty predictions gracefully."""
        wbf = WeightedBoxFusion()
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
            for _ in range(3)
        ]

        result = wbf.fuse(empty_preds)
        assert result["boxes"].shape[0] == 0

    def test_single_model(self):
        """Test WBF with a single model returns its predictions."""
        wbf = WeightedBoxFusion()
        predictions = _make_predictions(num_models=1, num_boxes=3)

        result = wbf.fuse(predictions)
        assert result["boxes"].shape[0] > 0

    def test_custom_weights(self):
        """Test WBF respects custom per-model weights."""
        wbf = WeightedBoxFusion(weights=[0.7, 0.3])
        predictions = _make_predictions(num_models=2, num_boxes=4)

        result = wbf.fuse(predictions, weights=[0.7, 0.3])
        assert result["boxes"].shape[0] > 0

    def test_iou_threshold_effect(self):
        """Test that different IoU thresholds affect clustering."""
        predictions = _make_predictions(num_models=2, num_boxes=10)

        wbf_strict = WeightedBoxFusion(iou_threshold=0.9)
        wbf_loose = WeightedBoxFusion(iou_threshold=0.1)

        result_strict = wbf_strict.fuse(predictions)
        result_loose = wbf_loose.fuse(predictions)

        # Loose threshold should cluster more aggressively → fewer output boxes
        assert result_loose["boxes"].shape[0] <= result_strict["boxes"].shape[0]

    def test_repr(self):
        """Test string representation."""
        wbf = WeightedBoxFusion(weights=[0.5, 0.5], iou_threshold=0.6)
        assert "WeightedBoxFusion" in repr(wbf)


class TestVotingEnsemble:
    """Test suite for VotingEnsemble strategy."""

    def test_basic_fusion(self):
        """Test voting ensemble produces output."""
        voting = VotingEnsemble()
        num_classes = 10
        predictions = [
            {"logits": torch.randn(1, num_classes)},
            {"logits": torch.randn(1, num_classes)},
            {"logits": torch.randn(1, num_classes)},
        ]

        result = voting.fuse(predictions)
        assert "labels" in result
        assert "scores" in result


class TestCascadeFusion:
    """Test suite for CascadeFusion strategy."""

    def test_basic_fusion(self):
        """Test cascade fusion produces output."""
        cascade = CascadeFusion()
        predictions = _make_predictions(num_models=2, num_boxes=5)

        result = cascade.fuse(predictions)
        assert "boxes" in result
        assert "scores" in result


class TestStackingEnsemble:
    """Test suite for StackingEnsemble strategy."""

    def test_basic_fusion(self):
        """Test stacking ensemble produces output."""
        stacking = StackingEnsemble()
        num_classes = 10
        predictions = [
            {"logits": torch.randn(1, num_classes)},
            {"logits": torch.randn(1, num_classes)},
            {"logits": torch.randn(1, num_classes)},
        ]

        result = stacking.fuse(predictions)
        assert "labels" in result


class TestNMSFusion:
    """Test suite for NMSFusion strategy."""

    def test_basic_fusion(self):
        """Test NMS fusion produces output."""
        nms = NMSFusion()
        predictions = _make_predictions(num_models=2, num_boxes=5)

        result = nms.fuse(predictions)
        assert "boxes" in result
        assert "scores" in result

    def test_empty_input(self):
        """Test NMS handles empty input."""
        nms = NMSFusion()
        empty_preds = [
            {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
        ]
        result = nms.fuse(empty_preds)
        assert result["boxes"].shape[0] == 0


class TestGetStrategy:
    """Test get_strategy factory function."""

    def test_get_wbf(self):
        """Test getting WBF strategy by name."""
        strategy = get_strategy("wbf")
        assert isinstance(strategy, WeightedBoxFusion)

    def test_get_voting(self):
        """Test getting VotingEnsemble by name."""
        strategy = get_strategy("voting")
        assert isinstance(strategy, VotingEnsemble)

    def test_get_cascade(self):
        """Test getting CascadeFusion by name."""
        strategy = get_strategy("cascade")
        assert isinstance(strategy, CascadeFusion)

    def test_get_nms(self):
        """Test getting NMSFusion by name."""
        strategy = get_strategy("nms")
        assert isinstance(strategy, NMSFusion)

    def test_invalid_strategy(self):
        """Test that invalid strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")

    def test_case_insensitive(self):
        """Test strategy lookup is case-insensitive."""
        strategy = get_strategy("WBF")
        assert isinstance(strategy, WeightedBoxFusion)
