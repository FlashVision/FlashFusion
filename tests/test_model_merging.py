"""Tests for model merging strategies."""

import pytest
import torch
import torch.nn as nn

from flashfusion.strategies.model_merging import ModelMerging


def _make_simple_model():
    """Create a small model for testing merging."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )


def _make_finetuned_variants(base_model, n_variants=3):
    """Create fine-tuned variants by adding noise to base model."""
    import copy

    variants = []
    for i in range(n_variants):
        variant = copy.deepcopy(base_model)
        with torch.no_grad():
            for param in variant.parameters():
                param.add_(torch.randn_like(param) * 0.1 * (i + 1))
        variants.append(variant)
    return variants


class TestModelMerging:
    """Tests for ModelMerging strategy."""

    def test_ties_merge(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 3)
        merger = ModelMerging(method="ties", density=0.3)
        merged = merger.merge(base, variants)
        assert merged is not None
        x = torch.randn(2, 10)
        out = merged(x)
        assert out.shape == (2, 10)

    def test_dare_merge(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 2)
        merger = ModelMerging(method="dare", density=0.5)
        merged = merger.merge(base, variants)
        assert merged is not None
        x = torch.randn(2, 10)
        out = merged(x)
        assert out.shape == (2, 10)

    def test_slerp_merge(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 2)
        merger = ModelMerging(method="slerp", temperature=0.5)
        merged = merger.merge(base, variants)
        assert merged is not None
        x = torch.randn(2, 10)
        out = merged(x)
        assert out.shape == (2, 10)

    def test_slerp_requires_two_models(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 3)
        merger = ModelMerging(method="slerp")
        with pytest.raises(ValueError, match="exactly 2 models"):
            merger.merge(base, variants)

    def test_task_arithmetic_merge(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 3)
        merger = ModelMerging(method="task_arithmetic")
        merged = merger.merge(base, variants, weights=[0.5, 0.3, 0.2])
        assert merged is not None
        x = torch.randn(2, 10)
        out = merged(x)
        assert out.shape == (2, 10)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            ModelMerging(method="invalid")

    def test_fuse_interface(self):
        base = _make_simple_model()
        variants = _make_finetuned_variants(base, 2)
        merger = ModelMerging(method="task_arithmetic")
        merged = merger.fuse([base] + variants)
        assert merged is not None

    def test_fuse_requires_two_models(self):
        base = _make_simple_model()
        merger = ModelMerging(method="ties")
        with pytest.raises(ValueError, match="at least 2"):
            merger.fuse([base])

    def test_repr(self):
        merger = ModelMerging(method="ties", density=0.5)
        r = repr(merger)
        assert "ModelMerging" in r
        assert "ties" in r
