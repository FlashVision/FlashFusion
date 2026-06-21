"""Tests for Test-Time Augmentation."""

import pytest
import torch
import torch.nn as nn

from flashfusion.strategies.tta import TestTimeAugmentation


def _make_classifier():
    """Create a simple classifier for TTA testing."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 10),
    )


class TestTTA:
    """Tests for TestTimeAugmentation strategy."""

    def test_basic_predict(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=False)
        inputs = torch.randn(2, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert "predictions" in result
        assert "num_augmentations" in result

    def test_multi_scale(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(scales=[0.8, 1.0, 1.2], flip_horizontal=False)
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert result["num_augmentations"] == 3

    def test_flip_augmentation(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=True, flip_vertical=True)
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert result["num_augmentations"] == 4  # no_flip, h, v, h+v

    def test_rotation_augmentation(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(
            scales=[1.0], flip_horizontal=False, rotations=[0, 90, 180, 270]
        )
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert result["num_augmentations"] == 4

    def test_combined_augmentations(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(
            scales=[1.0, 1.5], flip_horizontal=True, rotations=[0, 180]
        )
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        # 2 scales x 2 flips x 2 rotations = 8
        assert result["num_augmentations"] == 8

    def test_mean_merge(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(scales=[1.0, 1.5], merge_mode="mean", flip_horizontal=False)
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert "predictions" in result

    def test_max_merge(self):
        model = _make_classifier()
        tta = TestTimeAugmentation(scales=[1.0, 1.5], merge_mode="max", flip_horizontal=False)
        inputs = torch.randn(1, 3, 32, 32)
        result = tta.predict(model, inputs)
        assert "predictions" in result

    def test_rotate_tensor(self):
        tensor = torch.randn(1, 3, 16, 16)
        rotated_90 = TestTimeAugmentation._rotate_tensor(tensor, 90)
        rotated_180 = TestTimeAugmentation._rotate_tensor(tensor, 180)
        rotated_360 = TestTimeAugmentation._rotate_tensor(tensor, 360)
        assert rotated_90.shape == tensor.shape
        assert rotated_180.shape == tensor.shape
        assert torch.allclose(rotated_360, tensor)

    def test_repr(self):
        tta = TestTimeAugmentation(scales=[0.5, 1.0])
        r = repr(tta)
        assert "TestTimeAugmentation" in r
