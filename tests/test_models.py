"""Tests for FlashFusion model creation and forward pass."""

import pytest
import torch
import torch.nn as nn

from flashfusion.models.fusion import FlashFusion


class DummyDetector(nn.Module):
    """Minimal detector for testing that outputs boxes, scores, labels."""

    def __init__(self, num_classes: int = 10, num_detections: int = 5):
        super().__init__()
        self.linear = nn.Linear(3 * 320 * 320, 64)
        self.num_classes = num_classes
        self.num_detections = num_detections

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        return {
            "boxes": torch.rand(batch_size, self.num_detections, 4) * 320,
            "scores": torch.rand(batch_size, self.num_detections),
            "labels": torch.randint(0, self.num_classes, (batch_size, self.num_detections)),
        }


class TestFlashFusionModel:
    """Test suite for the FlashFusion model class."""

    def test_creation_with_modules(self):
        """Test creating FlashFusion with nn.Module instances."""
        model_a = DummyDetector(num_classes=10)
        model_b = DummyDetector(num_classes=10)

        fusion = FlashFusion(
            models=[model_a, model_b],
            input_size=(320, 320),
            device="cpu",
        )

        assert fusion.num_models == 2
        assert len(fusion.model_names) == 2

    def test_forward_pass(self):
        """Test forward pass produces valid output."""
        model_a = DummyDetector()
        model_b = DummyDetector()

        fusion = FlashFusion(
            models=[model_a, model_b],
            input_size=(320, 320),
            device="cpu",
        )

        x = torch.randn(2, 3, 320, 320)
        output = fusion(x)

        assert output is not None
        assert isinstance(output, dict)

    def test_num_models_property(self):
        """Test num_models property returns correct count."""
        models = [DummyDetector() for _ in range(3)]
        fusion = FlashFusion(models=models, device="cpu")
        assert fusion.num_models == 3

    def test_model_names_property(self):
        """Test model_names property returns class names."""
        models = [DummyDetector(), DummyDetector()]
        fusion = FlashFusion(models=models, device="cpu")
        assert all(name == "DummyDetector" for name in fusion.model_names)

    def test_repr(self):
        """Test string representation."""
        model = DummyDetector()
        fusion = FlashFusion(models=[model], device="cpu")
        repr_str = repr(fusion)
        assert "FlashFusion" in repr_str

    def test_from_models_invalid_path(self):
        """Test from_models raises on invalid paths."""
        with pytest.raises((NotImplementedError, FileNotFoundError, ValueError)):
            FlashFusion.from_models(
                model_paths=["nonexistent_model.pt"],
                strategy="wbf",
                device="cpu",
            )

    def test_device_resolution_cpu(self):
        """Test device resolution defaults to CPU when no CUDA."""
        model = DummyDetector()
        fusion = FlashFusion(models=[model], device="cpu")
        assert fusion.device == torch.device("cpu")

    def test_weights_assignment(self):
        """Test that weights are properly stored."""
        models = [DummyDetector(), DummyDetector()]
        weights = [0.6, 0.4]
        fusion = FlashFusion(models=models, weights=weights, device="cpu")
        assert fusion.weights == [0.6, 0.4]
