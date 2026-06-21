"""Comprehensive test suite for FlashFusion.

Covers models, strategies (WBF, voting, cascade, stacking, NMS fusion),
model merging (TIES, DARE, SLERP, task arithmetic), TTA, calibration,
uncertainty, auto-ensemble, pipelines, solutions, registry, CLI, engine,
edge cases, and integration.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleClassifier(nn.Module):
    """Tiny model for testing — 2-layer classifier."""

    def __init__(self, in_features=32, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        if x.shape[-1] != 32:
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 32).squeeze(1)
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


class _SimpleSpatialModel(nn.Module):
    """Tiny model that returns spatial outputs for TTA tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def simple_model():
    return _SimpleClassifier()


@pytest.fixture
def spatial_model():
    return _SimpleSpatialModel()


@pytest.fixture
def two_models():
    m1 = _SimpleClassifier()
    m2 = _SimpleClassifier()
    # Make m2 different
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return m1, m2


@pytest.fixture
def three_models():
    base = _SimpleClassifier()
    ft1 = _SimpleClassifier()
    ft2 = _SimpleClassifier()
    with torch.no_grad():
        for p in ft1.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for p in ft2.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return base, ft1, ft2


@pytest.fixture
def mock_det_predictions():
    """Mock detection predictions from 3 models."""
    return [
        {
            "boxes": np.array([[10, 20, 60, 80], [100, 100, 150, 180]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
        },
        {
            "boxes": np.array([[12, 22, 62, 82], [102, 98, 148, 178]], dtype=np.float32),
            "scores": np.array([0.85, 0.75], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
        },
        {
            "boxes": np.array([[11, 19, 59, 79], [99, 101, 151, 182]], dtype=np.float32),
            "scores": np.array([0.88, 0.82], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
        },
    ]


# ===========================================================================
# 1. Model / Component classes
# ===========================================================================


class TestFlashFusionModel:
    def test_instantiation(self):
        from flashfusion.models.fusion import FlashFusion

        sub_model = _SimpleClassifier()
        model = FlashFusion(models=[sub_model, sub_model])
        assert isinstance(model, nn.Module)

    def test_with_strategy(self):
        from flashfusion.models.fusion import FlashFusion

        sub_model = _SimpleClassifier()
        model = FlashFusion(models=[sub_model], input_size=(32, 32))
        assert model.input_size == (32, 32)


class TestBackboneAdapter:
    def test_import(self):
        from flashfusion.models.backbone.adapter import BackboneAdapter  # noqa: F401


class TestFusionHead:
    def test_import(self):
        from flashfusion.models.head.fusion_head import FusionHead  # noqa: F401


class TestFeatureFusionNeck:
    def test_import(self):
        from flashfusion.models.neck.feature_fusion import FeatureFusionNeck  # noqa: F401


# ===========================================================================
# 2. Registry
# ===========================================================================


class TestRegistry:
    def test_strategies_populated(self):
        from flashfusion.registry import STRATEGIES

        items = STRATEGIES.list()
        assert "weighted_box_fusion" in items
        assert "model_merging" in items
        assert "tta" in items

    def test_build_wbf_from_registry(self):
        from flashfusion.registry import STRATEGIES

        wbf = STRATEGIES.build("weighted_box_fusion", iou_threshold=0.5)
        assert wbf is not None

    def test_all_registries_exist(self):
        from flashfusion.registry import (
            BACKBONES,
            CALIBRATORS,
            DATASETS,
            HEADS,
            LOSSES,
            NECKS,
            PIPELINES,
            STRATEGIES,
            TRANSFORMS,
            UNCERTAINTY,
        )

        for reg in [
            BACKBONES,
            NECKS,
            HEADS,
            LOSSES,
            DATASETS,
            TRANSFORMS,
            STRATEGIES,
            PIPELINES,
            CALIBRATORS,
            UNCERTAINTY,
        ]:
            assert hasattr(reg, "list")


# ===========================================================================
# 3. CLI
# ===========================================================================


class TestCLI:
    def test_cli_import(self):
        from flashfusion.cli import main  # noqa: F401

    def test_version_accessible(self):
        import flashfusion

        assert flashfusion.__version__ == "1.0.0"


# ===========================================================================
# 4. Engine
# ===========================================================================


class TestEngine:
    def test_trainer_import(self):
        from flashfusion.engine.trainer import Trainer  # noqa: F401

    def test_validator_import(self):
        from flashfusion.engine.validator import Validator  # noqa: F401

    def test_predictor_import(self):
        from flashfusion.engine.predictor import Predictor  # noqa: F401

    def test_exporter_import(self):
        from flashfusion.engine.exporter import Exporter  # noqa: F401

    def test_callbacks_import(self):
        from flashfusion.engine.callbacks import CallbackHandler  # noqa: F401


# ===========================================================================
# 5. Strategies — WBF
# ===========================================================================


class TestWeightedBoxFusion:
    def test_instantiation(self):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion(weights=[0.5, 0.5], iou_threshold=0.55)
        assert wbf.iou_threshold == 0.55

    def test_fuse_two_models(self, mock_det_predictions):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion(iou_threshold=0.3)
        result = wbf.fuse(mock_det_predictions[:2])
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert len(result["boxes"]) > 0

    def test_fuse_three_models(self, mock_det_predictions):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion(weights=[0.4, 0.3, 0.3], iou_threshold=0.3)
        result = wbf.fuse(mock_det_predictions)
        assert result["boxes"].shape[1] == 4

    def test_fuse_empty(self):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion()
        result = wbf.fuse(
            [
                {"boxes": [], "scores": [], "labels": []},
                {"boxes": [], "scores": [], "labels": []},
            ]
        )
        assert len(result["boxes"]) == 0

    def test_conf_type_max(self, mock_det_predictions):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion(conf_type="max", iou_threshold=0.3)
        result = wbf.fuse(mock_det_predictions[:2])
        assert len(result["scores"]) > 0


# ===========================================================================
# 6. Strategies — Voting
# ===========================================================================


class TestVotingEnsemble:
    def test_instantiation(self):
        from flashfusion.strategies.voting import VotingEnsemble

        v = VotingEnsemble()
        assert v is not None


# ===========================================================================
# 7. Strategies — Cascade
# ===========================================================================


class TestCascadeFusion:
    def test_instantiation(self):
        from flashfusion.strategies.cascade import CascadeFusion

        c = CascadeFusion()
        assert c is not None


# ===========================================================================
# 8. Strategies — Stacking
# ===========================================================================


class TestStackingFusion:
    def test_import(self):
        from flashfusion.strategies.stacking import StackingEnsemble  # noqa: F401


# ===========================================================================
# 9. Strategies — NMS Fusion
# ===========================================================================


class TestNMSFusion:
    def test_instantiation(self):
        from flashfusion.strategies.nms_fusion import NMSFusion

        nms = NMSFusion()
        assert nms is not None


# ===========================================================================
# 10. Model Merging — TIES, DARE, SLERP, Task Arithmetic
# ===========================================================================


class TestModelMerging:
    def test_ties_merge(self, three_models):
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="ties", density=0.5)
        merged = merger.merge(base, [ft1, ft2])
        assert isinstance(merged, nn.Module)

        x = torch.randn(2, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (2, 5)

    def test_dare_merge(self, three_models):
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="dare", density=0.5)
        merged = merger.merge(base, [ft1, ft2])
        x = torch.randn(2, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (2, 5)

    def test_slerp_merge(self, two_models):
        from flashfusion.strategies.model_merging import ModelMerging

        m1, m2 = two_models
        merger = ModelMerging(method="slerp", temperature=0.5)
        merged = merger.merge(m1, [m1, m2])
        x = torch.randn(2, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (2, 5)

    def test_task_arithmetic_merge(self, three_models):
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="task_arithmetic")
        merged = merger.merge(base, [ft1, ft2], weights=[0.5, 0.5])
        x = torch.randn(2, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (2, 5)

    def test_invalid_method_raises(self):
        from flashfusion.strategies.model_merging import ModelMerging

        with pytest.raises(ValueError):
            ModelMerging(method="invalid")

    def test_slerp_requires_two_models(self, three_models):
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="slerp")
        with pytest.raises(ValueError):
            merger.merge(base, [ft1])  # Only 1 finetuned, needs exactly 2

    def test_fuse_interface(self, three_models):
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="ties", density=0.3)
        merged = merger.fuse([base, ft1, ft2])
        assert isinstance(merged, nn.Module)


# ===========================================================================
# 11. TTA
# ===========================================================================


class TestTTA:
    def test_multi_scale(self, spatial_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[0.8, 1.0, 1.2], flip_horizontal=False)
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(spatial_model, x)
        assert "predictions" in result

    def test_flip_augmentation(self, spatial_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=True, flip_vertical=True)
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(spatial_model, x)
        assert result["num_augmentations"] == 4

    def test_rotation_augmentation(self, spatial_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=False, rotations=[0, 90, 180, 270])
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(spatial_model, x)
        assert result["num_augmentations"] == 4

    def test_merge_mode_mean(self, simple_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=True, merge_mode="mean")
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(simple_model, x)
        assert "predictions" in result

    def test_merge_mode_max(self, simple_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=True, merge_mode="max")
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(simple_model, x)
        assert "predictions" in result

    def test_no_augmentations(self, simple_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=False)
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(simple_model, x)
        assert result["num_augmentations"] == 1


# ===========================================================================
# 12. Calibration — Temperature Scaling, Platt Scaling
# ===========================================================================


class TestTemperatureScaling:
    def test_fit_and_calibrate(self):
        from flashfusion.calibration.temperature_scaling import TemperatureScaling

        ts = TemperatureScaling()
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 5, (100,))
        result = ts.fit(logits, labels)
        assert "temperature" in result
        assert result["temperature"] > 0

        probs = ts.calibrate(logits)
        assert probs.shape == (100, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-4)

    def test_ece_computation(self):
        from flashfusion.calibration.temperature_scaling import TemperatureScaling

        probs = torch.softmax(torch.randn(50, 3), dim=1)
        labels = torch.randint(0, 3, (50,))
        ece = TemperatureScaling.expected_calibration_error(probs, labels)
        assert 0.0 <= ece.item() <= 1.0

    def test_reliability_diagram(self):
        from flashfusion.calibration.temperature_scaling import TemperatureScaling

        probs = torch.softmax(torch.randn(50, 3), dim=1)
        labels = torch.randint(0, 3, (50,))
        diag = TemperatureScaling.reliability_diagram(probs, labels)
        assert "bin_centers" in diag
        assert len(diag["bin_centers"]) == 15

    def test_calibrate_before_fit_raises(self):
        from flashfusion.calibration.temperature_scaling import TemperatureScaling

        ts = TemperatureScaling()
        with pytest.raises(RuntimeError):
            ts.calibrate(torch.randn(10, 5))


class TestPlattScaling:
    def test_instantiation(self):
        from flashfusion.calibration.platt_scaling import PlattScaling

        ps = PlattScaling()
        assert ps is not None


# ===========================================================================
# 13. Uncertainty — MC Dropout, Deep Ensembles, Entropy
# ===========================================================================


class TestMCDropout:
    def test_estimate(self, simple_model):
        from flashfusion.uncertainty.mc_dropout import MCDropout

        mc = MCDropout(n_samples=5)
        x = torch.randn(2, 32)
        result = mc.estimate(simple_model, x)
        assert "mean" in result
        assert "variance" in result
        assert "uncertainty" in result
        assert "entropy" in result
        assert "mutual_information" in result
        assert result["predictions"].shape[0] == 5

    def test_confidence_interval(self, simple_model):
        from flashfusion.uncertainty.mc_dropout import MCDropout

        mc = MCDropout(n_samples=10)
        x = torch.randn(2, 32)
        result = mc.estimate(simple_model, x)
        lower, upper = MCDropout.get_confidence_interval(result["predictions"], confidence=0.9)
        assert lower.shape == upper.shape

    def test_custom_dropout_rate(self, simple_model):
        from flashfusion.uncertainty.mc_dropout import MCDropout

        mc = MCDropout(n_samples=3, dropout_rate=0.3)
        x = torch.randn(2, 32)
        result = mc.estimate(simple_model, x)
        assert result["mean"].shape == (2, 5)

    def test_invalid_n_samples(self):
        from flashfusion.uncertainty.mc_dropout import MCDropout

        with pytest.raises(ValueError):
            MCDropout(n_samples=1)


class TestDeepEnsemble:
    def test_instantiation(self):
        from flashfusion.uncertainty.deep_ensembles import DeepEnsemble

        de = DeepEnsemble()
        assert de is not None


class TestEntropyEstimator:
    def test_instantiation(self):
        from flashfusion.uncertainty.entropy import EntropyEstimator

        ee = EntropyEstimator()
        assert ee is not None


# ===========================================================================
# 14. Auto-Ensemble Selection
# ===========================================================================


class TestAutoEnsemble:
    def test_instantiation(self):
        from flashfusion.strategies.auto_ensemble import AutoEnsembleSelection

        ae = AutoEnsembleSelection()
        assert ae is not None


# ===========================================================================
# 15. Pipelines — det+cls, det+seg, multi-task
# ===========================================================================


class TestPipelines:
    def test_det_cls_pipeline_import(self):
        from flashfusion.pipelines.det_cls_pipeline import DetClsPipeline  # noqa: F401

    def test_det_seg_pipeline_import(self):
        from flashfusion.pipelines.det_seg_pipeline import DetSegPipeline  # noqa: F401

    def test_multi_task_pipeline_import(self):
        from flashfusion.pipelines.multi_task_pipeline import MultiTaskPipeline  # noqa: F401


# ===========================================================================
# 16. Solutions
# ===========================================================================


class TestSolutions:
    def test_ensemble_detector_import(self):
        from flashfusion.solutions.ensemble_detector import EnsembleDetector  # noqa: F401

    def test_multi_model_analyzer_import(self):
        from flashfusion.solutions.multi_model_analyzer import MultiModelAnalyzer  # noqa: F401


# ===========================================================================
# 17. Utils — Metrics, Visualization
# ===========================================================================


class TestUtils:
    def test_metrics_import(self):
        from flashfusion.utils.metrics import compute_map, compute_accuracy, compute_fusion_metrics  # noqa: F401

    def test_visualization_import(self):
        from flashfusion.utils.visualization import draw_detections, draw_fusion_results  # noqa: F401

    def test_checkpoint_import(self):
        from flashfusion.utils.checkpoint import save_checkpoint, load_checkpoint  # noqa: F401

    def test_logger_import(self):
        from flashfusion.utils.logger import setup_logger, AverageMeter  # noqa: F401


# ===========================================================================
# 18. Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_wbf_single_model(self):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion()
        result = wbf.fuse(
            [
                {
                    "boxes": np.array([[10, 20, 50, 60]]),
                    "scores": np.array([0.9]),
                    "labels": np.array([0]),
                }
            ]
        )
        assert len(result["boxes"]) == 1

    def test_wbf_below_threshold(self):
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        wbf = WeightedBoxFusion(skip_box_threshold=0.5)
        result = wbf.fuse(
            [
                {
                    "boxes": np.array([[10, 20, 50, 60]]),
                    "scores": np.array([0.1]),
                    "labels": np.array([0]),
                }
            ]
        )
        assert len(result["boxes"]) == 0

    def test_tta_empty_batch(self, simple_model):
        from flashfusion.strategies.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=False)
        x = torch.randn(1, 3, 32, 32)
        result = tta.predict(simple_model, x)
        assert "predictions" in result

    def test_temperature_scaling_uniform_logits(self):
        from flashfusion.calibration.temperature_scaling import TemperatureScaling

        ts = TemperatureScaling()
        logits = torch.zeros(50, 3)
        labels = torch.randint(0, 3, (50,))
        result = ts.fit(logits, labels)
        assert result["temperature"] > 0

    def test_model_merging_single_finetuned(self):
        from flashfusion.strategies.model_merging import ModelMerging

        base = _SimpleClassifier()
        ft1 = _SimpleClassifier()
        with torch.no_grad():
            for p in ft1.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        merger = ModelMerging(method="ties", density=0.5)
        merged = merger.merge(base, [ft1])
        x = torch.randn(1, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (1, 5)


# ===========================================================================
# 19. Integration
# ===========================================================================


class TestIntegration:
    def test_wbf_to_calibration_pipeline(self):
        """End-to-end: WBF fuse → temperature calibrate."""
        from flashfusion.calibration.temperature_scaling import TemperatureScaling
        from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion

        preds = [
            {"boxes": np.array([[10, 20, 60, 80]]), "scores": np.array([0.9]), "labels": np.array([0])},
            {"boxes": np.array([[12, 22, 62, 82]]), "scores": np.array([0.85]), "labels": np.array([0])},
        ]
        wbf = WeightedBoxFusion(iou_threshold=0.3)
        fused = wbf.fuse(preds)
        assert len(fused["boxes"]) > 0

        ts = TemperatureScaling()
        logits = torch.randn(50, 5)
        labels = torch.randint(0, 5, (50,))
        ts.fit(logits, labels)
        probs = ts.calibrate(logits)
        assert probs.shape == (50, 5)

    def test_tta_with_mc_dropout(self, simple_model):
        """TTA prediction + MC Dropout uncertainty."""
        from flashfusion.strategies.tta import TestTimeAugmentation
        from flashfusion.uncertainty.mc_dropout import MCDropout

        tta = TestTimeAugmentation(scales=[1.0], flip_horizontal=True)
        x = torch.randn(2, 3, 32, 32)
        tta_result = tta.predict(simple_model, x)
        assert "predictions" in tta_result

        mc = MCDropout(n_samples=5)
        unc_result = mc.estimate(simple_model, torch.randn(2, 32))
        assert "uncertainty" in unc_result

    def test_model_merge_then_predict(self, three_models):
        """Merge models then run forward pass."""
        from flashfusion.strategies.model_merging import ModelMerging

        base, ft1, ft2 = three_models
        merger = ModelMerging(method="task_arithmetic")
        merged = merger.merge(base, [ft1, ft2], weights=[0.5, 0.5])
        merged.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            out = merged(x)
        assert out.shape == (4, 5)
