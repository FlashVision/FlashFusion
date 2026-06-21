"""FlashFusion — Multi-model vision fusion for ensemble, cascade, and multi-task pipelines."""

__version__ = "1.0.0"

from flashfusion.models.fusion import FlashFusion
from flashfusion.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashfusion.engine.trainer import Trainer
from flashfusion.engine.validator import Validator
from flashfusion.engine.predictor import Predictor
from flashfusion.engine.exporter import Exporter
from flashfusion.cfg import get_config
from flashfusion.strategies import WeightedBoxFusion, VotingEnsemble, CascadeFusion
from flashfusion.solutions import EnsembleDetector, MultiModelAnalyzer
from flashfusion.analytics import Benchmark

__all__ = [
    "FlashFusion", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "WeightedBoxFusion", "VotingEnsemble", "CascadeFusion",
    "EnsembleDetector", "MultiModelAnalyzer",
    "Benchmark",
    "__version__",
]
