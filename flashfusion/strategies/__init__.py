"""FlashFusion fusion strategies for combining multi-model outputs."""

from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion
from flashfusion.strategies.voting import VotingEnsemble
from flashfusion.strategies.cascade import CascadeFusion
from flashfusion.strategies.stacking import StackingEnsemble
from flashfusion.strategies.nms_fusion import NMSFusion
from flashfusion.strategies.model_merging import ModelMerging
from flashfusion.strategies.tta import TestTimeAugmentation
from flashfusion.strategies.auto_ensemble import AutoEnsembleSelection

__all__ = [
    "WeightedBoxFusion",
    "VotingEnsemble",
    "CascadeFusion",
    "StackingEnsemble",
    "NMSFusion",
    "ModelMerging",
    "TestTimeAugmentation",
    "AutoEnsembleSelection",
    "get_strategy",
]

_STRATEGY_MAP = {
    "wbf": WeightedBoxFusion,
    "weighted_box_fusion": WeightedBoxFusion,
    "voting": VotingEnsemble,
    "cascade": CascadeFusion,
    "stacking": StackingEnsemble,
    "nms": NMSFusion,
    "nms_fusion": NMSFusion,
    "model_merging": ModelMerging,
    "ties": ModelMerging,
    "dare": ModelMerging,
    "slerp": ModelMerging,
    "task_arithmetic": ModelMerging,
    "tta": TestTimeAugmentation,
    "test_time_augmentation": TestTimeAugmentation,
    "auto_ensemble": AutoEnsembleSelection,
}


def get_strategy(name: str, **kwargs):
    """Get a fusion strategy by name.

    Args:
        name: Strategy name (wbf, voting, cascade, stacking, nms, model_merging, tta, auto_ensemble).
        **kwargs: Strategy-specific keyword arguments.

    Returns:
        Instantiated strategy object.

    Raises:
        ValueError: If strategy name is not recognized.
    """
    name_lower = name.lower()
    if name_lower not in _STRATEGY_MAP:
        available = ", ".join(sorted(_STRATEGY_MAP.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: [{available}]")

    if name_lower in ("ties", "dare", "slerp", "task_arithmetic"):
        kwargs.setdefault("method", name_lower)

    return _STRATEGY_MAP[name_lower](**kwargs)
