"""FlashFusion fusion strategies for combining multi-model outputs."""

from flashfusion.strategies.weighted_box_fusion import WeightedBoxFusion
from flashfusion.strategies.voting import VotingEnsemble
from flashfusion.strategies.cascade import CascadeFusion
from flashfusion.strategies.stacking import StackingEnsemble
from flashfusion.strategies.nms_fusion import NMSFusion

__all__ = [
    "WeightedBoxFusion",
    "VotingEnsemble",
    "CascadeFusion",
    "StackingEnsemble",
    "NMSFusion",
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
}


def get_strategy(name: str, **kwargs):
    """Get a fusion strategy by name.

    Args:
        name: Strategy name (wbf, voting, cascade, stacking, nms).
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
    return _STRATEGY_MAP[name_lower](**kwargs)
