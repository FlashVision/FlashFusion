"""Registry pattern for pluggable components.

Usage:
    from flashfusion.registry import BACKBONES, STRATEGIES, PIPELINES

    @STRATEGIES.register("MyFusion")
    class MyFusion:
        ...

    strategy = STRATEGIES.build("MyFusion", **kwargs)
"""

from typing import Any, Callable, Dict, Optional


class Registry:
    """A registry that maps names to classes/functions."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        def decorator(obj):
            key = name or obj.__name__
            if key in self._registry:
                raise KeyError(f"{self._name}: '{key}' is already registered")
            self._registry[key] = obj
            return obj

        if callable(name):
            obj = name
            key = obj.__name__
            self._registry[key] = obj
            return obj

        return decorator

    def build(self, name: str, **kwargs) -> Any:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: '{name}' not found. Available: [{available}]"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Any:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self._name}: '{name}' not found. Available: [{available}]"
            )
        return self._registry[name]

    def list(self) -> list:
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={self.list()})"


BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
STRATEGIES = Registry("strategies")
PIPELINES = Registry("pipelines")
