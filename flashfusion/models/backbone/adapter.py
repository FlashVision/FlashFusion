"""Backbone adapters for integrating different source models into FlashFusion.

Adapters normalize feature outputs from heterogeneous backbones (e.g., ResNet,
EfficientNet, CSPDarknet) into a unified feature format for fusion.
"""

from typing import List

import torch
import torch.nn as nn

from flashfusion.registry import BACKBONES


@BACKBONES.register("adapter")
class BackboneAdapter(nn.Module):
    """Adapter that wraps a source backbone and normalizes its feature outputs.

    Args:
        backbone: Source backbone module.
        output_channels: Target number of output channels for each feature level.
        feature_levels: Number of feature pyramid levels to output.
        freeze: Whether to freeze backbone parameters.
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_channels: int = 256,
        feature_levels: int = 3,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_channels = output_channels
        self.feature_levels = feature_levels

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.adapters = nn.ModuleList([
            nn.Conv2d(output_channels, output_channels, 1)
            for _ in range(feature_levels)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract and adapt features from the source backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of adapted feature tensors at each pyramid level.
        """
        features = self._extract_features(x)
        adapted = []
        for i, (feat, adapter) in enumerate(zip(features, self.adapters)):
            adapted.append(adapter(feat))
        return adapted

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from backbone.

        Override this method for custom backbone feature extraction.
        """
        if hasattr(self.backbone, "extract_features"):
            return self.backbone.extract_features(x)
        output = self.backbone(x)
        if isinstance(output, (list, tuple)):
            return list(output[: self.feature_levels])
        return [output]

    @property
    def out_channels(self) -> int:
        """Return the number of output channels."""
        return self.output_channels


def adapt_backbone(
    backbone: nn.Module,
    source_channels: List[int],
    target_channels: int = 256,
    freeze: bool = True,
) -> "ChannelAdapter":
    """Create a channel adapter for a backbone with known output channels.

    Args:
        backbone: Source backbone module.
        source_channels: List of channel counts for each feature level.
        target_channels: Unified target channel count.
        freeze: Whether to freeze backbone parameters.

    Returns:
        ChannelAdapter wrapping the backbone.
    """
    return ChannelAdapter(backbone, source_channels, target_channels, freeze)


class ChannelAdapter(nn.Module):
    """Adapter that maps backbone features to a unified channel dimension.

    Args:
        backbone: Source backbone module.
        source_channels: Channel counts per feature level from the backbone.
        target_channels: Target unified channel count.
        freeze: Whether to freeze backbone weights.
    """

    def __init__(
        self,
        backbone: nn.Module,
        source_channels: List[int],
        target_channels: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = backbone

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, target_channels, 1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.SiLU(inplace=True),
            )
            for ch in source_channels
        ])
        self.target_channels = target_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass: extract features and project to target channels."""
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            features = [features]

        projected = []
        for feat, proj in zip(features, self.projections):
            projected.append(proj(feat))
        return projected
