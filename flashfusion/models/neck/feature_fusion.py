"""Feature-level fusion neck for combining multi-model feature pyramids.

Fuses feature maps from multiple backbones at the feature pyramid level,
enabling deep integration of complementary representations.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfusion.registry import NECKS


@NECKS.register("feature_fusion")
class FeatureFusionNeck(nn.Module):
    """Feature Pyramid Fusion Neck.

    Combines multi-scale feature pyramids from multiple models using
    learned attention weights or concatenation + convolution.

    Args:
        in_channels: Number of channels per feature level.
        out_channels: Output channels after fusion.
        num_models: Number of source models to fuse.
        num_levels: Number of feature pyramid levels.
        fusion_mode: Fusion mode ('attention', 'concat', 'add').
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_models: int = 2,
        num_levels: int = 3,
        fusion_mode: str = "attention",
    ):
        super().__init__()
        self.num_models = num_models
        self.num_levels = num_levels
        self.fusion_mode = fusion_mode

        if fusion_mode == "attention":
            self.attention = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(in_channels * num_models, num_models),
                        nn.Softmax(dim=-1),
                    )
                    for _ in range(num_levels)
                ]
            )
        elif fusion_mode == "concat":
            self.projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels * num_models, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    )
                    for _ in range(num_levels)
                ]
            )

        self.out_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, multi_features: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Fuse feature pyramids from multiple models.

        Args:
            multi_features: List of feature pyramids, one per model.
                Each pyramid is a list of tensors at different scales.

        Returns:
            Fused feature pyramid as a list of tensors.
        """
        fused_levels = []

        for level_idx in range(self.num_levels):
            level_features = [feats[level_idx] for feats in multi_features]

            if self.fusion_mode == "attention":
                fused = self._attention_fusion(level_features, level_idx)
            elif self.fusion_mode == "concat":
                fused = self._concat_fusion(level_features, level_idx)
            else:
                fused = self._add_fusion(level_features)

            fused = self.out_convs[level_idx](fused)
            fused_levels.append(fused)

        return fused_levels

    def _attention_fusion(self, features: List[torch.Tensor], level_idx: int) -> torch.Tensor:
        """Fuse features using learned attention weights."""
        stacked = torch.stack(features, dim=1)  # (B, N, C, H, W)
        B, N, C, H, W = stacked.shape

        concat = torch.cat(features, dim=1)  # (B, N*C, H, W)
        weights = self.attention[level_idx](concat)  # (B, N)
        weights = weights.view(B, N, 1, 1, 1)

        fused = (stacked * weights).sum(dim=1)  # (B, C, H, W)
        return fused

    def _concat_fusion(self, features: List[torch.Tensor], level_idx: int) -> torch.Tensor:
        """Fuse features using concatenation and projection."""
        concat = torch.cat(features, dim=1)
        return self.projections[level_idx](concat)

    def _add_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using element-wise addition."""
        result = features[0]
        for feat in features[1:]:
            if feat.shape != result.shape:
                feat = F.interpolate(feat, size=result.shape[2:], mode="bilinear", align_corners=False)
            result = result + feat
        return result / len(features)
