"""Fusion decision heads for combining multi-model features into predictions.

These heads sit on top of fused features and produce the final task-specific
outputs (detections, classifications, segmentation masks, etc.).
"""

from typing import Dict, List

import torch
import torch.nn as nn

from flashfusion.registry import HEADS


@HEADS.register("fusion_head")
class FusionHead(nn.Module):
    """Multi-model fusion decision head.

    Combines features from multiple models and produces unified predictions.

    Args:
        in_channels: Number of input channels from fused features.
        num_classes: Number of output classes.
        num_models: Number of source models being fused.
        hidden_dim: Hidden dimension for fusion MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_models: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_models = num_models

        self.attention = nn.Sequential(
            nn.Linear(in_channels * num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.box_regressor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        task: str = "detection",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the fusion head.

        Args:
            features: List of feature tensors from each model, shape (B, C).
            task: Task type ('detection', 'classification').

        Returns:
            Dictionary with 'scores' and optionally 'boxes' tensors.
        """
        stacked = torch.stack(features, dim=1)  # (B, num_models, C)
        B, N, C = stacked.shape

        concat = stacked.reshape(B, N * C)
        attn_weights = self.attention(concat)  # (B, num_models)
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, C)

        output = {"scores": self.classifier(fused)}
        if task == "detection":
            output["boxes"] = self.box_regressor(fused)

        return output


@HEADS.register("lightweight_fusion_head")
class LightweightFusionHead(nn.Module):
    """Lightweight fusion head using simple weighted averaging.

    Suitable for edge deployment where computational budget is limited.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        num_models: Number of source models.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_models: int = 2,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with learned weighted averaging."""
        weights = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(features, dim=0)  # (num_models, B, C)
        fused = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # (B, C)
        return {"scores": self.classifier(fused)}
