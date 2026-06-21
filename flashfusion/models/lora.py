"""LoRA (Low-Rank Adaptation) utilities for FlashFusion.

Provides LoRA and QLoRA adapters for efficient fine-tuning of fusion layers
without modifying the frozen base model weights.
"""

from typing import Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer.

    Adds a low-rank decomposition (A @ B) to an existing linear layer,
    enabling efficient fine-tuning with minimal trainable parameters.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA output.
        dropout: Dropout probability applied to input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * (x @ A^T @ B^T)."""
        x = self.dropout(x)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """A Linear layer wrapped with LoRA adaptation.

    Performs: output = Linear(x) + LoRA(x)
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.lora = LoRALayer(
            in_features=original.in_features,
            out_features=original.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base linear + LoRA delta."""
        return self.original(x) + self.lora(x)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[list] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA adapters to a model's linear layers.

    Replaces targeted Linear layers with LoRALinear that adds the low-rank
    delta in the forward pass.

    Args:
        model: Target model to adapt.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: List of module name patterns to target. If None, targets all Linear layers.
        dropout: Dropout probability for LoRA layers.

    Returns:
        Model with LoRA adapters applied.
    """
    replacements = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules and not any(t in name for t in target_modules):
            continue
        replacements[name] = module

    for name, linear in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        lora_linear = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_linear)

    model.requires_grad_(False)
    for param_name, param in model.named_parameters():
        if "lora" in param_name:
            param.requires_grad_(True)

    return model


def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[list] = None,
    bits: int = 4,
) -> nn.Module:
    """Apply QLoRA (Quantized LoRA) adapters to a model.

    Quantizes the base model weights to reduce memory footprint while
    applying LoRA adapters for fine-tuning.

    Args:
        model: Target model to adapt.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Module name patterns to target.
        bits: Quantization bits (4 or 8).

    Returns:
        Model with QLoRA adapters.
    """
    model = apply_lora(model, rank=rank, alpha=alpha, target_modules=target_modules)
    # Quantization placeholder — requires bitsandbytes or similar backend
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model for inference.

    After merging, the LoRA layers are removed and the model operates
    at full speed without the overhead of separate LoRA computations.

    Args:
        model: Model with LoRA adapters.

    Returns:
        Model with LoRA weights merged into base layers.
    """
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora = module.lora
            with torch.no_grad():
                delta = (lora.lora_B @ lora.lora_A) * lora.scaling
                module.original.weight.add_(delta)
            replacements[name] = module.original

    for name, linear in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    return model
