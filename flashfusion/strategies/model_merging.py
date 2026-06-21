"""Model merging strategies for combining trained model weights.

Implements TIES, DARE, SLERP, and Task Arithmetic methods for merging
multiple fine-tuned models into a single unified model.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("model_merging")
class ModelMerging:
    """Unified model merging interface supporting multiple strategies.

    Args:
        method: Merging method ('ties', 'dare', 'slerp', 'task_arithmetic').
        weights: Per-model weights for weighted merging.
        density: Fraction of parameters to retain (TIES/DARE).
        temperature: Interpolation temperature for SLERP.

    Example:
        >>> merger = ModelMerging(method="ties", density=0.3)
        >>> merged = merger.merge(base_model, [ft_model_1, ft_model_2])
    """

    METHODS = ("ties", "dare", "slerp", "task_arithmetic")

    def __init__(
        self,
        method: str = "ties",
        weights: Optional[List[float]] = None,
        density: float = 0.3,
        temperature: float = 0.5,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Options: {self.METHODS}")
        self.method = method
        self.weights = weights
        self.density = density
        self.temperature = temperature

    def merge(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        weights: Optional[List[float]] = None,
    ) -> nn.Module:
        """Merge multiple fine-tuned models into a single model.

        Args:
            base_model: The pretrained base model.
            finetuned_models: List of fine-tuned model variants.
            weights: Override per-model merge weights.

        Returns:
            Merged model combining knowledge from all models.
        """
        merge_weights = weights or self.weights
        if merge_weights is None:
            merge_weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        if self.method == "ties":
            return self._ties_merge(base_model, finetuned_models, merge_weights)
        elif self.method == "dare":
            return self._dare_merge(base_model, finetuned_models, merge_weights)
        elif self.method == "slerp":
            if len(finetuned_models) != 2:
                raise ValueError("SLERP requires exactly 2 models to interpolate between")
            return self._slerp_merge(finetuned_models[0], finetuned_models[1])
        elif self.method == "task_arithmetic":
            return self._task_arithmetic_merge(base_model, finetuned_models, merge_weights)
        raise ValueError(f"Unknown method: {self.method}")

    def _ties_merge(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        weights: List[float],
    ) -> nn.Module:
        """TIES merging: Trim, Elect Sign, Disjoint Merge.

        1. Compute task vectors (delta from base)
        2. Trim: zero out low-magnitude deltas (keep top-k% by density)
        3. Elect sign: resolve sign conflicts via majority vote
        4. Disjoint merge: weighted average of agreeing parameters
        """
        base_sd = base_model.state_dict()
        task_vectors = []
        for model in finetuned_models:
            tv = {}
            for key, param in model.state_dict().items():
                if key in base_sd:
                    tv[key] = param - base_sd[key]
                else:
                    tv[key] = param
            task_vectors.append(tv)

        trimmed = self._trim_task_vectors(task_vectors)
        elected_signs = self._elect_signs(trimmed)
        merged_tv = self._disjoint_merge(trimmed, elected_signs, weights)

        merged_model = copy.deepcopy(base_model)
        merged_sd = merged_model.state_dict()
        for key in merged_tv:
            if key in merged_sd:
                merged_sd[key] = base_sd[key] + merged_tv[key]
        merged_model.load_state_dict(merged_sd)
        return merged_model

    def _trim_task_vectors(self, task_vectors: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Trim low-magnitude values, keeping only top density fraction."""
        trimmed = []
        for tv in task_vectors:
            trimmed_tv = {}
            for key, delta in tv.items():
                if delta.dtype.is_floating_point and delta.numel() > 0:
                    flat = delta.abs().flatten()
                    k = max(1, int(flat.numel() * self.density))
                    threshold = torch.topk(flat, k).values[-1]
                    mask = delta.abs() >= threshold
                    trimmed_tv[key] = delta * mask.float()
                else:
                    trimmed_tv[key] = delta
            trimmed.append(trimmed_tv)
        return trimmed

    @staticmethod
    def _elect_signs(task_vectors: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Elect majority sign for each parameter across task vectors."""
        elected = {}
        keys = task_vectors[0].keys()
        for key in keys:
            signs = torch.stack([torch.sign(tv[key]) for tv in task_vectors], dim=0)
            sign_sum = signs.sum(dim=0)
            elected[key] = torch.sign(sign_sum)
        return elected

    @staticmethod
    def _disjoint_merge(
        task_vectors: List[Dict[str, torch.Tensor]],
        elected_signs: Dict[str, torch.Tensor],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Merge only parameters that agree with the elected sign."""
        merged = {}
        for key in elected_signs:
            accumulated = torch.zeros_like(elected_signs[key])
            weight_sum = torch.zeros_like(elected_signs[key])
            for tv, w in zip(task_vectors, weights):
                agreement = (torch.sign(tv[key]) == elected_signs[key]).float()
                agreement[elected_signs[key] == 0] = 0.0
                accumulated += tv[key] * agreement * w
                weight_sum += agreement * w
            weight_sum = weight_sum.clamp(min=1e-8)
            merged[key] = accumulated / weight_sum
        return merged

    def _dare_merge(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        weights: List[float],
    ) -> nn.Module:
        """DARE merging: Drop And REscale.

        Randomly drops a fraction of delta parameters and rescales
        the remaining to preserve expected magnitude.
        """
        base_sd = base_model.state_dict()
        merged_model = copy.deepcopy(base_model)
        merged_sd = merged_model.state_dict()

        for key in merged_sd:
            if not merged_sd[key].dtype.is_floating_point:
                continue

            accumulated_delta = torch.zeros_like(merged_sd[key])
            for model, w in zip(finetuned_models, weights):
                model_sd = model.state_dict()
                if key not in model_sd:
                    continue
                delta = model_sd[key] - base_sd[key]
                drop_mask = (torch.rand_like(delta.float()) > (1.0 - self.density)).float()
                rescale_factor = 1.0 / max(self.density, 1e-8)
                dropped_delta = delta * drop_mask * rescale_factor
                accumulated_delta += dropped_delta * w

            merged_sd[key] = base_sd[key] + accumulated_delta

        merged_model.load_state_dict(merged_sd)
        return merged_model

    def _slerp_merge(self, model_a: nn.Module, model_b: nn.Module) -> nn.Module:
        """SLERP: Spherical Linear Interpolation between two models.

        Interpolates on the unit hypersphere in parameter space,
        preserving the norm geometry better than linear interpolation.
        """
        merged_model = copy.deepcopy(model_a)
        merged_sd = merged_model.state_dict()
        sd_a = model_a.state_dict()
        sd_b = model_b.state_dict()

        t = self.temperature

        for key in merged_sd:
            if not merged_sd[key].dtype.is_floating_point:
                continue

            va = sd_a[key].flatten().float()
            vb = sd_b[key].flatten().float()

            norm_a = va.norm()
            norm_b = vb.norm()

            if norm_a < 1e-8 or norm_b < 1e-8:
                merged_sd[key] = ((1.0 - t) * sd_a[key] + t * sd_b[key])
                continue

            va_unit = va / norm_a
            vb_unit = vb / norm_b

            cos_omega = torch.clamp(torch.dot(va_unit, vb_unit), -1.0, 1.0)
            omega = torch.acos(cos_omega)

            if omega.abs() < 1e-6:
                merged_sd[key] = ((1.0 - t) * sd_a[key] + t * sd_b[key])
                continue

            sin_omega = torch.sin(omega)
            coeff_a = torch.sin((1.0 - t) * omega) / sin_omega
            coeff_b = torch.sin(t * omega) / sin_omega

            merged_flat = coeff_a * va + coeff_b * vb
            interp_norm = (1.0 - t) * norm_a + t * norm_b
            merged_flat = merged_flat / merged_flat.norm() * interp_norm

            merged_sd[key] = merged_flat.reshape(sd_a[key].shape).to(sd_a[key].dtype)

        merged_model.load_state_dict(merged_sd)
        return merged_model

    def _task_arithmetic_merge(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        weights: List[float],
    ) -> nn.Module:
        """Task Arithmetic: weighted addition of task vectors to base model.

        Each task vector is (finetuned_params - base_params), and the
        final model is base + sum(weight_i * task_vector_i).
        """
        base_sd = base_model.state_dict()
        merged_model = copy.deepcopy(base_model)
        merged_sd = merged_model.state_dict()

        for key in merged_sd:
            if not merged_sd[key].dtype.is_floating_point:
                continue

            accumulated = torch.zeros_like(merged_sd[key])
            for model, w in zip(finetuned_models, weights):
                model_sd = model.state_dict()
                if key in model_sd:
                    task_vector = model_sd[key] - base_sd[key]
                    accumulated += w * task_vector

            merged_sd[key] = base_sd[key] + accumulated

        merged_model.load_state_dict(merged_sd)
        return merged_model

    def fuse(self, models: List[nn.Module], **kwargs) -> nn.Module:
        """Fuse models (strategy interface compatible).

        Uses first model as base and rest as fine-tuned variants.
        """
        if len(models) < 2:
            raise ValueError("Model merging requires at least 2 models")
        base = models[0]
        finetuned = models[1:]
        return self.merge(base, finetuned, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelMerging(method='{self.method}', density={self.density}, "
            f"temperature={self.temperature})"
        )
