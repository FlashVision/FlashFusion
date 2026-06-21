"""Test-Time Augmentation (TTA) for inference-time ensemble.

Applies multiple augmentations at inference, runs predictions on each,
and fuses the results for improved robustness and accuracy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfusion.registry import STRATEGIES


@STRATEGIES.register("tta")
class TestTimeAugmentation:
    """Test-Time Augmentation for multi-scale and geometric ensemble.

    Applies augmentations (multi-scale, flips, rotations) at inference,
    collects predictions, and merges them for more robust outputs.

    Args:
        scales: List of scale factors for multi-scale inference.
        flip_horizontal: Enable horizontal flip augmentation.
        flip_vertical: Enable vertical flip augmentation.
        rotations: List of rotation angles in degrees (0, 90, 180, 270).
        merge_mode: How to merge predictions ('mean', 'max', 'voting').
        batch_augmentations: Process all augmentations in a single batch.

    Example:
        >>> tta = TestTimeAugmentation(scales=[0.8, 1.0, 1.2], flip_horizontal=True)
        >>> result = tta.predict(model, input_tensor)
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        rotations: Optional[List[int]] = None,
        merge_mode: str = "mean",
        batch_augmentations: bool = True,
    ) -> None:
        self.scales = scales or [1.0]
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.rotations = rotations or [0]
        self.merge_mode = merge_mode
        self.batch_augmentations = batch_augmentations

    def predict(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        post_process_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run TTA prediction on inputs.

        Args:
            model: The model to run predictions with.
            inputs: Input tensor of shape (B, C, H, W).
            post_process_fn: Optional function to post-process raw model outputs.

        Returns:
            Fused predictions from all augmented variants.
        """
        model.eval()
        augmented_inputs = self._generate_augmentations(inputs)

        all_outputs = []
        with torch.no_grad():
            can_batch = (
                self.batch_augmentations
                and len(augmented_inputs) > 0
                and len(set(aug["tensor"].shape[1:] for aug in augmented_inputs)) == 1
            )
            if can_batch:
                batch = torch.cat([aug["tensor"] for aug in augmented_inputs], dim=0)
                batch_output = model(batch)
                chunk_sizes = [aug["tensor"].shape[0] for aug in augmented_inputs]
                split_outputs = torch.split(batch_output, chunk_sizes, dim=0)
                for aug_info, output in zip(augmented_inputs, split_outputs):
                    reversed_output = self._reverse_augmentation(output, aug_info)
                    all_outputs.append(reversed_output)
            else:
                for aug_info in augmented_inputs:
                    output = model(aug_info["tensor"])
                    reversed_output = self._reverse_augmentation(output, aug_info)
                    all_outputs.append(reversed_output)

        fused = self._merge_outputs(all_outputs)

        if post_process_fn is not None:
            fused = post_process_fn(fused)

        return fused

    def _generate_augmentations(self, inputs: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate all augmentation variants of the input.

        Returns list of dicts with 'tensor' and augmentation metadata for reversal.
        """
        _, _, h, w = inputs.shape
        augmentations = []

        for scale in self.scales:
            for rotation in self.rotations:
                for h_flip in [False, True] if self.flip_horizontal else [False]:
                    for v_flip in [False, True] if self.flip_vertical else [False]:
                        aug_tensor = inputs.clone()

                        if scale != 1.0:
                            new_h = int(h * scale)
                            new_w = int(w * scale)
                            aug_tensor = F.interpolate(
                                aug_tensor,
                                size=(new_h, new_w),
                                mode="bilinear",
                                align_corners=False,
                            )

                        if h_flip:
                            aug_tensor = torch.flip(aug_tensor, dims=[3])

                        if v_flip:
                            aug_tensor = torch.flip(aug_tensor, dims=[2])

                        if rotation != 0:
                            aug_tensor = self._rotate_tensor(aug_tensor, rotation)

                        augmentations.append(
                            {
                                "tensor": aug_tensor,
                                "scale": scale,
                                "rotation": rotation,
                                "h_flip": h_flip,
                                "v_flip": v_flip,
                                "original_size": (h, w),
                            }
                        )

        return augmentations

    def _reverse_augmentation(self, output: torch.Tensor, aug_info: Dict[str, Any]) -> torch.Tensor:
        """Reverse augmentation on model output to align with original space.

        Only applies spatial reversals if output retains spatial dimensions (4D).
        For classification outputs (2D), no spatial reversal is needed.
        """
        if output.dim() < 4:
            return output

        if aug_info["rotation"] != 0:
            output = self._rotate_tensor(output, -aug_info["rotation"])

        if aug_info["v_flip"]:
            output = torch.flip(output, dims=[2])

        if aug_info["h_flip"]:
            output = torch.flip(output, dims=[3])

        if aug_info["scale"] != 1.0:
            orig_h, orig_w = aug_info["original_size"]
            output = F.interpolate(
                output,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

        return output

    def _merge_outputs(self, outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Merge outputs from all augmentation variants.

        Args:
            outputs: List of tensors from each augmentation.

        Returns:
            Dictionary with fused predictions.
        """
        if not outputs:
            return {"predictions": torch.tensor([])}

        stacked = torch.stack(outputs, dim=0)

        if self.merge_mode == "mean":
            fused = stacked.mean(dim=0)
        elif self.merge_mode == "max":
            fused = stacked.max(dim=0).values
        elif self.merge_mode == "voting":
            votes = stacked.argmax(dim=-1)
            fused = torch.mode(votes, dim=0).values
            return {
                "predictions": fused,
                "labels": fused,
                "num_augmentations": len(outputs),
            }
        else:
            fused = stacked.mean(dim=0)

        if fused.dim() >= 2 and fused.shape[-1] > 1:
            probabilities = F.softmax(fused, dim=-1)
            scores, labels = torch.max(probabilities, dim=-1)
            return {
                "predictions": fused,
                "probabilities": probabilities,
                "labels": labels,
                "scores": scores,
                "num_augmentations": len(outputs),
            }

        return {
            "predictions": fused,
            "num_augmentations": len(outputs),
        }

    @staticmethod
    def _rotate_tensor(tensor: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate a 4D tensor by 90-degree multiples.

        Args:
            tensor: Input tensor of shape (B, C, H, W).
            angle: Rotation angle (must be multiple of 90).
        """
        angle = angle % 360
        if angle == 0:
            return tensor
        elif angle == 90:
            return tensor.transpose(2, 3).flip(3)
        elif angle == 180:
            return tensor.flip(2).flip(3)
        elif angle == 270:
            return tensor.transpose(2, 3).flip(2)
        else:
            raise ValueError(f"Only 90-degree multiples supported, got {angle}")

    def fuse(self, predictions: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Strategy interface: fuse pre-computed TTA predictions."""
        all_preds = []
        for pred in predictions:
            if "logits" in pred:
                all_preds.append(pred["logits"])
            elif "predictions" in pred:
                all_preds.append(pred["predictions"])

        if not all_preds:
            return {"predictions": torch.tensor([])}

        return self._merge_outputs(all_preds)

    def __repr__(self) -> str:
        return (
            f"TestTimeAugmentation(scales={self.scales}, "
            f"flip_h={self.flip_horizontal}, flip_v={self.flip_vertical}, "
            f"rotations={self.rotations}, merge='{self.merge_mode}')"
        )
