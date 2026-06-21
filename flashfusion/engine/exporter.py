"""FlashFusion Exporter — Exports fused pipelines to ONNX and TorchScript.

Handles model tracing, ONNX conversion, simplification, and validation
of the exported fused pipeline.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from flashfusion.cfg.config import FusionConfig


class Exporter:
    """FlashFusion export engine.

    Exports the fused model pipeline to ONNX or TorchScript format
    for deployment on edge devices or inference servers.

    Args:
        config: FusionConfig instance.

    Example:
        >>> from flashfusion import Exporter
        >>> from flashfusion.cfg import get_config
        >>> config = get_config("configs/flashfusion_ensemble_320.yaml")
        >>> exporter = Exporter(config)
        >>> path = exporter.export(format="onnx")
    """

    SUPPORTED_FORMATS = ("onnx", "torchscript")

    def __init__(self, config: FusionConfig):
        self.config = config
        self.model: Optional[nn.Module] = None

    def export(
        self,
        format: str = "onnx",
        output: Optional[str] = None,
        simplify: bool = False,
        opset_version: int = 17,
    ) -> str:
        """Export the fused model to the specified format.

        Args:
            format: Export format ('onnx' or 'torchscript').
            output: Output file path. Auto-generated if None.
            simplify: Whether to simplify the ONNX model.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported model file.
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Use one of {self.SUPPORTED_FORMATS}")

        self._build_model()
        self.model.eval()

        if output is None:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            output = str(save_dir / f"flashfusion.{format}")

        if format == "onnx":
            return self._export_onnx(output, simplify=simplify, opset_version=opset_version)
        elif format == "torchscript":
            return self._export_torchscript(output)

        return output

    def _build_model(self) -> None:
        """Build the fusion model from config for export."""
        from flashfusion.models.fusion import FlashFusion

        model_sources = []
        for m in self.config.models:
            if m.checkpoint and Path(m.checkpoint).exists():
                model_sources.append(m.checkpoint)
            else:
                model_sources.append(m.source)

        self.model = FlashFusion(
            models=model_sources,
            input_size=self.config.input_size,
            device="cpu",
        )

        checkpoint_path = Path(self.config.save_dir) / "best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def _export_onnx(
        self,
        output_path: str,
        simplify: bool = False,
        opset_version: int = 17,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Output file path.
            simplify: Whether to run onnxsim.
            opset_version: ONNX opset version.

        Returns:
            Path to the exported ONNX file.
        """
        dummy_input = self._get_dummy_input()

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        if simplify:
            self._simplify_onnx(output_path)

        self._validate_onnx(output_path)
        return output_path

    def _export_torchscript(self, output_path: str) -> str:
        """Export model to TorchScript format."""
        dummy_input = self._get_dummy_input()
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(output_path)
        return output_path

    def _get_dummy_input(self) -> torch.Tensor:
        """Create dummy input tensor for tracing."""
        h, w = self.config.input_size
        return torch.randn(1, 3, h, w)

    def _simplify_onnx(self, model_path: str) -> None:
        """Simplify ONNX model using onnxsim."""
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(model_path)
            simplified, check = simplify(model)
            if check:
                onnx.save(simplified, model_path)
        except ImportError:
            print("Warning: onnxsim not installed. Skipping simplification.")

    def _validate_onnx(self, model_path: str) -> None:
        """Validate the exported ONNX model."""
        try:
            import onnx

            model = onnx.load(model_path)
            onnx.checker.check_model(model)
        except ImportError:
            pass
