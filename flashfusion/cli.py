"""FlashFusion CLI — Command-line interface for FlashFusion.

Usage:
    flashfusion version
    flashfusion settings
    flashfusion check
    flashfusion train --config configs/flashfusion_det_cls_320.yaml
    flashfusion predict --config configs/flashfusion_ensemble_320.yaml --source image.jpg
    flashfusion fuse --models model1.pt model2.pt --strategy wbf --source image.jpg
    flashfusion export --config configs/flashfusion_ensemble_320.yaml --format onnx
"""

import argparse
import sys
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser for FlashFusion CLI."""
    parser = argparse.ArgumentParser(
        prog="flashfusion",
        description="FlashFusion — Multi-model vision fusion CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show FlashFusion version")

    # settings
    subparsers.add_parser("settings", help="Show current settings and environment info")

    # check
    subparsers.add_parser("check", help="Verify installation and dependencies")

    # train
    train_parser = subparsers.add_parser("train", help="Train fusion layers")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, cuda:0)")
    train_parser.add_argument("--workers", type=int, default=4, help="Number of data workers")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Run multi-model fusion prediction")
    predict_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    predict_parser.add_argument("--source", type=str, required=True, help="Image/video/directory path")
    predict_parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, cuda:0)")
    predict_parser.add_argument("--save-dir", type=str, default="workspace/predict", help="Output directory")
    predict_parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")

    # fuse
    fuse_parser = subparsers.add_parser("fuse", help="Run fusion pipeline on inputs")
    fuse_parser.add_argument("--models", nargs="+", required=True, help="Paths to model weights")
    fuse_parser.add_argument("--strategy", type=str, default="wbf", help="Fusion strategy (wbf, voting, cascade)")
    fuse_parser.add_argument("--source", type=str, required=True, help="Image/video/directory path")
    fuse_parser.add_argument("--weights", nargs="+", type=float, default=None, help="Model weights for fusion")
    fuse_parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, cuda:0)")
    fuse_parser.add_argument("--save-dir", type=str, default="workspace/fuse", help="Output directory")

    # export
    export_parser = subparsers.add_parser("export", help="Export fused pipeline")
    export_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    export_parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, torchscript)")
    export_parser.add_argument("--output", type=str, default=None, help="Output file path")
    export_parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")

    return parser


def cmd_version() -> None:
    """Print FlashFusion version and system information."""
    from flashfusion import __version__

    print(f"FlashFusion v{__version__}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("PyTorch: not installed")

    try:
        import torchvision
        print(f"TorchVision: {torchvision.__version__}")
    except ImportError:
        print("TorchVision: not installed")

    sys.exit(0)


def cmd_settings() -> None:
    """Display current FlashFusion settings."""
    from flashfusion import __version__

    print(f"FlashFusion v{__version__}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {Path.cwd()}")

    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("\nPyTorch: not installed")

    sys.exit(0)


def cmd_check() -> None:
    """Check FlashFusion installation and dependencies."""
    print("Checking FlashFusion installation...\n")
    checks_passed = 0
    checks_failed = 0

    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
            checks_passed += 1
        except ImportError:
            print(f"  ✗ {name} — not installed")
            checks_failed += 1

    optional_deps = [
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    print("\nOptional dependencies:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  - {name} (not installed)")

    print(f"\nResults: {checks_passed} passed, {checks_failed} failed")
    sys.exit(0 if checks_failed == 0 else 1)


def cmd_train(args: argparse.Namespace) -> None:
    """Launch fusion training."""
    from flashfusion.cfg import get_config
    from flashfusion.engine.trainer import Trainer

    config = get_config(args.config)
    trainer = Trainer(config, device=args.device, workers=args.workers, resume=args.resume)
    trainer.train()


def cmd_predict(args: argparse.Namespace) -> None:
    """Run multi-model fusion prediction."""
    from flashfusion.cfg import get_config
    from flashfusion.engine.predictor import Predictor

    config = get_config(args.config)
    predictor = Predictor(config, device=args.device)
    predictor.predict(source=args.source, save_dir=args.save_dir, conf_threshold=args.conf_threshold)


def cmd_fuse(args: argparse.Namespace) -> None:
    """Run fusion pipeline directly on models."""
    from flashfusion.models.fusion import FlashFusion

    model = FlashFusion.from_models(
        model_paths=args.models,
        strategy=args.strategy,
        weights=args.weights,
        device=args.device,
    )
    results = model.predict(args.source)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fusion results saved to {save_dir}")
    for r in results:
        print(f"  {r}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export fused pipeline."""
    from flashfusion.cfg import get_config
    from flashfusion.engine.exporter import Exporter

    config = get_config(args.config)
    exporter = Exporter(config)
    output_path = exporter.export(format=args.format, output=args.output, simplify=args.simplify)
    print(f"Exported to: {output_path}")


def main() -> None:
    """FlashFusion CLI entry point."""
    parser = get_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": lambda: cmd_version(),
        "settings": lambda: cmd_settings(),
        "check": lambda: cmd_check(),
        "train": lambda: cmd_train(args),
        "predict": lambda: cmd_predict(args),
        "fuse": lambda: cmd_fuse(args),
        "export": lambda: cmd_export(args),
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn is None:
        parser.print_help()
        sys.exit(1)

    cmd_fn()


if __name__ == "__main__":
    main()
