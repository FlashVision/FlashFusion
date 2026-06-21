"""FlashFusion Trainer — Trains fusion layers on top of frozen base models.

The trainer freezes all base model parameters and only trains the fusion
components (heads, necks, strategy parameters) using the configured dataset.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashfusion.cfg.config import FusionConfig
from flashfusion.engine.callbacks import CallbackHandler


class Trainer:
    """FlashFusion training engine.

    Trains fusion layers (heads, necks, learned weights) while keeping
    base model backbones frozen.

    Args:
        config: FusionConfig instance with training parameters.
        device: Target device (auto, cpu, cuda).
        workers: Number of data loading workers.
        resume: Path to checkpoint for resuming training.

    Example:
        >>> from flashfusion import Trainer
        >>> from flashfusion.cfg import get_config
        >>> config = get_config("configs/flashfusion_det_cls_320.yaml")
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: FusionConfig,
        device: str = "auto",
        workers: int = 4,
        resume: Optional[str] = None,
    ):
        self.config = config
        self.device = self._resolve_device(device)
        self.workers = workers
        self.resume = resume
        self.callbacks = CallbackHandler()

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.start_epoch = 0
        self.best_metric = 0.0

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with training results and metrics.
        """
        self._setup()
        self.callbacks.on_train_start(self)

        results = {}
        for epoch in range(self.start_epoch, self.config.epochs):
            self.callbacks.on_epoch_start(self, epoch)

            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)

            self.callbacks.on_epoch_end(self, epoch, train_loss=train_loss, val_metrics=val_metrics)

            if self._should_stop():
                break

            if val_metrics.get("primary_metric", 0) > self.best_metric:
                self.best_metric = val_metrics["primary_metric"]
                self._save_checkpoint(epoch, is_best=True)

            self._save_checkpoint(epoch, is_best=False)

            results[f"epoch_{epoch}"] = {
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }

        self.callbacks.on_train_end(self, results)
        return results

    def _setup(self) -> None:
        """Initialize model, optimizer, data loaders, and callbacks."""
        self._build_model()
        self._build_optimizer()
        self._build_dataloaders()

        if self.resume:
            self._load_checkpoint(self.resume)

    def _build_model(self) -> None:
        """Build the fusion model from config."""
        from flashfusion.models.fusion import FlashFusion

        model_sources = [m.source for m in self.config.models]
        self.model = FlashFusion(
            models=model_sources,
            input_size=self.config.input_size,
            device=str(self.device),
        )
        self.model.to(self.device)

    def _build_optimizer(self) -> None:
        """Build optimizer targeting only trainable (fusion) parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        weight_decay = getattr(self.config, "weight_decay", 1e-4)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
        )

    def _should_stop(self) -> bool:
        """Check if any callback has signaled early stopping."""
        for cb in self.callbacks._callbacks:
            if hasattr(cb, "should_stop") and cb.should_stop:
                return True
        return False

    def _build_dataloaders(self) -> None:
        """Build training and validation data loaders."""
        from flashfusion.data.dataloader import create_train_val_loaders

        data_path = getattr(self.config, "data_path", None)
        if data_path is None:
            raise ValueError(
                "Dataset loading requires a configured dataset path. "
                "Set 'data.train' in your config YAML or config.data_path."
            )

        self.train_loader, self.val_loader = create_train_val_loaders(
            root=data_path,
            batch_size=self.config.batch_size,
            input_size=self.config.input_size,
            num_workers=self.workers,
            val_split=getattr(self.config, "val_split", 0.2),
        )

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch in pbar:
            self.optimizer.zero_grad()

            images = batch["images"].to(self.device)
            targets = batch["targets"]

            outputs = self.model(images)
            loss = self._compute_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        from flashfusion.engine.validator import Validator

        validator = Validator(self.config, device=str(self.device))
        return validator.validate(self.model, self.val_loader)

    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """Compute fusion training loss using FusionLoss."""
        if not hasattr(self, "_loss_fn"):
            from flashfusion.losses.fusion_loss import FusionLoss
            self._loss_fn = FusionLoss().to(self.device)

        predictions = outputs if isinstance(outputs, dict) else {"boxes": outputs}
        target_dict = targets if isinstance(targets, dict) else {"boxes": targets}

        return self._loss_fn(predictions, target_dict)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        torch.save(checkpoint, save_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, save_dir / "best.pt")

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint for resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("best_metric", 0.0)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
