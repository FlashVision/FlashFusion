"""FlashFusion Callbacks — Hook system for training events.

Provides a callback mechanism for logging, checkpointing, early stopping,
and custom user-defined actions during training.
"""

from typing import Any, Dict, List, Optional


class Callback:
    """Base callback class for FlashFusion training events.

    Subclass this to implement custom behavior at various training stages.

    Example:
        >>> class MyCallback(Callback):
        ...     def on_epoch_end(self, trainer, epoch, **kwargs):
        ...         print(f"Epoch {epoch} finished!")
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called at the beginning of training."""

    def on_train_end(self, trainer: Any, results: Dict[str, Any]) -> None:
        """Called at the end of training."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Called at the end of each epoch."""

    def on_batch_start(self, trainer: Any, batch: int) -> None:
        """Called at the start of each batch."""

    def on_batch_end(self, trainer: Any, batch: int, loss: float = 0.0) -> None:
        """Called at the end of each batch."""

    def on_validation_start(self, trainer: Any) -> None:
        """Called before validation."""

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float] = None) -> None:
        """Called after validation."""


class CallbackHandler:
    """Manages and dispatches callbacks during training.

    Args:
        callbacks: Optional list of Callback instances to register.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self._callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Register a new callback."""
        self._callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a registered callback."""
        self._callbacks.remove(callback)

    def on_train_start(self, trainer: Any) -> None:
        """Dispatch train_start to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any, results: Dict[str, Any]) -> None:
        """Dispatch train_end to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_end(trainer, results)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Dispatch epoch_start to all callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Dispatch epoch_end to all callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_end(trainer, epoch, **kwargs)

    def on_batch_start(self, trainer: Any, batch: int) -> None:
        """Dispatch batch_start to all callbacks."""
        for cb in self._callbacks:
            cb.on_batch_start(trainer, batch)

    def on_batch_end(self, trainer: Any, batch: int, loss: float = 0.0) -> None:
        """Dispatch batch_end to all callbacks."""
        for cb in self._callbacks:
            cb.on_batch_end(trainer, batch, loss=loss)

    def on_validation_start(self, trainer: Any) -> None:
        """Dispatch validation_start to all callbacks."""
        for cb in self._callbacks:
            cb.on_validation_start(trainer)

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float] = None) -> None:
        """Dispatch validation_end to all callbacks."""
        for cb in self._callbacks:
            cb.on_validation_end(trainer, metrics=metrics)


class EarlyStoppingCallback(Callback):
    """Stop training when a metric stops improving.

    Sets a `should_stop` flag that the Trainer checks after each epoch,
    instead of raising an exception.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        metric_name: Metric to monitor.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, metric_name: str = "primary_metric"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self._best_value = 0.0
        self._counter = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Check for improvement and set stop flag if needed."""
        val_metrics = kwargs.get("val_metrics", {})
        current = val_metrics.get(self.metric_name, 0.0)

        if current > self._best_value + self.min_delta:
            self._best_value = current
            self._counter = 0
            self.should_stop = False
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.should_stop = True


class LoggingCallback(Callback):
    """Log training progress to console."""

    def on_epoch_end(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Log epoch results."""
        train_loss = kwargs.get("train_loss", 0.0)
        val_metrics = kwargs.get("val_metrics", {})
        primary = val_metrics.get("primary_metric", 0.0)
        print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, metric={primary:.4f}")
