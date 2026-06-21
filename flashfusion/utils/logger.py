"""FlashFusion logging utilities — Structured logger and metric tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "flashfusion",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name.
        log_file: Optional path to log file. If provided, logs to both
                  file and stdout.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        fmt: Log message format string.

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> logger = setup_logger("flashfusion.train", log_file="train.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class AverageMeter:
    """Track running average of a metric over batches/epochs.

    Maintains count, sum, and computes mean on demand. Useful for
    tracking loss and metric values during training.

    Example:
        >>> meter = AverageMeter("loss")
        >>> meter.update(0.5, n=32)
        >>> meter.update(0.3, n=32)
        >>> print(f"{meter.name}: {meter.avg:.4f}")
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        """Reset all tracked values to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with new value.

        Args:
            val: New metric value.
            n: Number of samples this value represents (for weighted averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name='{self.name}', avg={self.avg:.4f}, count={self.count})"
