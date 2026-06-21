"""Multi-task parallel pipeline for running all vision tasks simultaneously.

Runs detection, classification, segmentation, and OCR models in parallel
and returns a unified scene understanding result.
"""

import concurrent.futures
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from flashfusion.registry import PIPELINES


@PIPELINES.register("multi_task")
class MultiTaskPipeline:
    """Multi-task parallel execution pipeline.

    Runs multiple vision tasks (detection, classification, segmentation, OCR)
    in parallel using thread/process pools, then merges results.

    Args:
        tasks: Dictionary mapping task names to model identifiers.
        parallel: Whether to run tasks in parallel (True) or sequential.
        max_workers: Maximum number of parallel workers.
        device: Target device.

    Example:
        >>> pipeline = MultiTaskPipeline(
        ...     tasks={
        ...         "detection": "flashdet-m",
        ...         "classification": "flashcls-m",
        ...         "segmentation": "flashseg-m",
        ...     },
        ...     parallel=True,
        ... )
        >>> results = pipeline("image.jpg")
    """

    def __init__(
        self,
        tasks: Optional[Dict[str, Union[str, Any]]] = None,
        parallel: bool = True,
        max_workers: int = 4,
        device: str = "auto",
    ):
        self.tasks = tasks or {}
        self.parallel = parallel
        self.max_workers = max_workers
        self.device = self._resolve_device(device)
        self._task_runners: Dict[str, Callable] = {}

    def __call__(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Run the multi-task pipeline.

        Args:
            source: Input image path or array.

        Returns:
            Dictionary with results for each task.
        """
        return self.predict(source)

    def predict(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Run all tasks on the input.

        Args:
            source: Input image.

        Returns:
            Dictionary mapping task names to their results.
        """
        image = self._load_image(source)

        if self.parallel and len(self.tasks) > 1:
            return self._run_parallel(image)
        else:
            return self._run_sequential(image)

    def _run_parallel(self, image: np.ndarray) -> Dict[str, Any]:
        """Run all tasks in parallel using thread pool."""
        results: Dict[str, Any] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task_name, model_id in self.tasks.items():
                future = executor.submit(self._run_task, task_name, model_id, image)
                futures[future] = task_name

            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    results[task_name] = {"error": str(e)}

        return results

    def _run_sequential(self, image: np.ndarray) -> Dict[str, Any]:
        """Run all tasks sequentially."""
        results: Dict[str, Any] = {}
        for task_name, model_id in self.tasks.items():
            try:
                results[task_name] = self._run_task(task_name, model_id, image)
            except Exception as e:
                results[task_name] = {"error": str(e)}
        return results

    def _run_task(self, task_name: str, model_id: Any, image: np.ndarray) -> Dict[str, Any]:
        """Run a single task.

        Args:
            task_name: Name of the task.
            model_id: Model identifier or module.
            image: Input image.

        Returns:
            Task results.
        """
        if task_name in self._task_runners:
            return self._task_runners[task_name](image)

        model = self._resolve_model(model_id)
        if model is None:
            return {"error": f"Cannot load model for task '{task_name}' from '{model_id}'"}

        import cv2

        resized = cv2.resize(image, (320, 320))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor)

        if isinstance(output, dict):
            result = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu().numpy()
                else:
                    result[k] = v
            return result
        elif isinstance(output, torch.Tensor):
            return {"output": output.cpu().numpy()}

        return {"output": output}

    def _resolve_model(self, model_id: Any) -> Optional[torch.nn.Module]:
        """Resolve a model identifier to a loaded nn.Module."""
        if isinstance(model_id, torch.nn.Module):
            model_id.to(self.device)
            model_id.eval()
            return model_id

        from pathlib import Path as _Path

        path = _Path(str(model_id))
        if path.exists() and path.suffix in (".pt", ".pth"):
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                checkpoint.to(self.device)
                checkpoint.eval()
                return checkpoint
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model = checkpoint["model"]
                model.to(self.device)
                model.eval()
                return model

        return None

    def add_task_runner(self, task_name: str, runner: Callable) -> None:
        """Register a custom task runner function.

        Args:
            task_name: Name of the task.
            runner: Callable that takes an image and returns results.
        """
        self._task_runners[task_name] = runner

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from source."""
        if isinstance(source, np.ndarray):
            return source
        import cv2

        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {source}")
        return img

    @property
    def task_names(self) -> List[str]:
        """Return list of configured task names."""
        return list(self.tasks.keys())

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self) -> str:
        return f"MultiTaskPipeline(tasks={list(self.tasks.keys())}, parallel={self.parallel})"
