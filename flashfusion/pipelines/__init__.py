"""FlashFusion pre-built multi-model pipelines."""

from flashfusion.pipelines.det_cls_pipeline import DetClsPipeline
from flashfusion.pipelines.det_seg_pipeline import DetSegPipeline
from flashfusion.pipelines.multi_task_pipeline import MultiTaskPipeline

__all__ = ["DetClsPipeline", "DetSegPipeline", "MultiTaskPipeline"]
