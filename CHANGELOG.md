# Changelog

All notable changes to FlashFusion will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-01

### Added
- Initial release of FlashFusion
- Core fusion model (`FlashFusion`) supporting multiple base models
- Weighted Box Fusion (WBF) strategy for detection ensembles
- Voting ensemble strategy (majority and soft voting)
- Cascade fusion with early exit support
- Stacking strategy with meta-learner
- NMS-based fusion for overlapping detections
- Registry system for pluggable components
- CLI tool with train, predict, fuse, and export commands
- Comprehensive YAML configuration system
- Docker support for containerized deployment
- CI/CD with GitHub Actions

### Planned (not yet fully implemented)
- Pre-built pipelines: Det→Cls, Det+Seg, Multi-Task (models require FlashVision weights)
- Training engine for fusion layer fine-tuning (requires dataset configuration)
- ONNX export for fused pipelines (requires trained checkpoints)
- LoRA/QLoRA support for efficient adaptation
- Full documentation and examples
