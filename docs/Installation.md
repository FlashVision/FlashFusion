# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchVision 0.15+

## Install from Source

```bash
git clone https://github.com/FlashVision/FlashFusion.git
cd FlashFusion
pip install -e .
```

## Install with Optional Dependencies

```bash
# With ONNX export support
pip install -e ".[export]"

# With analytics (matplotlib, pandas)
pip install -e ".[analytics]"

# With FlashVision model support
pip install -e ".[flash]"

# Everything
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"
```

## Docker

```bash
cd docker
docker compose up -d
docker exec -it flashfusion bash
```

## Verify Installation

```bash
flashfusion check
```

This will verify all required and optional dependencies are installed correctly.

## Environment Setup

For a quick environment setup with conda:

```bash
bash setup_env.sh
```

## GPU Support

FlashFusion automatically detects CUDA availability. To verify:

```bash
flashfusion version
```

This shows your PyTorch version and CUDA status.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: torch` | Install PyTorch: `pip install torch torchvision` |
| `CUDA not available` | Install CUDA-enabled PyTorch from pytorch.org |
| `onnx export fails` | Install export deps: `pip install -e ".[export]"` |
| `matplotlib not found` | Install analytics deps: `pip install -e ".[analytics]"` |
