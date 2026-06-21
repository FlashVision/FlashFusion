# Contributing to FlashFusion

Thank you for your interest in contributing to FlashFusion! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/FlashVision/FlashFusion.git
cd FlashFusion
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 120 characters
- Follow PEP 8 conventions
- Add type hints to all public functions

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit a PR with a clear description

## Adding a New Fusion Strategy

1. Create a new file in `flashfusion/strategies/`
2. Register it using the `@STRATEGIES.register()` decorator
3. Add tests in `tests/test_strategies.py`
4. Update `flashfusion/strategies/__init__.py`
5. Add documentation in `docs/Fusion-Strategies.md`

## Adding a New Pipeline

1. Create a new file in `flashfusion/pipelines/`
2. Register it using the `@PIPELINES.register()` decorator
3. Add an example in `examples/`
4. Update `flashfusion/pipelines/__init__.py`

## Reporting Issues

- Use the GitHub issue tracker
- Include reproduction steps, expected vs actual behavior
- Include your environment details (OS, Python version, PyTorch version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
