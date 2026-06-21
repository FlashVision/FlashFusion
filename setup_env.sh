#!/usr/bin/env bash
# FlashFusion — Environment Setup Script
# Usage: bash setup_env.sh [--dev] [--all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "================================================"
echo "  FlashFusion Environment Setup"
echo "================================================"
echo ""

# Parse arguments
INSTALL_MODE="default"
for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_MODE="dev"
            ;;
        --all)
            INSTALL_MODE="all"
            ;;
        --help|-h)
            echo "Usage: bash setup_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev    Install with development dependencies"
            echo "  --all    Install with all optional dependencies"
            echo "  --help   Show this help message"
            exit 0
            ;;
    esac
done

# Check Python version
echo "[1/4] Checking Python version..."
PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "  Python $PYTHON_VERSION ✓"

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment at $VENV_DIR..."
if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists, skipping creation."
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created virtual environment ✓"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "  pip upgraded ✓"

# Install FlashFusion
echo ""
echo "[4/4] Installing FlashFusion ($INSTALL_MODE mode)..."
case $INSTALL_MODE in
    dev)
        pip install -e ".[dev]" --quiet
        ;;
    all)
        pip install -e ".[all]" --quiet
        ;;
    *)
        pip install -e . --quiet
        ;;
esac
echo "  FlashFusion installed ✓"

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Verify installation:"
echo "  flashfusion version"
echo ""
