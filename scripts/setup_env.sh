#!/bin/bash
set -e

echo "=========================================================="
echo " Setting up Neuro-Oncology Diagnostic Suite (Hacknite 2026)"
echo "=========================================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="/opt/homebrew/bin/python3.12"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON_BIN -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing project dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Generating benchmark synthetic 3D multi-modal NIfTI cases..."
python data_generator/generate_synthetic_cases.py

echo "Running unit verification tests..."
python -m unittest discover -s tests -p "test_*.py"

echo ""
echo "=========================================================="
echo "✅ Environment setup complete! Launch the app with:"
echo "   ./scripts/run_app.sh"
echo "=========================================================="
