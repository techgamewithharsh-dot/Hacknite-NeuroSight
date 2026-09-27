#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Real-Time Brain Tumor Progression & Pseudo-Progression Analyzer..."
streamlit run app.py --server.port=8501 --server.headless=false
