#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Neuro-Oncology Diagnostic Terminal & FastAPI Server"
echo " Web UI & REST API available at: http://localhost:8000"
echo "=========================================================="

python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
