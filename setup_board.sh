#!/usr/bin/env bash
# Create the board venv (ONNX backend only) and install requirements-board.txt.
# Usage: ./setup_board.sh [venv-dir]      (default: .venv next to this script)
set -euo pipefail

cd "$(dirname "$0")"
VENV="${1:-.venv}"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import venv, ensurepip" 2>/dev/null; then
    echo "Python venv support missing. On Ubuntu: sudo apt install python3-venv" >&2
    exit 1
fi

echo "Creating venv in $VENV with $("$PYTHON" --version)"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements-board.txt

"$VENV/bin/python" -c "import cv2, lap, numpy, onnxruntime; \
print('OK: numpy', numpy.__version__, '| opencv', cv2.__version__, \
'| onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"

if [ ! -f yolov8n.onnx ]; then
    echo "NOTE: yolov8n.onnx not found here. Export it on the laptop (see README) and copy it over."
fi
echo "Run: $VENV/bin/python main.py --backend onnx --no-display --stream --source 0"
