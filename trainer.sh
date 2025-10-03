#!/usr/bin/env bash
set -euo pipefail

echo "Checking Hugging Face authentication..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set. Authentication is required for pushing to hub."
    exit 1
fi

python3 -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f'Authenticated as: {user[\"name\"]}')
except Exception as e:
    print(f'Authentication failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo "Checking if running in Hugging Face Space..."
if [ -n "${SPACE_ID:-}" ]; then
    echo "Running in Hugging Face Space: $SPACE_ID"
else
    echo "Not running in a Hugging Face Space"
    exit 1
fi

DEVICE_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)

echo "Detected GPU count: $DEVICE_COUNT"

if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "No GPUs detected. Exiting..."
    exit 1
fi

: ${INIT:=0}
: ${INST:=false}
echo "Using initialization value: $INIT"

echo "Preprocessing data..."
if [ -x "venv/bin/python" ]; then
    venv/bin/python prep.py
else
    python3 prep.py
fi

echo "Done preprocessing, training on $DEVICE_COUNT devices.."

deepspeed --num_gpus=$DEVICE_COUNT train.py

echo "Training complete. Exiting..."
python3 -c "from util import Space; space = Space(); space.reset()"
