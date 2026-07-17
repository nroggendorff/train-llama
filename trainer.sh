#!/usr/bin/env bash
set -euo pipefail

echo "Checking if running in Hugging Face Space..."
if [ -n "${SPACE_ID:-}" ]; then
    IS_SPACE=true
    echo "Running in Hugging Face Space: $SPACE_ID"
else
    IS_SPACE=false
    echo "Running locally"
fi

if [ -n "${HF_TOKEN:-}" ]; then
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
elif [ "$IS_SPACE" = true ]; then
    echo "ERROR: HF_TOKEN environment variable is not set. Authentication is required for pushing to hub."
    exit 1
else
    echo "HF_TOKEN not set, skipping authentication and hub push"
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

: ${INST:=false}

if [ "$IS_SPACE" = true ]; then
    python3 -c "import time; open('.timer_start', 'w').write(str(time.time()))"
    echo "Timer started"

    echo "Checking training state repo..."
    python3 -c "
from util import get_state_repo_id, ensure_state_repo
repo_id, is_custom = get_state_repo_id()
try:
    ensure_state_repo(repo_id, is_custom)
    print(f'Training state repo ready: {repo_id}' + (' (custom)' if is_custom else ''))
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
"
fi

echo "Preprocessing data..."
if [ -x "venv/bin/python" ]; then
    venv/bin/python prep.py
else
    python3 prep.py
fi

echo "Done preprocessing, training on $DEVICE_COUNT devices.."

deepspeed --num_gpus=$DEVICE_COUNT train.py

echo "Training run finished. Exiting.."
