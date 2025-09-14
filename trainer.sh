#!/usr/bin/env bash

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
: ${INSTRUCT:=false}
echo "Using initialization value: $INIT"

echo "Preprocessing data..."
if [ -x ".venv/bin/python" ]; then
    .venv/bin/python prep.py
else
    python3 prep.py
fi

echo "Done preprocessing, training on $DEVICE_COUNT devices.."

deepspeed --num_gpus=$DEVICE_COUNT train.py

echo "Training complete. Exiting..."
python3 -c "from util import Conclusion; raise Conclusion('Training complete.')"
