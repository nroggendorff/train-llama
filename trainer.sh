#!/usr/bin/env bash
set -euo pipefail

echo "=================================="
echo "Train-LLaMA Startup"
echo "=================================="
echo "Timestamp: $(date)"
echo ""

echo "Checking Hugging Face authentication..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Authentication is required for pushing to hub."
    exit 1
fi

python3 -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f' Authenticated as: {user[\"name\"]}')
except Exception as e:
    print(f' Authentication failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: HuggingFace authentication failed"
    exit 1
fi

echo ""
echo "Checking if running in Hugging Face Space..."
if [ -n "${SPACE_ID:-}" ]; then
    echo " Running in Hugging Face Space: $SPACE_ID"
else:
    echo " Not running in a Hugging Face Space"
    exit 1
fi

echo ""
echo "Detecting GPU configuration..."
DEVICE_COUNT=$(python3 - <<'PY'
try:
    import torch
    count = torch.cuda.device_count()
    print(count)
    if count > 0:
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)", file=__import__('sys').stderr)
except Exception as e:
    print(0)
    print(f"Error: {e}", file=__import__('sys').stderr)
PY
)

echo "Detected GPU count: $DEVICE_COUNT"

if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo " No GPUs detected. Cannot proceed with training."
    exit 1
fi

echo ""
echo "Configuration:"
: ${INIT:=0}
: ${INST:=false}
: ${USE_QUANT:=false}
echo "  INIT: $INIT"
echo "  INST: $INST"
echo "  USE_QUANT: $USE_QUANT"

echo ""
echo "Running pre-flight checks..."
if [ -f "preflight.py" ]; then
    if [ -x "venv/bin/python" ]; then
        venv/bin/python preflight.py
    else
        python3 preflight.py
    fi

    PREFLIGHT_EXIT=$?
    if [ $PREFLIGHT_EXIT -ne 0 ]; then
        echo " Pre-flight checks failed. Please review errors above."
        exit 1
    fi
    echo " Pre-flight checks passed"
else
    echo "⚠ preflight.py not found, skipping pre-flight checks"
fi

echo ""
echo "Running environment validation..."
if [ -f "validate_setup.py" ]; then
    if [ -x "venv/bin/python" ]; then
        venv/bin/python validate_setup.py
    else
        python3 validate_setup.py
    fi

    if [ $? -ne 0 ]; then
        echo " Validation failed. Please check the errors above."
        exit 1
    fi
    echo " Validation passed"
else
    echo "⚠ validate_setup.py not found, skipping validation"
fi

echo ""
echo "=================================="
echo "Starting Training Pipeline"
echo "=================================="

python3 -c "import time; open('.timer_start', 'w').write(str(time.time()))"
echo " Timer started"

echo ""
echo "[PHASE 1] Data Preprocessing"
echo "=================================="

if [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python3"
fi

$PYTHON_BIN prep.py

if [ $? -ne 0 ]; then
    echo " Preprocessing failed"
    exit 1
fi

echo " Preprocessing complete"

echo ""
echo "[PHASE 2] Model Training"
echo "=================================="
echo "Training on $DEVICE_COUNT GPU(s)..."

if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo "Using distributed training with DeepSpeed"
    deepspeed --num_gpus=$DEVICE_COUNT train.py
else
    echo "Using single GPU training"
    deepspeed --num_gpus=1 train.py
fi

TRAINING_EXIT_CODE=$?

echo ""
echo "=================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo " Training completed successfully"
else
    echo " Training failed with exit code $TRAINING_EXIT_CODE"
fi
echo "=================================="

if [ -f ".timer_start" ]; then
    START_TIME=$(cat .timer_start)
    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{($END_TIME - $START_TIME) / 60:.1f}')")
    echo "Total runtime: ${DURATION} minutes"
fi

echo ""
echo "Resetting space configuration..."
python3 -c "from util import Space; space = Space(); space.reset()"

if [ $? -ne 0 ]; then
    echo "⚠ Warning: Failed to reset space configuration"
fi

echo ""
echo "=================================="
echo "Pipeline Complete"
echo "=================================="
echo "Timestamp: $(date)"

exit $TRAINING_EXIT_CODE
