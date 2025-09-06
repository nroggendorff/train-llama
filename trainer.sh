#!/bin/bash
DEVICE_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')

if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "No GPUs detected. Exiting..."
    exit 1
fi

: ${INIT:=0}
: ${INSTRUCT:=false}
echo "Using initialization value: $INIT"

echo "Preprocessing data..."
python3 prep.py
if [ $? -ne 0 ]; then
    echo "Preprocessing failed. Exiting..."
    exit 1
fi

echo "Done preprocessing, training on $DEVICE_COUNT devices.."
deepspeed --num_gpus=$DEVICE_COUNT train.py
if [ $? -ne 0 ]; then
    echo "Training failed. Exiting..."
    exit 1
fi

echo "Training complete. Exiting..."
python3 -c "from util import Conclusion; raise Conclusion('Training complete.')"
