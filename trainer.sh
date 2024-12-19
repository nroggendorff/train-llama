#!/bin/bash
DEVICE_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')

if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "No GPUs detected. Exiting..."
    exit 1
fi

: ${INIT:=0}
echo "Using initializtion value: $INIT"

sed -i "s/init\": 0/init\": $INIT/g" config.json

echo "Preprocessing data..."
python prep.py
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
python -c "from util import Conclusion; raise Conclusion('Training complete.')"
