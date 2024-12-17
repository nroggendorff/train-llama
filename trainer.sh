#!/bin/bash
DEVICE_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')

sed -i 's/init": 0/init": $INIT/g' config.json

python prep.py
deepspeed --num_gpus=$DEVICE_COUNT train.py
