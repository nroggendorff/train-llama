#!/usr/bin/env bash
set -euo pipefail

venv/bin/pip install --no-cache-dir packaging wheel setuptools ninja

venv/bin/pip install --no-cache-dir numpy==1.26.4
venv/bin/pip install --no-cache-dir torch==2.4 --index-url https://download.pytorch.org/whl/cu118

if ! venv/bin/pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.5.9+cu118torch2.4-cp310-cp310-linux_x86_64.whl"; then
    echo "Prebuilt wheel failed, installing from source..."
    venv/bin/pip install --no-cache-dir flash-attn==2.5.9
fi

venv/bin/pip install --no-cache-dir trl==0.22.2 liger-kernel==0.6.2 deepspeed==0.17.5
