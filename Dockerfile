FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG APP=/home/user/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev build-essential \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
RUN mkdir -p ${APP} /.cache && chown -R user:user ${APP} /.cache/

WORKDIR ${APP}
USER user

ENV PATH="${APP}/venv/bin:$PATH"

RUN python3 -m venv venv

RUN venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    venv/bin/pip install --no-cache-dir --prefer-binary packaging ninja

RUN venv/bin/pip install --no-cache-dir --prefer-binary numpy==1.26.4

RUN venv/bin/pip install --no-cache-dir torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN venv/bin/pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.5.9+cu118torch2.4-cp310-cp310-linux_x86_64.whl"

RUN venv/bin/pip install --no-cache-dir --prefer-binary \
    transformers==4.46.3 \
    datasets==3.1.0 \
    tokenizers==0.20.3 \
    accelerate==1.2.1 \
    trl==0.12.1 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    liger-kernel==0.4.0 \
    deepspeed==0.15.4

RUN find venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find venv -type f -name "*.pyc" -delete && \
    find venv -type f -name "*.pyo" -delete && \
    find venv -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    touch __init__.py

COPY --chown=user:user . .

RUN mkdir -p \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model

ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    FLASH_ATTENTION=1 \
    ACCELERATE_USE_DEEPSPEED=true \
    NCCL_BLOCKING_WAIT=1 \
    NCCL_TIMEOUT=1200 \
    NCCL_SOCKET_IFNAME=^docker0,lo \
    NCCL_IB_DISABLE=0 \
    NCCL_NET_GDR_LEVEL=0 \
    CUDA_LAUNCH_BLOCKING=0

CMD ["bash", "./trainer.sh"]
