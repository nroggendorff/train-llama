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

RUN venv/bin/pip install --no-cache-dir --prefer-binary packaging wheel setuptools ninja && \
    venv/bin/pip install --no-cache-dir --prefer-binary numpy==1.26.4 && \
    venv/bin/pip install --no-cache-dir torch==2.4 --index-url https://download.pytorch.org/whl/cu118 && \
    venv/bin/pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.5.9+cu118torch2.4-cp310-cp310-linux_x86_64.whl" && \
    venv/bin/pip install --no-cache-dir --prefer-binary trl==0.22.2 liger-kernel==0.6.2 deepspeed==0.17.5 && \
    find venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find venv -type f -name "*.pyc" -delete && \
    find venv -type f -name "*.pyo" -delete && \
    find venv -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true

RUN touch __init__.py

COPY --chown=user:user . .

RUN mkdir -p \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model

CMD ["bash", "./trainer.sh"]
