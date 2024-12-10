# syntax=docker/dockerfile:experimental
FROM bitnami/deepspeed:latest

USER root
RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user . /app
RUN pip install --no-cache-dir -r requirements.txt

RUN touch __init__.py && \
    [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py"]
