# syntax=docker/dockerfile:experimental
FROM bitnami/deepspeed:latest

USER root
RUN useradd -m -u 1000 user

ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME="/home/user/.cache/huggingface"
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt update && apt install -y jq

COPY --chown=user . /app
RUN chown -R user:user /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U --no-cache-dir -r requirements.txt

RUN touch __init__.py
RUN [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

# USER user

# ENV HF_TOKEN={$HF_TOKEN}

# RUN python -c "print('Caching Data..'); \
#     import json; \
#     from datasets import load_dataset; \
#     config = json.load(open('config.json')); \
#     load_dataset(config['instruct-dataset'], split='train'); \
#     load_dataset(config['input-dataset'], split='train'); \
#     print('Cached Data.')"

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py"]
