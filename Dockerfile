# syntax=docker/dockerfile:experimental
FROM bitnami/deepspeed:latest

USER root
RUN useradd -m -u 1000 user

ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME="/home/user/.cache/huggingface"

COPY --chown=user ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
pip install -U --no-cache-dir -r requirements.txt

WORKDIR /app

COPY --chown=user . /app
RUN chown -R user:user /app

RUN touch __init__.py
RUN [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

USER user

RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
python util.py $(cat /run/secrets/HF_TOKEN)

RUN python -c "print('Caching Data..'); \
import json; \
from datasets import load_dataset; \
config = json.load(open('config.json')); \
load_dataset(config['instruct-dataset'], split='train') if config['instruct-finetune-bool'] else load_dataset(config['input-dataset'], split='train'); \
print('Cached Data.')"

RUN python -u prep.py

ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "train.py"]