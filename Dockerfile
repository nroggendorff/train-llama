# syntax=docker/dockerfile:experimental
FROM python:3.9

COPY ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -U -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user . /app

RUN touch __init__.py
RUN [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

RUN python -c "print('Caching datasets..') \
import json; \
from datasets import load_dataset; \
config = json.load(open('config.json')); \
load_dataset(config['instruct-dataset'], split='train') if config['instruct-finetune-bool'] else load_dataset(config['input-dataset'], split='train')"

RUN python -u prep.py

CMD ["python", "train.py"]