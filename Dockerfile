# syntax=docker/dockerfile:experimental
FROM bitnami/deepspeed:latest

USER root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
pip install -U --no-cache-dir -r requirements.txt

WORKDIR /app

COPY --chown=user . /app

RUN touch __init__.py
RUN [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
python util.py $(cat /run/secrets/HF_TOKEN)

RUN --mount=type=cache,target=/root/.cache/dataset python -c "print('Caching Data..'); \
import json; \
from datasets import load_dataset; \
config = json.load(open('config.json')); \
load_dataset(config['instruct-dataset'], split='train') if config['instruct-finetune-bool'] else load_dataset(config['input-dataset'], split='train'); \
print('Cached Data.')"

# RUN --mount=type=cache,target=/root/.cache/datasetv \
# python -u prep.py

CMD ["python", "prep.py &&", "python", "train.py"]