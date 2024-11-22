# syntax=docker/dockerfile:experimental

FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

RUN sudo apt update && sudo apt install jq

WORKDIR /app
COPY --chown=user ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -U --no-cache-dir -r requirements.txt

COPY --chown=user . /app

RUN touch __init__.py
RUN [ -f configlib ] && mv configlib config.py || true && \
    [ -f util ] && mv util util.py || true && \
    [ -f config ] && mv config config.json || true

ENV HF_TOKEN $(cat /run/secrets/HF_TOKEN)

RUN printf "from datasets import load_dataset\nload_dataset('$(jq '.input-dataset' config.json)', split='train')\nload_dataset('$(jq '.instruct-dataset' config.json)', split='train')" | python

RUN python -u prep.py

CMD ["python", "train.py"]