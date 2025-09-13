FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ARG APP=/home/user/app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev build-essential \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash user || true
RUN mkdir -p ${APP} && chown -R user:user ${APP}

WORKDIR ${APP}

COPY --chown=user:user . ${APP}

USER user

ENV PATH="${APP}/.venv/bin:$PATH"

RUN python3 -m venv .venv \
    && .venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

RUN touch __init__.py

RUN mkdir -p \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model
ENV INIT=0
ENV INSTRUCT=false

RUN chmod +x ./trainer.sh || true

CMD ["./trainer.sh"]
