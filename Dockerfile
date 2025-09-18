FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG APP=/home/user/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev build-essential \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

RUN mkdir -p ${APP} && chown -R user:user ${APP}
RUN mkdir -p /.cache && chown -R user:user /.cache/

WORKDIR ${APP}

COPY --chown=user:user installer.sh .

USER user

ENV PATH="${APP}/venv/bin:$PATH"

RUN python3 -m venv venv
RUN bash installer.sh

RUN touch __init__.py

COPY --chown=user:user . .

RUN mkdir -p \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model

CMD ["bash", "./trainer.sh"]
