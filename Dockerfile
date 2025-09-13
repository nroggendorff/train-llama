FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

USER root
RUN usermod -l user ubuntu && groupmod -n user ubuntu

ARG APP=/home/user/app
WORKDIR ${APP}

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN chown -R user:user ${APP}
USER user

RUN python3 -m venv .venv
ENV PATH="${APP}/.venv/bin:$PATH"

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN touch __init__.py

RUN mkdir -p \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model

ENV INIT=0
ENV INSTRUCT=false

CMD ["./trainer.sh"]
