FROM bitnami/deepspeed

WORKDIR /app

COPY requirements.txt /app
USER root

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 . /app
RUN useradd -m -u 1000 user && \
    chmod -R 755 /app && \
    mkdir -p /opt/bitnami/deepspeed && \
    chown -R user:user /opt/bitnami/deepspeed

CMD ["/bin/bash", "-c", "python prep.py && deepspeed --num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())') train.py"]
