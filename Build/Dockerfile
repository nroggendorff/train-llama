FROM bitnami/deepspeed

WORKDIR /app

COPY requirements.txt /app
USER root

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 . /app

RUN mkdir -p /opt/bitnami/deepspeed && \
    useradd -m user && \
    chmod -R 777 /opt/bitnami/deepspeed /app

CMD ["deepspeed", "--num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')", "trainer.py"]
