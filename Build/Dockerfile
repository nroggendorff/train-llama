FROM bitnami/deepspeed

WORKDIR /app

COPY requirements.txt /app
USER root

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 . /app
RUN useradd -m -u 1000 user && chmod -R 777 /app

CMD ["/bin/bash", "-c", "python trainer.py"]
