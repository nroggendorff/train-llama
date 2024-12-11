FROM bitnami/deepspeed:latest

USER root
RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=1000:1000 . /app

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py"]
