FROM bitnami/deepspeed

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())') train.py"]
