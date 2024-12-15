FROM bitnami/deepspeed

USER root
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

ENV PYTHONUNBUFFERED=1

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /opt/bitnami/deepspeed && \
    chown -R user:user /opt/bitnami/deepspeed

COPY --chown=user . .

USER user

CMD ["/bin/bash", "-c", "python prep.py && deepspeed --num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())') train.py"]
