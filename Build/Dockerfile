FROM bitnami/deepspeed

USER root
RUN useradd -m -u 1000 user

ARG APP=/home/user/app
WORKDIR ${APP}

COPY requirements.txt ${APP}
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN install -d -o user -g user \
    /home/user/app/prepared_dataset/data \
    /home/user/app/prepared_tokenizer \
    /home/user/app/prepared_model

USER user
CMD ["/bin/bash", "-c", "python prep.py && deepspeed --num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())') train.py"]
