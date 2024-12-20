FROM bitnami/deepspeed

USER root
RUN useradd -m -u 1000 user

ARG APP=/home/user/app
WORKDIR ${APP}

COPY requirements.txt ${APP}
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN install -d -o user -g user \
    ${APP}/prepared_dataset/data \
    ${APP}/prepared_tokenizer \
    ${APP}/prepared_model
RUN chmod -R 777 ${APP} && \
    chmod -R 777 /opt/bitnami

ENV INIT=0
ENV INSTRUCT=false

USER user
CMD ["bash", "trainer.sh"]
