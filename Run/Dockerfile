FROM nroggendorff/train-llama:latest

USER root

ARG CONFIG="config.json"

RUN tmp=$(mktemp) && \
    jq '.init = 1 | ."instruct-finetune-bool" = false' "${CONFIG}" > "$tmp" && \
    mv "$tmp" "${CONFIG}"
RUN chown -R user:user config.json

RUN python -u prep.py

CMD ["python", "train.py"]