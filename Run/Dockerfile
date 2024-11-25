FROM nroggendorff/train-llama:latest

USER root

ARG CONFIG="config.json"

RUN tmp=$(mktemp) && \
    jq '.init = 1' "${CONFIG}" > "$tmp" && \
    mv "$tmp" "${CONFIG}"

RUN python -u prep.py

CMD ["python", "train.py"]