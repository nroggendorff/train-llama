FROM nroggendorff/train-llama:latest

USER root
RUN apt update && apt install -y jq

ARG CONFIG="config.json"

RUN tmp=$(mktemp) \
jq '.INIT = "abcde"' "${CONFIG}" > "$tmp" && mv "$tmp" "${CONFIG}"
RUN python -u prep.py

CMD ["python", "train.py"]