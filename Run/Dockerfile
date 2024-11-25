FROM nroggendorff/train-llama:latest

ARG CONFIG="config.json"
RUN tmp=$(mktemp) && \
    jq '.init = 1 | ."instruct-finetune-bool" = false' "${CONFIG}" > "$tmp" && \
    mv "$tmp" "${CONFIG}"
RUN chown -R user:user config.json

CMD ["bash", "-c", "python prep.py && python train.py"]