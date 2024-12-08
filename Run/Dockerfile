FROM nroggendorff/train-llama:latest

ARG CONFIG="config.json"
RUN tmp=$(mktemp) && \
    jq '.init = 0 | ."instruct-finetune-bool" = false' "${CONFIG}" > "$tmp" && \
    mv "$tmp" "${CONFIG}"
RUN chown -R user:user config.json

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py"]
