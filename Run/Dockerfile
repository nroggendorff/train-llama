FROM nroggendorff/train-llama:latest

RUN jq '.init = 0 | ."instruct-finetune-bool" = false' config.json > temp.json && \
    mv temp.json config.json && \
    chown -R user:user config.json

CMD ["bash", "-c", "python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py"]