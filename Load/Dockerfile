FROM nroggendorff/train-llama:latest

ARG CONFIG="config.json"
RUN tmp=$(mktemp) && \
    jq '.init = 0 | ."instruct-finetune-bool" = false' "${CONFIG}" > "$tmp" && \
    mv "$tmp" "${CONFIG}"
RUN chown -R user:user config.json

RUN python -c "print('Caching Data..'); \
    import json; \
    from datasets import load_dataset; \
    config = json.load(open('config.json')); \
    load_dataset(config['instruct-dataset'], split='train'); \
    load_dataset(config['input-dataset'], split='train'); \
    print('Cached Data.')"
