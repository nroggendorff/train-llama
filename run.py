from huggingface_hub import HfApi
from random import randint
import tempfile

api = HfApi()
repo_id = "nroggendorff/train-llama"

file_content = "FROM nroggendorff/train-llama:latest \n" \
"RUN jq '.init = 0 | .\"instruct-finetune-bool\" = false' config.json > temp.json && \\ \n" \
"    mv temp.json config.json && \\ \n" \
"    chown -R user:user config.json \n" \
"CMD [\"bash\", \"-c\", \"python prep.py && deepspeed --num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) train.py\"]"
print(file_content)
comment = "Merge when ready " + ''.join([chr(u) for u in [randint(33, 126) for _ in range(8)]])

with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile') as tmp_file:
    tmp_file.write(file_content)
    tmp_file.flush()
    
    pr_id = api.upload_file(
        path_or_fileobj=tmp_file.name,
        path_in_repo="Dockerfile",
        repo_id=repo_id,
        repo_type="space",
        create_pr=True,
        commit_message=comment
    ).pr_num

try:
    lastpr = pr_id - 1

    for _ in range(5):
        item_deets = api.get_discussion_details(repo_id, lastpr, repo_type='space')
        if item_deets.title == comment and item_deets.status == "open":
            api.change_discussion_status(repo_id, lastpr, 'closed', repo_type='space')
            break
        else:
            lastpr -= 1
except TypeError:
    pass
