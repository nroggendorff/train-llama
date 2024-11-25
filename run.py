from huggingface_hub import HfApi
from random import randint

api = HfApi()
repo_id = "nroggendorff/train-llama"

comment = "Merge when ready " + ''.join([chr(u) for u in [randint(33, 126) for _ in range(8)]])

pr_id = api.upload_folder(
    folder_path="./Run",
    repo_id=repo_id,
    repo_type="space",
    create_pr=True,
    commit_message=comment
).pr_num - 1

isnt_v = api.get_discussion_details(repo_id, pr_id, repo_type='space').title == comment

api.change_discussion_status(repo_id, pr_id, 'closed', repo_type='space')