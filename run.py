from huggingface_hub import HfApi

api = HfApi()
repo_id = "nroggendorff/train-llama"

pr_id = api.upload_folder(
    folder_path="./Run",
    repo_id=repo_id,
    repo_type="space",
    create_pr=True,
    commit_message="Merge when ready"
).pr_revision

api.change_discussion_status(repo_id, pr_id, 'closed', repo_type='space')