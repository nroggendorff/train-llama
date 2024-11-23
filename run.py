from huggingface_hub import HfApi

HfApi().upload_folder(
    folder_path="./Run",
    repo_id="nroggendorff/train-llama",
    repo_type="space",
    create_pr=True,
    commit_message="Merge when ready" #grr
)