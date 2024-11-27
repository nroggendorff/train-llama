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
    print("No diff detected.")