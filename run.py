from huggingface_hub import HfApi

api = HfApi()
repo_id = "nroggendorff/train-llama"

try:
    api.restart_space(repo_id, factory_reboot=True)
except Exception as e:
    print(f"Error restarting space: {e}")
