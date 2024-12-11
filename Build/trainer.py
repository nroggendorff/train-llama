import prep

import torch

import os
import subprocess

from util import Space
from doubt import discord_logging

devices = torch.cuda.device_count()

@discord_logging(app_name='trainer', webhook_url=os.getenv('DISCORD_WEBHOOK_URL'))
def trainer():
    prep.main()
    subprocess.run(["deepspeed", f"--num_gpus={devices}", "train.py"])

if __name__ == "__main__":
    try:
        trainer()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        Space().stop()
