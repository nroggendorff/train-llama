import prep
import torch
import os
from util import Space
from doubt import discord_logging
import deepspeed.launcher.launch as launch
import sys

devices = torch.cuda.device_count()

@discord_logging(app_name='trainer', webhook_url=os.getenv('DISCORD_WEBHOOK_URL'))
def trainer():
    prep.main()

    sys.argv = [
        "--num_gpus", str(devices),
        "train.py"
    ]

    launch.main()

if __name__ == "__main__":
    try:
        trainer()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        Space().stop()
