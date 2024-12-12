import prep
import torch
import os
from util import Space
from doubt import discord_logging
import deepspeed.launcher.launch as launch

devices = torch.cuda.device_count()

@discord_logging(app_name='trainer', webhook_url=os.getenv('DISCORD_WEBHOOK_URL'))
def trainer():
    prep.main()
    args = f"train.py --num_gpus={devices}".split()
    launch.main(args)

if __name__ == "__main__":
    try:
        trainer()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        Space().stop()
