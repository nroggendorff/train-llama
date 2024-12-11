import train
import prep

import os

from util import Space
from doubt import discord_logging

@discord_logging(app_name='trainer', webhook_url=os.getenv('DISCORD_WEBHOOK_URL'))
def trainer():
    try:
        prep.main()
        train.main()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        Space().stop()

if __name__ == "__main__":
    trainer()
