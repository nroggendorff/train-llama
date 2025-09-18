import os
from huggingface_hub import HfApi

from config import Config

import json
import os


def get_dataset_size():
    with open(os.path.join("prepared_dataset", "dataset_info.json"), "r") as f:
        info = json.load(f)
    total = 0
    for split in info.get("splits", {}).values():
        total += split.get("num_examples", 0)
    return total


config = Config()


class Conclusion(Exception):
    def __init__(self, message="Script execution has completed."):
        super().__init__(message)


class Space:
    def __init__(self):
        self.api = HfApi()
        self.repo_id = (
            f"{os.getenv('SPACE_AUTHOR_NAME')}/{os.getenv('SPACE_REPO_NAME')}"
        )

    def stop(self, message=None):
        if message:
            raise Conclusion(f"{type(message).__name__}: {message}")
        raise Conclusion()

    def pause(self):
        self.api.pause_space(self.repo_id)

    def resume(self):
        self.api.restart_space(self.repo_id)

    def reset(
        self,
        init=config.INIT,
        inst=config.INSTRUCT_FINETUNE_BOOL,
        shard_size=config.SHARD_SIZE,
    ):
        dataset_size = get_dataset_size()

        if dataset_size > shard_size * (init + 2):
            if not inst:
                new_init = 0
                new_instruct = True
            else:
                return self.pause()
        else:
            new_init = init + 1
            new_instruct = inst

        self.api.add_space_variable(
            repo_id=self.repo_id,
            key="INIT",
            value=str(new_init),
        )
        self.api.add_space_variable(
            repo_id=self.repo_id,
            key="INSTRUCT",
            value="true" if new_instruct else "false",
        )
