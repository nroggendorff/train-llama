import os
import tempfile
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


def upload_model(trainer, repo_id, commit_message):
    api = HfApi()

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Saving model and tokenizer to temporary directory...")

        trainer.model.save_pretrained(temp_dir)
        trainer.processing_class.save_pretrained(temp_dir)

        all_files = []

        for root, _, files in os.walk(model_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, model_path)
                all_files.append((full_path, rel_path))

        for root, _, files in os.walk(tokenizer_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, tokenizer_path)
                all_files.append((full_path, rel_path))

        print(f"Uploading {len(all_files)} files...")

        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            delete_patterns=["*"],
            ignore_patterns=[".git/**"],
        )

        print("Upload completed successfully")


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
            key="INST",
            value="true" if new_instruct else "false",
        )
