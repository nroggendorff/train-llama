import os
import tempfile
import time
import json
import signal
import sys
from huggingface_hub import HfApi
from transformers import TrainerCallback

from config import Config


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

        total_files = sum(len(files) for _, _, files in os.walk(temp_dir))
        print(f"Uploading {total_files} files...")

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


class TrainingTimer:
    def __init__(self, timeout_minutes=config.TIMEOUT):
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()

    def is_expired(self):
        return time.time() - self.start_time >= self.timeout_seconds

    def remaining_minutes(self):
        remaining = self.timeout_seconds - (time.time() - self.start_time)
        return max(0, remaining / 60)


class TimerCallback(TrainerCallback):
    def __init__(self):
        self.timer = TrainingTimer()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and self.timer.is_expired():
            print(f"Timer expired at step {state.global_step}, stopping training")

            trainer = kwargs.get("trainer")
            if trainer and trainer.is_world_process_zero():
                repo_id = (
                    config.OUTPUT_REPO + "-it"
                    if config.INSTRUCT_FINETUNE_BOOL
                    else config.OUTPUT_REPO
                )

                try:
                    upload_model(
                        trainer, repo_id, f"Timer stop at step {state.global_step}"
                    )
                    print("Model saved successfully")
                except Exception as e:
                    print(f"Failed to save model: {e}")

            control.should_training_stop = True
        return control


def handle_timer_signal(signum, frame):
    print(f"Received signal {signum}, timer forcing shutdown")
    sys.exit(0)


def setup_timer_signals():
    signal.signal(signal.SIGTERM, handle_timer_signal)
    signal.signal(signal.SIGINT, handle_timer_signal)


def get_timer_callback():
    setup_timer_signals()
    return TimerCallback()
