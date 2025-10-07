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

        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            delete_patterns=["!*.md"],
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
        actual_samples=None,
        actual_epochs=None,
    ):
        dataset_size = get_dataset_size()

        if actual_samples is None:
            actual_samples = dataset_size
        if actual_epochs is None:
            actual_epochs = config.EPOCHS

        self._update_samples_through(actual_samples, actual_epochs)

        total_trained = sum(
            int(entry["samples"] * entry["epochs"]) for entry in config.SAMPLES_THROUGH
        ) + int(actual_samples * actual_epochs)

        if total_trained >= dataset_size:
            if not inst:
                new_init = 0
                new_instruct = True
                self._clear_samples_through()
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

    def _update_samples_through(self, samples, epochs):
        current_value = os.environ.get("SAMPLES_THROUGH", "")

        new_entry = f"{samples}x{epochs}"

        if current_value:
            updated_value = f"{current_value.rstrip(',')}, {new_entry},"
        else:
            updated_value = f"{new_entry},"

        self.api.add_space_variable(
            repo_id=self.repo_id,
            key="SAMPLES_THROUGH",
            value=updated_value,
        )

        print(f"Updated SAMPLES_THROUGH with: {new_entry}")

    def _clear_samples_through(self):
        self.api.add_space_variable(
            repo_id=self.repo_id,
            key="SAMPLES_THROUGH",
            value="",
        )
        print("Cleared SAMPLES_THROUGH for instruction fine-tuning phase")


class TrainingTimer:
    def __init__(self, timeout_minutes=config.TIMEOUT):
        self.timeout_seconds = timeout_minutes * 60
        if os.path.exists(".timer_start"):
            with open(".timer_start", "r") as f:
                self.start_time = float(f.read().strip())
        else:
            self.start_time = time.time()

    def is_expired(self):
        return time.time() - self.start_time >= self.timeout_seconds

    def remaining_minutes(self):
        remaining = self.timeout_seconds - (time.time() - self.start_time)
        return max(0, remaining / 60)


class TimerCallback(TrainerCallback):
    def __init__(self):
        self.timer = TrainingTimer()
        self.samples_trained = 0
        self.starting_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.starting_step = state.global_step
        return control

    def on_step_end(self, args, state, control, **kwargs):
        samples_per_step = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        self.samples_trained = (
            state.global_step - self.starting_step
        ) * samples_per_step

        if state.global_step % 10 == 0 and self.timer.is_expired():
            print(f"Timer expired at step {state.global_step}, stopping training")
            print(f"Trained on {self.samples_trained} samples")

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

    def get_training_stats(self):
        dataset_size = get_dataset_size()
        epochs_completed = (
            self.samples_trained / dataset_size if dataset_size > 0 else 0
        )
        return self.samples_trained, epochs_completed


def handle_timer_signal(signum, frame):
    print(f"Received signal {signum}, timer forcing shutdown")
    sys.exit(0)


def setup_timer_signals():
    signal.signal(signal.SIGTERM, handle_timer_signal)
    signal.signal(signal.SIGINT, handle_timer_signal)


def get_timer_callback():
    setup_timer_signals()
    return TimerCallback()
