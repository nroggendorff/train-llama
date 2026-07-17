import os
import json
import hashlib
import tempfile
import time
import signal
import sys
from huggingface_hub import HfApi, hf_hub_download
from transformers import TrainerCallback

from config import Config


def _repo_exists(api, repo_id, repo_type):
    try:
        return api.repo_exists(repo_id=repo_id, repo_type=repo_type)
    except AttributeError:
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return True
        except Exception:
            return False


def retry_on_failure(func, *args, **kwargs):
    config = Config()
    max_retries = config.MAX_RETRIES

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                raise


def download_training_state(repo_id, repo_type="dataset"):
    try:
        return json.loads(
            open(
                hf_hub_download(
                    repo_id=repo_id, filename="training_state.json", repo_type=repo_type
                )
            ).read()
        )
    except Exception as e:
        print(f"No resumable training state found for {repo_id}: {e}")
        return None


STATE_OWNER_KEY = "_owner_repo"


def get_state_repo_id():
    custom = os.environ.get("TEMP_DATA_REPO")
    if custom:
        return custom, True

    namespace = config.base_output_repo.split("/")[0]
    digest = hashlib.sha256(config.base_output_repo.encode()).hexdigest()
    return f"{namespace}/{digest}", False


def ensure_state_repo(repo_id, is_custom):
    api = HfApi()

    if is_custom:
        retry_on_failure(
            api.create_repo,
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        return

    if not _repo_exists(api, repo_id, "dataset"):
        retry_on_failure(
            api.create_repo, repo_id=repo_id, repo_type="dataset", private=True
        )
        return

    existing = download_training_state(repo_id, repo_type="dataset")
    if not existing or existing.get(STATE_OWNER_KEY) != config.base_output_repo:
        raise RuntimeError(
            f"The dataset repo '{repo_id}' already exists and does not appear to belong to "
            f"this training run (expected owner '{config.base_output_repo}'). This should be "
            "essentially impossible given the hash-based naming, but to be safe: please move "
            f"whatever is currently at https://huggingface.co/datasets/{repo_id} elsewhere, or "
            "set the TEMP_DATA_REPO environment variable to use a different repo for training "
            "state (a custom TEMP_DATA_REPO is never auto-deleted)."
        )


def push_training_state(repo_id, state):
    api = HfApi()
    state = {**state, STATE_OWNER_KEY: config.base_output_repo}

    def do_upload():
        api.upload_file(
            path_or_fileobj=json.dumps(state).encode(),
            path_in_repo="training_state.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update training state (step {state.get('global_step', 0)})",
        )

    retry_on_failure(do_upload)


def delete_state_repo(repo_id):
    api = HfApi()
    retry_on_failure(api.delete_repo, repo_id=repo_id, repo_type="dataset")


def save_model_to_disk(trainer, output_dir="output", state=None):
    print(f"Saving model and tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(trainer, "model_engine") and trainer.model_engine is not None:
        trainer.model_engine.module.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)

    trainer.processing_class.save_pretrained(output_dir)

    if state is not None:
        open(os.path.join(output_dir, "training_state.json"), "w").write(
            json.dumps(state)
        )

    print(f"Model and tokenizer saved to {output_dir}")


def upload_model(trainer, repo_id, commit_message, extra_files=None):
    api = HfApi()

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Saving model and tokenizer to temporary directory...")

        if hasattr(trainer, "model_engine") and trainer.model_engine is not None:
            trainer.model_engine.module.save_pretrained(temp_dir)
        else:
            trainer.model.save_pretrained(temp_dir)

        trainer.processing_class.save_pretrained(temp_dir)

        for filename, content in (extra_files or {}).items():
            open(os.path.join(temp_dir, filename), "w").write(content)

        total_files = sum(len(files) for _, _, files in os.walk(temp_dir))
        print(f"Uploading {total_files} files...")

        def to_delete():
            delete_patterns = set()
            for file_info in api.list_repo_files(repo_id=repo_id, repo_type="model"):
                file_extension = os.path.splitext(file_info)[1]
                if file_extension and file_extension != ".md":
                    delete_patterns.add(f"**/*{file_extension}")
            return list(delete_patterns)

        def do_upload():
            api.create_repo(repo_id=repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                delete_patterns=to_delete(),
                ignore_patterns=[".git/**"],
            )

        retry_on_failure(do_upload)
        print("Upload completed successfully")


def check_tokenizer_has_instruct_config(tokenizer):
    config = Config()

    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    instruct_tokens = config.SPECIAL_TOKENS.get("additional_special_tokens", [])
    has_instruct_tokens = (
        all(token in tokenizer.get_vocab() for token in instruct_tokens)
        if instruct_tokens
        else False
    )

    if has_chat_template and has_instruct_tokens:
        print(
            "Tokenizer already has instruction configuration (chat template and instruction tokens)"
        )
        return True
    elif has_chat_template:
        print("Tokenizer has chat template but missing instruction tokens")
        return False
    elif has_instruct_tokens:
        print("Tokenizer has instruction tokens but missing chat template")
        return False
    else:
        print("Tokenizer does not have instruction configuration")
        return False


config = Config()


class Conclusion(Exception):
    def __init__(self, e=RuntimeError(), message="Script execution has completed."):
        self.recoverable_errors = [
            "timeout",
            "max",
            "requests",
            "reset",
            "refused",
            "broken",
        ]

        err_str = str(e).lower()

        if config.IS_SPACE and any(word in err_str for word in self.recoverable_errors):
            try:
                Space().resume()
            except Exception as resume_err:
                print(f"Failed to resume after recoverable error: {resume_err}")

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
        retry_on_failure(self.api.pause_space, self.repo_id)

    def resume(self):
        retry_on_failure(self.api.restart_space, self.repo_id)

    def reset(self, inst=config.INSTRUCT_FINETUNE_BOOL):
        if inst:
            return self.pause()

        def update_variables():
            self.api.add_space_variable(repo_id=self.repo_id, key="INST", value="true")
            self.api.add_space_variable(
                repo_id=self.repo_id, key="RESUME", value="false"
            )

        retry_on_failure(update_variables)

    def mark_resume(self):
        def update_variable():
            self.api.add_space_variable(
                repo_id=self.repo_id, key="RESUME", value="true"
            )

        retry_on_failure(update_variable)


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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and self.timer.is_expired():
            print(f"Timer expired at step {state.global_step}, stopping training")
            control.should_training_stop = True
        return control


def handle_timer_signal(signum, frame):
    print(f"Received signal {signum}, timer forcing shutdown")
    sys.exit(0)


def setup_timer_signals():
    signal.signal(signal.SIGTERM, handle_timer_signal)
    signal.signal(signal.SIGINT, handle_timer_signal)


def get_timer_callback():
    if not config.IS_SPACE:
        return None
    setup_timer_signals()
    return TimerCallback()
