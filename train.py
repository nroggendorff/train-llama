import os
import json
import torch
import warnings
from datetime import timedelta

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

from trainer import Trainer
from config import Config
from util import *

config = Config()


def print_gpu_info():
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )
        print()


def train_model(args, model, tokenizer, dataset):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    timer_callback = get_timer_callback()

    state_repo_id, state_repo_is_custom = get_state_repo_id()

    resume_step = 0
    if config.IS_SPACE:
        try:
            ensure_state_repo(state_repo_id, state_repo_is_custom)
        except Exception as e:
            print(f"Training state repo check failed: {e}")
            raise
        if config.RESUME:
            prior_state = download_training_state(state_repo_id, repo_type="dataset")
            if prior_state:
                resume_step = prior_state.get("global_step", 0)
                print(
                    f"Resuming from global step {resume_step} (state repo: {state_repo_id})"
                )
    elif os.path.exists("training_state.json"):
        resume_step = json.loads(open("training_state.json").read()).get(
            "global_step", 0
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        callbacks=[timer_callback] if timer_callback else [],
        resume_step=resume_step,
    )

    train_output = trainer.train()
    state = {"global_step": trainer.global_step, "completed": train_output.completed}

    if trainer.is_world_process_zero():
        if config.IS_SPACE:
            repo_id = (
                config.OUTPUT_REPO + f"-{config.INST_SUFFIX}"
                if config.INSTRUCT_FINETUNE_BOOL
                else config.OUTPUT_REPO
            )
            msg = f"Training loss: {train_output.training_loss:.4f} (step {trainer.global_step})"

            print("Pushing model to hub...")
            try:
                upload_model(trainer, repo_id, msg)
                print("Model and tokenizer uploaded successfully")
            except Exception as e:
                print(f"Failed to push model to hub: {e}")
                raise

            print(f"Pushing training state to {state_repo_id}...")
            try:
                ensure_state_repo(state_repo_id, state_repo_is_custom)
                push_training_state(state_repo_id, state)
            except Exception as e:
                print(f"Failed to push training state: {e}")
                raise

            if train_output.completed:
                print("Training fully completed, advancing pipeline..")
                try:
                    Space().reset(inst=config.INSTRUCT_FINETUNE_BOOL)
                except Exception as e:
                    print(f"Failed to advance pipeline: {e}")

                if config.INSTRUCT_FINETUNE_BOOL and not state_repo_is_custom:
                    print(
                        f"Entire training pipeline complete, deleting training state repo {state_repo_id}.."
                    )
                    try:
                        delete_state_repo(state_repo_id)
                    except Exception as e:
                        print(f"Failed to delete training state repo: {e}")
            elif not config.RESUME:
                print("Training paused before completion, marking for resume..")
                try:
                    Space().mark_resume()
                except Exception as e:
                    print(f"Failed to mark space for resume: {e}")
            else:
                print("Training paused before completion; will continue on next run.")
        else:
            save_model_to_disk(trainer, state=state)

        print("Trained Model.")
    else:
        print(
            f"Not the main process, skipping model saving. Trained model on device {os.environ.get('LOCAL_RANK', -1)}."
        )


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank == 0:
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="torch.utils.checkpoint"
        )
    else:
        warnings.filterwarnings("ignore")

    print("Initializing distributed training..")

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        timeout = timedelta(seconds=7200)
        dist.init_process_group(backend="nccl", timeout=timeout, init_method=None)
    else:
        raise RuntimeError(
            "LOCAL_RANK not set. This script requires DeepSpeed/distributed training."
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(
        f"Initialized distributed training: rank={local_rank}, world_size={world_size}"
    )

    if local_rank == 0:
        print_gpu_info()

    print("Loading Prepared Data..")
    try:
        print("Loading dataset from disk...")
        dataset = load_from_disk("prepared_dataset")

        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")

        if local_rank == 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"Tokenizer vocab size: {len(tokenizer)}")

        model_kwargs = {"use_cache": False}

        if config.BF16:
            model_kwargs["dtype"] = torch.bfloat16
        elif config.FP16:
            model_kwargs["dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(
                "prepared_model",
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as e:
            if local_rank == 0:
                print(
                    f"Failed to load with Flash Attention 2, falling back to default: {e}"
                )
            model = AutoModelForCausalLM.from_pretrained(
                "prepared_model", **model_kwargs
            )

        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

        print("Loaded Prepared Data.")
    except Exception as e:
        print(f"Failed to load dataset or tokenizer: {e}")
        raise

    print("Initializing Training Arguments..")
    args = config.getConfig()
    print("Initialized Arguments.")

    print("Starting Training..")
    train_model(args, model, tokenizer, dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in main: {e}")
        import traceback

        traceback.print_exc()
        if config.IS_SPACE:
            try:
                from util import Space

                Space().stop(e)
            except Exception:
                pass
        raise
