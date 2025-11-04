import os
import torch
from datetime import timedelta

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch.distributed as dist

from trainer import Trainer
from config import Config
from util import *

config = Config()


def train_model(args, model, device, tokenizer, dataset):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[get_timer_callback()],
    )

    if trainer.is_world_process_zero():
        try:
            model = model.to(device)

            test_input = tokenizer(["I love pizza, but"], return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                test_output = model(**test_input)
            print("Model test output shape:", test_output.logits.shape)
            print("Model test output dtype:", test_output.logits.dtype)

            del test_input, test_output
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error processing test batch: {e}")

    torch.cuda.empty_cache()

    train = trainer.train()

    if trainer.is_world_process_zero():
        repo_id = (
            config.OUTPUT_REPO + f"-{config.INST_SUFFIX}"
            if config.INSTRUCT_FINETUNE_BOOL
            else config.OUTPUT_REPO
        )
        msg = f"Training loss: {train.training_loss:.4f}"

        print("Pushing model to hub...")
        try:
            upload_model(trainer, repo_id, msg)
            print("Model and tokenizer uploaded successfully")
        except Exception as e:
            print(f"Failed to push model to hub: {e}")
            raise

        print("Trained Model.")
    else:
        print(
            f"Not the main process, skipping model saving. Trained model on device {os.environ.get('LOCAL_RANK', -1)}."
        )


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Initializing accelerator..")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        timeout = timedelta(seconds=7200)
        dist.init_process_group(backend="nccl", timeout=timeout, init_method=None)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")

    print(f"Using device: {device}, rank: {local_rank}")

    print("Loading Prepared Data..")
    try:
        print("Loading dataset from disk...")
        dataset = load_from_disk("prepared_dataset")

        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")

        print(f"Dataset size: {len(dataset)}")

        model_kwargs = {"attn_implementation": "flash_attention_2"}

        if config.FP16:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained("prepared_model", **model_kwargs)
        print("Loaded Prepared Data.")
    except Exception as e:
        print(f"Failed to load dataset or tokenizer: {e}")
        raise

    print("Initializing Arguments..")
    args = config.getConfig()
    print("Initialized Arguments.")

    print("Training Model..")
    train_model(args, model, device, tokenizer, dataset)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in main: {e}")
        import traceback

        traceback.print_exc()
        try:
            from util import Space

            Space().stop(e)
        except Exception:
            pass
        raise
