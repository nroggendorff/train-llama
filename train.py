import os
import torch
from datetime import timedelta

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
import torch.distributed as dist

from config import Config
from util import *

config = Config()


def train_model(args, model, device, tokenizer, dataset, push):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
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
        try:
            if push:
                repo_id = (
                    config.OUTPUT_REPO + "-it"
                    if config.INSTRUCT_FINETUNE_BOOL
                    else config.OUTPUT_REPO
                )
                msg = f"Training loss: {train.training_loss:.4f}"

                print("Pushing model to hub...")
                trainer.model.push_to_hub(repo_id, commit_message=msg, force=True)
                trainer.processing_class.push_to_hub(repo_id, commit_message=msg, force=True)

                print("Model pushed to hub successfully")
            else:
                trainer.model.save_pretrained("trained_model")
                trainer.processing_class.save_pretrained("trained_tokenizer")

            print("Trained Model.")
        except Exception as e:
            print(f"Failed to save model: {e}")
    else:
        print(
            f"Not the main process, skipping model saving. Trained model on device {os.environ.get('LOCAL_RANK', -1)}."
        )


def main(push_to_hub=config.PUSH_TO_HUB):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Initializing accelerator..")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

        timeout = timedelta(seconds=7200)
        dist.init_process_group(
            backend="nccl",
            timeout=timeout,
            init_method=None
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")

    print(f"Using device: {device}, rank: {local_rank}")

    print("Loading Prepared Data..")
    try:
        print("Loading dataset from disk...")
        dataset = load_from_disk("prepared_dataset")

        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")

        model_kwargs = {"attn_implementation": "flash_attention_2"}

        if config.FP16:
            model_kwargs["dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained("prepared_model", **model_kwargs)
        print("Loaded Prepared Data.")
    except Exception as e:
        print(f"Failed to load dataset or tokenizer: {e}")
        raise

    print("Initializing Arguments..")
    args = config.getConfig()
    print("Initialized Arguments.")

    print("Training Model..")
    train_model(args, model, device, tokenizer, dataset, push_to_hub)


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
