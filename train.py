import os
import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
import torch.distributed as dist

from config import Config
from util import *

config = Config()

def train_model(args, model, device, tokenizer, dataset, push):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset
    )

    if trainer.is_world_process_zero():
        try:
            model = model.to(device)
            test_input = tokenizer(
                ["I love pizza, but"], 
                return_tensors="pt"
            ).to(device)
            test_output = model(**test_input)
            print("Model test output shape:", test_output.logits.shape)

            del test_input, test_output
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error processing test batch: {e}")

    train = trainer.train()

    if trainer.is_world_process_zero():
        try:
            if push:
                repo_id = config.OUTPUT_REPO + "-it" if config.INSTRUCT_FINETUNE_BOOL else config.OUTPUT_REPO
                msg = f"Training loss: {train.training_loss:.4f}"
                trainer.model.push_to_hub(repo_id, commit_message=msg, force=True)
                trainer.tokenizer.push_to_hub(repo_id, commit_message=msg, force=True)
            else:
                trainer.model.save_pretrained("trained_model")
                trainer.tokenizer.save_pretrained("trained_tokenizer")
            raise Conclusion("Trained Model.")
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise
    else:
        print(f"Not the main process, skipping model saving. Trained model on device {os.environ.get('LOCAL_RANK', -1)}.")

def main(push_to_hub=config.PUSH_TO_HUB):
    print("Initializing accelerator..")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}, rank: {local_rank}")

    print("Loading Prepared Data..")
    try:
        dataset = load_from_disk("prepared_dataset")
        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")
        model = AutoModelForCausalLM.from_pretrained("prepared_model")
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
        Space().stop(e)
