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
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise

def main(push_to_hub=config.PUSH_TO_HUB):
    print("Initializing accelerator..")

    if torch.cuda.is_available():
        try:
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
                torch.cuda.set_device(local_rank)
                try:
                    dist.init_process_group(backend="nccl")
                except Exception as e:
                    print(f"Failed to initialize distributed training: {e}")
                    raise
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device("cuda")

            torch.cuda.synchronize(device)
        except RuntimeError as e:
            print(f"CUDA initialization failed: {e}")
            device = torch.device("cpu")
            print("Falling back to CPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    print(f"Using device: {device}, rank: {os.environ.get('LOCAL_RANK', 0)}")

    print("Loading Prepared Data..")
    try:
        dataset = load_from_disk("/tmp/prepared_dataset")
        tokenizer = AutoTokenizer.from_pretrained("/tmp/prepared_tokenizer")
        model = AutoModelForCausalLM.from_pretrained("/tmp/prepared_model")
        print("Loaded Prepared Data.")
    except Exception as e:
        print(f"Failed to load dataset or tokenizer: {e}")
        raise

    print("Initializing Arguments..")
    args = config.getConfig()
    print("Initialized Arguments.")

    print("Training Model..")
    train_model(args, model, device, tokenizer, dataset, push_to_hub)
    raise Conclusion("Trained Model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        Space().stop(e)
