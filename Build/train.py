import os
import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTTrainer

from config import Config
from util import *

config = Config()

def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0 else config.OUTPUT_REPO
        model = LlamaForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise

    if dist.is_initialized():
        dist.barrier()

    return model

def create_model(tokenizer):
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.FACTOR,
        intermediate_size=config.FACTOR * 4,
        num_hidden_layers=config.FACTOR // 2 ** 5,
        num_attention_heads=config.FACTOR // 2 ** 4,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False
    )

    try:
        model = LlamaForCausalLM(model_config)
    except Exception as e:
        print(f"Failed to create model: {e}")
        raise

    if dist.is_initialized():
        dist.barrier()
        rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        if rank == 0:
            first_param = next(model.parameters())
            print(f"First param sum before sync: {first_param.sum().item()}")

        try:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        except Exception as e:
            print(f"Failed to broadcast parameters: {e}")
            dist.destroy_process_group()
            raise

        dist.barrier()

        if rank == 0:
            first_param = next(model.parameters())
            print(f"First param sum after sync: {first_param.sum().item()}")

    return model

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

def main(push_to_hub=config.PUSH_TO_HUB, is_inst=config.INSTRUCT_FINETUNE_BOOL):
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
        dataset = load_from_disk("prepared_dataset")
        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")
        print("Loaded Prepared Data.")
    except Exception as e:
        print(f"Failed to load dataset or tokenizer: {e}")
        raise

    print("Initializing Arguments..")
    args = config.getConfig()
    print("Initialized Arguments.")

    print("Getting Model..")
    try:
        model = load_model(tokenizer) if is_inst or config.INIT > 0 else create_model(tokenizer)
        print("Got Model.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise

    print("Training Model..")
    train_model(args, model, device, tokenizer, dataset, push_to_hub)
    raise Conclusion("Trained Model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        Space().stop(e)
