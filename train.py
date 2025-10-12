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
from quant_utils import is_quantized_model, cleanup_memory

config = Config()


def setup_lora_for_quantized_model(model):
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training

        if hasattr(model, "peft_config"):
            print("Model already has LoRA adapters, skipping setup")
            return model

        print("Setting up LoRA adapters for quantized model...")

        if not hasattr(model, "is_loaded_in_4bit") or not model.is_loaded_in_4bit:
            print("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        lora_config = config.get_lora_config()
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        return model

    except ImportError as e:
        raise ImportError(f"peft is required for LoRA training: {e}")
    except Exception as e:
        print(f"Error setting up LoRA: {e}")
        raise


def train_model(args, model, device, tokenizer, dataset):
    if config.USE_QUANTIZATION and is_quantized_model(model):
        print("Detected quantized model, applying LoRA...")
        try:
            model = setup_lora_for_quantized_model(model)
        except Exception as e:
            print(f"Failed to setup LoRA for quantized model: {e}")
            raise
    else:
        print("Using full model training (no quantization/LoRA)...")
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
            except Exception as e:
                print(f"Warning: Could not enable gradient checkpointing: {e}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    print("Creating trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[get_timer_callback()],
        )
    except Exception as e:
        print(f"Failed to create trainer: {e}")
        raise

    if trainer.is_world_process_zero():
        print("Running model sanity check...")
        try:
            test_device = "cuda" if torch.cuda.is_available() else "cpu"

            if not config.USE_QUANTIZATION:
                model = model.to(test_device)

            test_input = tokenizer(
                ["I love pizza, but"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(test_device)

            with torch.no_grad():
                test_output = model(**test_input)

            print(f"Model test passed")
            print(f"Output shape: {test_output.logits.shape}")
            print(f"Output dtype: {test_output.logits.dtype}")

            del test_input, test_output
            cleanup_memory()

        except Exception as e:
            print(f"Warning: Model sanity check failed: {e}")
            print("Continuing anyway...")

    cleanup_memory()

    print("Starting training...")
    try:
        train_result = trainer.train()
        print(f"Training completed. Loss: {train_result.training_loss:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    if trainer.is_world_process_zero():
        repo_id = (
            config.OUTPUT_REPO + "-it"
            if config.INSTRUCT_FINETUNE_BOOL
            else config.OUTPUT_REPO
        )
        msg = f"Training loss: {train_result.training_loss:.4f}"

        print("Preparing model for upload...")
        try:
            if config.USE_QUANTIZATION and hasattr(model, "peft_config"):
                print("Merging LoRA adapters back into base model...")
                try:
                    model = model.merge_and_unload()
                    print("LoRA merge successful")
                except Exception as merge_error:
                    print(f"Warning: LoRA merge failed: {merge_error}")
                    print("Will attempt to save model with adapters...")

            print("Uploading model to hub...")
            upload_model(trainer, repo_id, msg)
            print("Model uploaded successfully")

        except Exception as e:
            print(f"Failed to upload model: {e}")
            import traceback

            traceback.print_exc()
            raise

        print("Training and upload complete.")
    else:
        rank = os.environ.get("LOCAL_RANK", -1)
        print(f"Worker {rank}: Training complete, skipping upload.")


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Quantization: {config.USE_QUANTIZATION}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max length: {config.MAX_LENGTH}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 50)

    print("Initializing distributed training...")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        timeout = timedelta(seconds=7200)

        try:
            dist.init_process_group(backend="nccl", timeout=timeout, init_method=None)
            device = torch.device(f"cuda:{local_rank}")
            print(f"Initialized distributed training on rank {local_rank}")
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            raise
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {device}")

    print(f"Using device: {device}")

    print("\nLoading prepared data...")
    try:
        print("Loading dataset...")
        dataset = load_from_disk("prepared_dataset")
        print(f"Dataset loaded: {len(dataset)} examples")

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")
        print(f"Tokenizer loaded: {len(tokenizer)} tokens")

        print("  Loading model...")
        model_kwargs = {}

        if not config.USE_QUANTIZATION:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if config.FP16:
                model_kwargs["torch_dtype"] = torch.float16
            print("  Loading model with flash attention 2 and fp16...")
        else:
            print("  Loading model with 4-bit quantization...")
            model_kwargs["quantization_config"] = config.get_quantization_config()
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained("prepared_model", **model_kwargs)

        if config.USE_QUANTIZATION:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        print(f"Model loaded")

        model_size = sum(p.numel() for p in model.parameters())
        trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {model_size:,}")
        print(f"  Trainable parameters: {trainable_size:,}")

    except Exception as e:
        print(f"Failed to load prepared data: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\nInitializing training arguments...")
    try:
        args = config.getConfig()
        print("Training arguments initialized")
    except Exception as e:
        print(f"Failed to initialize arguments: {e}")
        raise

    print("\nStarting model training...")
    try:
        train_model(args, model, device, tokenizer, dataset)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"CRITICAL ERROR")
        print(f"{'='*50}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*50}")

        import traceback

        traceback.print_exc()

        try:
            from util import Space

            Space().stop(e)
        except Exception as space_error:
            print(f"Failed to stop space: {space_error}")

        raise
