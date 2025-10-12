import sys
import torch


def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"{package_name or module_name}")
        return True
    except ImportError as e:
        print(f"{package_name or module_name}: {e}")
        return False


def check_cuda():
    print("\n=== CUDA Check ===")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        return True
    else:
        print(" CUDA not available")
        return False


def check_quantization_support():
    print("\n=== Quantization Support ===")
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes version: {bnb.__version__}")

        try:
            from transformers import BitsAndBytesConfig

            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
            )
            print("BitsAndBytesConfig created successfully")
            return True
        except Exception as e:
            print(f"Failed to create BitsAndBytesConfig: {e}")
            return False
    except ImportError as e:
        print(f"bitsandbytes import failed: {e}")
        return False


def check_peft_support():
    print("\n=== PEFT/LoRA Support ===")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("PEFT imports successful")

        try:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print("LoraConfig created successfully")
            return True
        except Exception as e:
            print(f"Failed to create LoraConfig: {e}")
            return False
    except ImportError as e:
        print(f"PEFT import failed: {e}")
        return False


def check_flash_attention():
    print("\n=== Flash Attention ===")
    try:
        import flash_attn

        print(f"flash_attn version: {flash_attn.__version__}")
        return True
    except ImportError as e:
        print(f"flash_attn import failed: {e}")
        return False


def check_deepspeed():
    print("\n=== DeepSpeed ===")
    try:
        import deepspeed

        print(f"deepspeed version: {deepspeed.__version__}")
        return True
    except ImportError as e:
        print(f"deepspeed import failed: {e}")
        return False


def check_config():
    print("\n=== Configuration ===")
    try:
        from config import Config

        config = Config()
        print(f"Config loaded")
        print(f"USE_QUANTIZATION: {config.USE_QUANTIZATION}")
        print(f"BATCH_SIZE: {config.BATCH_SIZE}")
        print(f"MAX_LENGTH: {config.MAX_LENGTH}")
        print(f"FP16: {config.FP16}")

        if config.USE_QUANTIZATION:
            try:
                quant_config = config.get_quantization_config()
                print(f"Quantization config created")
            except Exception as e:
                print(f" Failed to create quantization config: {e}")
                return False

            try:
                lora_config = config.get_lora_config()
                print(f"LoRA config created")
            except Exception as e:
                print(f"Failed to create LoRA config: {e}")
                return False

        return True
    except Exception as e:
        print(f"Config check failed: {e}")
        return False


def check_quant_utils():
    print("\n=== Quantization Utils ===")
    try:
        from quant_utils import (
            is_quantized_model,
            detect_model_quantization,
            prepare_model_with_quantization,
            cleanup_memory,
        )

        print("All quant_utils functions imported")

        cleanup_memory()
        print("cleanup_memory() works")

        return True
    except Exception as e:
        print(f"quant_utils check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_minimal_model_creation():
    print("\n=== Minimal Model Test ===")
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
        )

        model = LlamaForCausalLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Created tiny test model: {param_count:,} parameters")

        del model
        torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f" Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("ENVIRONMENT VALIDATION")
    print("=" * 60)

    checks = []

    print("\n=== Core Dependencies ===")
    checks.append(check_import("torch", "PyTorch"))
    checks.append(check_import("transformers", "Transformers"))
    checks.append(check_import("datasets", "Datasets"))
    checks.append(check_import("trl", "TRL"))
    checks.append(check_import("tokenizers", "Tokenizers"))
    checks.append(check_import("accelerate", "Accelerate"))

    checks.append(check_cuda())
    checks.append(check_quantization_support())
    checks.append(check_peft_support())
    checks.append(check_flash_attention())
    checks.append(check_deepspeed())
    checks.append(check_config())
    checks.append(check_quant_utils())
    checks.append(check_minimal_model_creation())

    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"RESULTS: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("\nAll checks passed! Environment is ready.")
        return 0
    else:
        print(f"\n{total - passed} checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
