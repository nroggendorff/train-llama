import torch
import gc
from transformers import AutoModelForCausalLM, AutoConfig


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()


def is_quantized_model(model):
    if model is None:
        return False

    try:
        for name, param in model.named_parameters():
            if hasattr(param, "quant_state"):
                return True
            if param.dtype == torch.uint8:
                return True
            param_type = str(type(param))
            if "Int8Params" in param_type:
                return True

        for name, module in model.named_modules():
            module_type = str(type(module))
            if "Linear4bit" in module_type or "Linear8bit" in module_type:
                return True

        return False
    except Exception as e:
        print(f"Warning: Could not determine quantization state: {e}")
        return False


def check_bitsandbytes_available():
    try:
        import bitsandbytes

        return True
    except ImportError:
        return False


def dequantize_model(model):
    print("Dequantizing model from 4-bit to bfloat16...")

    if not check_bitsandbytes_available():
        raise ImportError("bitsandbytes is required for dequantization")

    try:
        from bitsandbytes.functional import dequantize_4bit
    except ImportError:
        raise ImportError("Could not import dequantize_4bit from bitsandbytes")

    config = model.config
    dequantized_state_dict = {}
    failed_params = []

    print("Extracting and dequantizing parameters...")
    for name, param in model.named_parameters():
        try:
            if hasattr(param, "quant_state") and param.quant_state is not None:
                dequantized = dequantize_4bit(
                    param.data, param.quant_state, quant_type="nf4"
                )
                dequantized_state_dict[name] = dequantized.to(torch.bfloat16).cpu()
            else:
                dequantized_state_dict[name] = param.data.to(torch.bfloat16).cpu()

            if param.data.device.type == "cuda":
                del param
                cleanup_memory()

        except Exception as e:
            print(f"Warning: Failed to dequantize {name}: {e}")
            failed_params.append(name)
            try:
                if hasattr(param, "shape"):
                    shape = param.shape
                else:
                    shape = (1,)
                dequantized_state_dict[name] = torch.zeros(
                    shape, dtype=torch.bfloat16, device="cpu"
                )
            except Exception as fallback_e:
                print(f"Could not create fallback tensor for {name}: {fallback_e}")

    if failed_params:
        print(
            f"Warning: {len(failed_params)} parameters could not be dequantized properly"
        )

    print("Creating new model with dequantized weights...")
    del model
    cleanup_memory()

    new_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    missing_keys, unexpected_keys = new_model.load_state_dict(
        dequantized_state_dict, strict=False
    )

    if missing_keys:
        print(f"Missing keys (will use random init): {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)} keys")

    del dequantized_state_dict
    cleanup_memory()

    print("Dequantization complete.")
    return new_model


def quantize_model_weights(model, quantization_config):
    print("Converting model to 4-bit quantization...")

    if not check_bitsandbytes_available():
        raise ImportError("bitsandbytes is required for quantization")

    config = model.config

    print("Extracting state dict...")
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v.cpu().clone()
        if v.device.type == "cuda":
            cleanup_memory()

    del model
    cleanup_memory()

    print("Creating quantized model...")
    try:
        quantized_model = AutoModelForCausalLM.from_config(
            config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Error creating quantized model: {e}")
        raise

    print("Loading weights into quantized model...")
    try:
        missing, unexpected = quantized_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys during quantization: {len(missing)} keys")
        if unexpected:
            print(f"Unexpected keys during quantization: {len(unexpected)} keys")
    except Exception as e:
        print(f"Warning during weight loading: {e}")
        print("Attempting parameter-by-parameter loading...")

        loaded_count = 0
        failed_count = 0
        for name in list(state_dict.keys()):
            try:
                param = quantized_model.get_parameter(name)
                with torch.no_grad():
                    if hasattr(param, "copy_"):
                        param.copy_(state_dict[name].to(param.device))
                    else:
                        param.data = state_dict[name].to(param.device)
                loaded_count += 1
            except Exception as param_e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"Failed to load {name}: {param_e}")

        print(f"Loaded {loaded_count} parameters, {failed_count} failed")

    del state_dict
    cleanup_memory()

    print("Preparing model for k-bit training...")
    try:
        from peft import prepare_model_for_kbit_training

        quantized_model = prepare_model_for_kbit_training(
            quantized_model, use_gradient_checkpointing=True
        )
    except ImportError:
        raise ImportError("peft is required for k-bit training preparation")
    except Exception as e:
        print(f"Warning during k-bit preparation: {e}")

    print("Quantization complete.")
    return quantized_model


def detect_model_quantization(model_path):
    try:
        print(f"Detecting quantization state of {model_path}...")

        config = AutoConfig.from_pretrained(model_path)

        if (
            hasattr(config, "quantization_config")
            and config.quantization_config is not None
        ):
            print("Model has quantization config in config.json")
            return True

        try:
            import os
            import json

            config_path = (
                os.path.join(model_path, "config.json")
                if os.path.isdir(model_path)
                else None
            )
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    if "quantization_config" in config_dict:
                        print("Found quantization_config in config.json")
                        return True
        except Exception as e:
            print(f"Could not check config.json: {e}")

        print("Loading small portion of model to check quantization...")
        try:
            test_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={0: "100MB", "cpu": "2GB"},
            )

            is_quant = is_quantized_model(test_model)
            del test_model
            cleanup_memory()

            return is_quant
        except Exception as load_err:
            print(f"Could not load model for inspection: {load_err}")
            return False

    except Exception as e:
        print(f"Warning: Could not detect quantization state: {e}")
        print("Assuming model is not quantized")
        return False


def prepare_model_with_quantization(model_path, use_quantization, tokenizer=None):
    from config import Config

    config = Config()

    model_is_quantized = detect_model_quantization(model_path)

    print(
        f"Model current state: {'quantized' if model_is_quantized else 'not quantized'}"
    )
    print(f"Target state: {'quantized' if use_quantization else 'not quantized'}")

    if use_quantization and not model_is_quantized:
        print("Path: Loading FP model → Converting to 4-bit")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            if tokenizer:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception as resize_e:
                    print(f"Warning: Could not resize embeddings: {resize_e}")

            model = quantize_model_weights(model, config.get_quantization_config())

        except Exception as e:
            print(f"Error during FP→4bit conversion: {e}")
            raise

    elif not use_quantization and model_is_quantized:
        print("Path: Loading 4-bit model → Converting to BF16")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=config.get_quantization_config(),
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            model = dequantize_model(model)

            if tokenizer:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception as resize_e:
                    print(f"Warning: Could not resize embeddings: {resize_e}")

        except Exception as e:
            print(f"Error during 4bit→BF16 conversion: {e}")
            raise

    elif use_quantization and model_is_quantized:
        print("Path: Loading 4-bit model (no conversion)")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=config.get_quantization_config(),
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            if tokenizer:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception as resize_e:
                    print(f"Warning: Could not resize embeddings: {resize_e}")

            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        except Exception as e:
            print(f"Error loading quantized model: {e}")
            raise

    else:
        print("Path: Loading FP model (no conversion)")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if use_quantization else torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            if tokenizer:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception as resize_e:
                    print(f"Warning: Could not resize embeddings: {resize_e}")

        except Exception as e:
            print(f"Error loading FP model: {e}")
            raise

    cleanup_memory()
    return model
