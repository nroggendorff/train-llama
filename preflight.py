import os
import sys
import torch


def check_environment_variables():
    print("\n=== Environment Variables ===")
    required = ["HF_TOKEN", "SPACE_ID"]
    optional = {
        "INIT": "0",
        "INST": "false",
        "USE_QUANT": "false",
        "BATCH_SIZE": "4",
        "LEARNING_RATE": "3e-4",
        "MAX_LENGTH": "2048",
        "EPOCHS": "3",
    }

    errors = []
    for var in required:
        value = os.environ.get(var)
        if not value:
            print(f" {var}: NOT SET")
            errors.append(f"Required variable {var} is not set")
        else:
            masked = "***" if "TOKEN" in var else value
            print(f"{var}: {masked}")

    for var, default in optional.items():
        value = os.environ.get(var, default)
        print(f"  {var}: {value}")

    return errors


def check_gpu_memory():
    print("\n=== GPU Memory ===")
    errors = []

    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        print(" CUDA not available")
        return errors

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)

        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        used_gb = total_gb - free_gb

        print(f"GPU {i}: {props.name}")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used: {used_gb:.1f} GB")
        print(f"  Free: {free_gb:.1f} GB")

        use_quant = os.environ.get("USE_QUANT", "false").lower() == "true"
        factor = int(os.environ.get("FACTOR", "12288"))
        batch_size = int(os.environ.get("BATCH_SIZE", "4"))
        max_length = int(os.environ.get("MAX_LENGTH", "2048"))

        estimated_params = (factor * 12) * (factor // 128) / 1_000_000

        if use_quant:
            estimated_memory = (estimated_params * 0.5) + (
                batch_size * max_length * 0.002
            )
        else:
            estimated_memory = (estimated_params * 2) + (
                batch_size * max_length * 0.004
            )

        print(f"  Estimated model size: ~{estimated_params:.0f}M params")
        print(f"  Estimated memory needed: ~{estimated_memory:.1f} GB")

        if free_gb < estimated_memory:
            warning = f"GPU {i} may not have enough memory ({free_gb:.1f} GB free, ~{estimated_memory:.1f} GB needed)"
            print(f"  ⚠ {warning}")
            if not use_quant:
                print(f"  → Consider setting USE_QUANT=true to reduce memory usage")
        else:
            print(f"   Sufficient memory available")

    return errors


def check_disk_space():
    print("\n=== Disk Space ===")
    errors = []

    try:
        import shutil

        total, used, free = shutil.disk_usage("/")

        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)

        print(f"Total: {total_gb:.1f} GB")
        print(f"Used: {used_gb:.1f} GB")
        print(f"Free: {free_gb:.1f} GB")

        shard_size = int(os.environ.get("SHARD_SIZE", "131072"))
        max_length = int(os.environ.get("MAX_LENGTH", "2048"))
        factor = int(os.environ.get("FACTOR", "12288"))

        estimated_dataset_gb = (shard_size * max_length * 2) / (1024**3)
        estimated_model_gb = ((factor * 12) * (factor // 128) * 2) / (1024**3)
        estimated_total_gb = estimated_dataset_gb + estimated_model_gb + 5

        print(f"Estimated space needed: ~{estimated_total_gb:.1f} GB")
        print(f"  Dataset: ~{estimated_dataset_gb:.1f} GB")
        print(f"  Model: ~{estimated_model_gb:.1f} GB")
        print(f"  Overhead: ~5 GB")

        if free_gb < estimated_total_gb:
            error = f"Insufficient disk space ({free_gb:.1f} GB free, ~{estimated_total_gb:.1f} GB needed)"
            print(f" {error}")
            errors.append(error)
        else:
            print(" Sufficient disk space available")

    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")

    return errors


def check_quantization_compatibility():
    print("\n=== Quantization Compatibility ===")
    errors = []

    use_quant = os.environ.get("USE_QUANT", "false").lower() == "true"

    if not use_quant:
        print("Quantization disabled, skipping checks")
        return errors

    print("Quantization enabled, checking dependencies...")

    try:
        import bitsandbytes

        print(f" bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        error = "bitsandbytes not installed (required for quantization)"
        print(f" {error}")
        errors.append(error)

    try:
        from peft import LoraConfig

        print(" peft available")
    except ImportError:
        error = "peft not installed (required for quantized training)"
        print(f" {error}")
        errors.append(error)

    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        print(
            f"GPU Compute Capability: {compute_capability[0]}.{compute_capability[1]}"
        )

        if compute_capability[0] < 7:
            warning = f"GPU compute capability {compute_capability[0]}.{compute_capability[1]} may have limited quantization support"
            print(f"⚠ {warning}")

    return errors


def check_repositories():
    print("\n=== Repository Access ===")
    errors = []

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        input_repo = os.environ.get("INPUT_REPO", os.environ.get("OUTPUT_REPO", ""))
        output_repo = os.environ.get("OUTPUT_REPO", "")
        init = int(os.environ.get("INIT", "0"))
        inst = os.environ.get("INST", "false").lower() == "true"

        if init > 0 or inst:
            check_repo = input_repo + "-it" if inst and init > 0 else input_repo
            print(f"Checking input repository: {check_repo}")
            try:
                info = api.repo_info(check_repo, repo_type="model")
                print(f" Input repository accessible")
            except Exception as e:
                error = f"Cannot access input repository {check_repo}: {e}"
                print(f" {error}")
                errors.append(error)

        if output_repo:
            print(f"Checking output repository: {output_repo}")
            try:
                try:
                    info = api.repo_info(output_repo, repo_type="model")
                    print(f" Output repository exists and accessible")
                except:
                    print(f"  Repository doesn't exist yet (will be created)")
            except Exception as e:
                print(f"⚠ Could not verify output repository: {e}")

    except Exception as e:
        print(f"⚠ Repository check failed: {e}")

    return errors


def check_datasets():
    print("\n=== Dataset Access ===")
    errors = []

    input_ds = os.environ.get("INPUT_DS", "nroggendorff/microrpus")
    inst_ds = os.environ.get("INST_DS", "nroggendorff/elephant")
    inst = os.environ.get("INST", "false").lower() == "true"

    dataset_to_check = inst_ds if inst else input_ds

    print(f"Checking dataset: {dataset_to_check}")

    try:
        from datasets import load_dataset

        test_ds = load_dataset(
            dataset_to_check,
            split="train",
            streaming=True,
        )

        sample = next(iter(test_ds.take(1)))

        if "text" not in sample:
            error = f"Dataset missing 'text' field"
            print(f" {error}")
            errors.append(error)
        else:
            print(f" Dataset accessible with 'text' field")
            print(f"  Sample length: {len(sample['text'])} chars")

    except Exception as e:
        error = f"Cannot access dataset {dataset_to_check}: {e}"
        print(f" {error}")
        errors.append(error)

    return errors


def check_config_sanity():
    print("\n=== Configuration Sanity ===")
    warnings = []

    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "3e-4"))
    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    epochs = float(os.environ.get("EPOCHS", "3"))
    factor = int(os.environ.get("FACTOR", "12288"))

    effective_batch = batch_size * grad_accum
    print(f"Effective batch size: {effective_batch} ({batch_size} * {grad_accum})")

    if effective_batch < 4:
        warnings.append("Very small effective batch size, training may be unstable")
        print(f"⚠ {warnings[-1]}")

    if learning_rate > 1e-3:
        warnings.append("High learning rate, may cause training instability")
        print(f"⚠ {warnings[-1]}")

    if learning_rate < 1e-5:
        warnings.append("Very low learning rate, training may be slow")
        print(f"⚠ {warnings[-1]}")

    if max_length > 4096:
        warnings.append("Very long sequences, may require significant memory")
        print(f"⚠ {warnings[-1]}")

    if epochs < 0.5:
        warnings.append("Very few epochs, model may underfit")
        print(f"⚠ {warnings[-1]}")

    if factor % 128 != 0:
        warnings.append("FACTOR not multiple of 128, may cause architecture issues")
        print(f"⚠ {warnings[-1]}")

    if not warnings:
        print(" Configuration looks reasonable")

    return warnings


def main():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)

    all_errors = []
    all_warnings = []

    all_errors.extend(check_environment_variables())
    all_errors.extend(check_gpu_memory())
    all_errors.extend(check_disk_space())
    all_errors.extend(check_quantization_compatibility())
    all_errors.extend(check_repositories())
    all_errors.extend(check_datasets())
    all_warnings.extend(check_config_sanity())

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if all_errors:
        print(f"\n {len(all_errors)} ERROR(S) FOUND:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix these errors before proceeding.")
        return 1

    if all_warnings:
        print(f"\n⚠ {len(all_warnings)} WARNING(S):")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
        print("\nWarnings are non-critical but should be reviewed.")

    if not all_errors and not all_warnings:
        print("\n All pre-flight checks passed!")
    elif not all_errors:
        print("\n No critical errors found. Proceeding with warnings.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"PRE-FLIGHT CHECK FAILED")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
