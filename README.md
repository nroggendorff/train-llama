# Train LLaMA

A containerized pipeline for training and fine-tuning LLaMA models using DeepSpeed and Hugging Face's TRL library.

## Overview

This project provides a streamlined workflow for training LLaMA models with the following features:

- Configurable model architecture and training parameters
- Support for both pre-training and instruction fine-tuning
- Distributed training using DeepSpeed
- **4-bit quantization support with automatic weight conversion**
- Automatic data preprocessing and tokenization
- Hugging Face Hub integration for model hosting
- Comprehensive validation and error handling

## Configuration

The training pipeline can be configured through environment variables. Reference the [config file](./config.py) for available parameters:

### Key Parameters

- `BATCH_SIZE`: Training batch size (default: 4)
- `EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Model learning rate (default: 3e-4)
- `MAX_LENGTH`: Maximum sequence length (default: 2048)
- `INPUT_DATASET`: Dataset for pre-training (default: "nroggendorff/microrpus")
- `INSTRUCT_DATASET`: Dataset for instruction fine-tuning (default: "nroggendorff/elephant")
- `OUTPUT_REPO`: Target repository for saving models
- **`USE_QUANT`**: Enable 4-bit quantization (default: `false`)

## Quantization

The pipeline now supports 4-bit quantization using the NF4 quantization type with double quantization. This feature significantly reduces memory usage (up to 75%) and enables training larger models on limited hardware.

### Key Features

- **4-bit NF4 quantization** with double quantization for maximum compression
- **BFloat16 compute dtype** for numerical stability
- **Automatic weight conversion** between formats:
  - Converts 16-bit models to 4-bit when `USE_QUANT=true`
  - Converts 4-bit models to 16-bit when `USE_QUANT=false`
  - Handles format mismatches gracefully during training resumption
- **LoRA (Low-Rank Adaptation)** for efficient quantized model training
- **Automatic adapter merging** before model upload

### Usage

Enable quantization by setting the environment variable:

```bash
USE_QUANT=true
```

### Memory Savings

Example memory usage for a 1B parameter model:

| Configuration | Memory Usage | Savings |
|--------------|--------------|---------|
| FP16 (no quant) | ~8 GB | - |
| 4-bit + LoRA | ~2 GB | 75% |

### How Weight Conversion Works

The system intelligently handles weight format conversion:

1. **Detection Phase**: Automatically detects if a model is quantized by:
   - Checking model configuration
   - Inspecting parameter data types
   - Analyzing module types

2. **Conversion Phase**:
   - **16-bit → 4-bit**: Quantizes weights using NF4, applies LoRA adapters
   - **4-bit → 16-bit**: Dequantizes weights, pads missing values with zeros
   - **Same Format**: No conversion needed, loads directly

3. **Training Phase**:
   - Quantized models use LoRA for parameter-efficient training
   - Full precision models train all parameters
   - Gradient checkpointing enabled for memory efficiency

4. **Upload Phase**:
   - LoRA adapters automatically merged back into base model
   - Saves as standard format compatible with all frameworks

### Performance Considerations

**When to use quantization:**
- Limited GPU memory (<16 GB)
- Training larger models (>1B parameters)
- Rapid prototyping with faster iteration
- Cost optimization on cloud GPUs

**When to use full precision:**
- Maximum model quality needed
- Sufficient GPU memory available
- Final production training runs
- Research requiring precise numerical behavior

## Requirements

- A Hugging Face account
- A Hugging Face token with write access
- Linked payment information with at least $10 USD balance
- Optional: A prepaid debit card (see [this post](https://huggingface.co/posts/nroggendorff/896561565033687))

Dependencies are automatically managed through the Docker container.

## Usage

1. **Duplicate the Space**: [Duplicate Space Hotlink](https://huggingface.co/spaces/nroggendorff/train-llama?duplicate=true)

2. **Configure Environment Variables** in the Space settings:
   ```bash
   HF_TOKEN=your_token_here
   OUTPUT_REPO=your_username/model_name
   USE_QUANT=true  # Optional: Enable quantization
   BATCH_SIZE=4    # Optional: Adjust for your GPU
   ```

3. **First Run**: Must use `INIT=0` and `INST=false`

4. **Subsequent Runs**: The system automatically increments `INIT` and toggles `INST`

## Validation and Safety

The pipeline includes comprehensive safety checks:

- **Pre-flight checks** (`preflight.py`): Validates environment before training
- **Environment validation** (`validate_setup.py`): Verifies all dependencies
- **Memory estimation**: Warns if GPU memory may be insufficient
- **Automatic error recovery**: Retries failed operations with exponential backoff
- **Graceful degradation**: Continues when non-critical operations fail

Run validation manually:
```bash
python3 preflight.py
python3 validate_setup.py
```

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed troubleshooting guidance covering:
- Quantization issues
- Memory problems
- Model loading errors
- Training failures
- Network and upload issues

Quick diagnostics:
```bash
python3 validate_setup.py  # Check environment
nvidia-smi  # Check GPU status
```

## Files Overview

- `config.py`: Central configuration management
- `prep.py`: Data preprocessing and model preparation
- `train.py`: Training loop with DeepSpeed
- `quant_utils.py`: Quantization and weight conversion utilities
- `util.py`: Helper functions (upload, timing, space management)
- `trainer.sh`: Main entry point orchestrating the pipeline
- `preflight.py`: Pre-training safety checks
- `validate_setup.py`: Environment validation
- `Dockerfile`: Container definition with all dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please adhere to [CONTRIBUTING](./CONTRIBUTING.md).

## Notes

- The pipeline uses Git LFS for handling large files
- Models are automatically pushed to Hugging Face Hub when training completes
- Training can be resumed from checkpoints by adjusting `INIT` and `INST` variables
- **Quantized training uses LoRA adapters which are merged back into the base model before uploading**
- **Weight conversion between quantization formats is automatic and transparent**
- **All operations include retry logic for network resilience**

## Advanced Configuration

### Custom Model Architectures

Adjust the `FACTOR` variable to control model size:
```bash
FACTOR=6144   # Smaller model (~300M params)
FACTOR=12288  # Default (~1B params)
FACTOR=24576  # Larger model (~4B params)
```

### Memory Optimization

For limited GPU memory:
```bash
USE_QUANT=true
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
MAX_LENGTH=1024
```

### Training Speed vs Quality

For faster training:
```bash
USE_QUANT=true
BATCH_SIZE=8
MAX_LENGTH=512
```

For highest quality:
```bash
USE_QUANT=false
BATCH_SIZE=4
MAX_LENGTH=2048
EPOCHS=5
```
