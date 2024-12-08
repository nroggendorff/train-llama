# Train LLaMA

A containerized pipeline for training and fine-tuning LLaMA models using DeepSpeed and Hugging Face's TRL library.

## Overview

This project provides a streamlined workflow for training LLaMA models with the following features:

- Configurable model architecture and training parameters
- Support for both pre-training and instruction fine-tuning
- Distributed training using DeepSpeed
- Automatic data preprocessing and tokenization
- Hugging Face Hub integration for model hosting

## Configuration

The training pipeline can be configured through a JSON configuration file. Reference the [config file](./Build/config) for available parameters:

Key parameters include:

- `batch-size`: Training batch size
- `epochs`: Number of training epochs
- `learning-rate`: Model learning rate
- `max-seq-length`: Maximum sequence length
- `input-dataset`: Dataset for pre-training
- `instruct-dataset`: Dataset for instruction fine-tuning
- `output-repo`: Target repository for saving models
- `instruct-finetune-bool`: Toggle between pre-training and instruction fine-tuning

## Requirements

- Docker
- NVIDIA GPU with CUDA support

Dependencies are managed through the requirements.txt

## Usage

1. Configure your training parameters in `Build/config`

2. Build the Docker image:

```bash
docker buildx build Build -f Build/Dockerfile -t nroggendorff/train-llama:latest
```

3. Run training:

```bash
docker run --gpus all -it nroggendorff/train-llama:latest
```

## GitHub Actions

The project includes automated workflows for:

- Building and pushing Docker images
- Triggering training runs on external compute resources
- Managing model versions through pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- The pipeline uses Git LFS for handling large files
- Models are automatically pushed to Hugging Face Hub when `push-to-hub` is enabled
- Training can be resumed from checkpoints by adjusting the `init` parameter
