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

The training pipeline can be configured through environment variables. Reference the [config file](./config.py) for available parameters:

Key parameters include:

- `batch-size`: Training batch size
- `epochs`: Number of training epochs
- `learning-rate`: Model learning rate
- `max-seq-length`: Maximum sequence length
- `input-dataset`: Dataset for pre-training
- `instruct-dataset`: Dataset for instruction fine-tuning
- `output-repo`: Target repository for saving models

## Requirements

- a huggingface account
- linked payment information with at least ten USD balance
- a prepaid debit card; reference [this post](https://huggingface.co/posts/nroggendorff/896561565033687) to know how to use it without risk of going bankrupt. (optional)

Dependencies are managed through the [requirements text](./requirements.txt).

## Usage

0. Duplicate [nroggendorff/train-llama](https://huggingface.co/spaces/nroggendorff/train-llama) and configure variables and secrets.

[Duplicate Space Hotlink](https://huggingface.co/spaces/nroggendorff/train-llama?duplicate=true)
The first run must use `INIT=0` and `INST=false`.

## GitHub Actions

The project includes automated workflows for:

- Building and pushing Docker images
- Triggering training runs on external compute resources
- Managing model versions through pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please adheer to [CONTRIBUTING](./CONTRIBUTING.md).

## Notes

- The pipeline uses Git LFS for handling large files.
- Models are automatically pushed to Hugging Face Hub when `push-to-hub` is enabled.
- Training can be resumed from checkpoints by adjusting the `INIT` and `INST` environment variables.
