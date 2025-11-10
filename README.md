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

- `BATCH_SIZE`: Training batch size
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Model learning rate
- `MAX_LENGTH`: Maximum sequence length
- `INPUT_DATASET`: Dataset for pre-training
- `INSTRUCT_DATASET`: Dataset for instruction fine-tuning
- `INPUT_TOKENIZER`: Pre-trained tokenizer to use (skips tokenizer creation)
- `OUTPUT_REPO`: Target repository for saving models

## Requirements

- a huggingface account
- a huggingface token with write access
- linked payment information with at least ten USD balance
- a prepaid debit card; reference [this post](https://huggingface.co/posts/nroggendorff/896561565033687) to know how to use it without risk of going bankrupt. (optional)

Dependencies are managed through the [requirements script](./installer.sh).

## Usage

Duplicate [nroggendorff/train-llama](https://huggingface.co/spaces/nroggendorff/train-llama) and configure variables and secrets.

[Duplicate Space Hotlink](https://huggingface.co/spaces/nroggendorff/train-llama?duplicate=true)
The first run must use `INIT=0` and `INST=false`.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please adhere to [CONTRIBUTING](./CONTRIBUTING.md).

## Notes

- The pipeline uses Git LFS for handling large files.
- Models are automatically pushed to Hugging Face Hub when `PUSH_TO_HUB` is enabled.
- Training can be resumed from checkpoints by adjusting the `INIT` and `INST` environment variables.
- When `INPUT_TOKENIZER` is set, tokenizer creation is skipped on the first run, and only the required data is loaded.
