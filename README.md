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
- `CUSTOM_PROCESSOR`: Custom dataset processing function (see below)
- `PRECISION`: `auto` (default), `bf16`, `fp16`, or `fp32`. `auto` uses bf16 on GPUs that support it (Ampere and newer) and falls back to fp16 otherwise, since bf16 avoids fp16's loss-scaling instability.
- `TIE_WORD_EMBEDDINGS`: `true` (default) or `false`. Tying the input/output embeddings meaningfully reduces parameter count for small models (the embedding table alone can be ~40% of total parameters at the default 200M/52k-vocab settings), leaving more of the parameter budget for the transformer body.
- `NUM_KEY_VALUE_HEADS`: Overrides the auto-computed number of key/value heads used for grouped-query attention (GQA). By default this is derived from `NUM_ATTENTION_HEADS` (roughly a 4:1 ratio, adjusted to the nearest valid divisor).
- `MAX_CONSECUTIVE_NAN_STEPS`: Number of consecutive non-finite loss values tolerated before training aborts with an error (default 50). Individual non-finite steps are skipped (not backpropagated) rather than corrupting the model.

### Custom Dataset Processing

The `CUSTOM_PROCESSOR` environment variable allows you to define custom logic for processing your dataset. This is useful when working with datasets in different formats.

The processor should contain the body of a Python function (without the `def` statement) that:
- Takes three parameters: `text`, `tok`, and `isinst`
- Processes the text according to your format
- Stores the result in a variable named `result`

**Example for simple text wrapping:**
```python
result = tok.bos_token + text.strip() + tok.eos_token
```

**Example for JSON-based datasets:**
```python
import json; data = json.loads(text); result = tok.bos_token + data['content'] + tok.eos_token
```

**Example for custom chat format:**
```python
messages = []; lines = text.split('\n'); [messages.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': line}) for i, line in enumerate(lines) if line.strip()]; result = tok.apply_chat_template(messages, tokenize=False) if isinst else tok.bos_token + text + tok.eos_token
```

## Requirements

- a huggingface account
- a huggingface token with write access
- linked payment information with at least ten USD balance
- a prepaid debit card; reference [this post](https://huggingface.co/posts/nroggendorff/896561565033687) to know how to use it without risk of going bankrupt. (optional)

Dependencies are managed through the [requirements script](./installer.sh).

## Usage

Duplicate [nroggendorff/train-llama](https://huggingface.co/spaces/nroggendorff/train-llama) and configure variables and secrets.

[Duplicate Space Hotlink](https://huggingface.co/spaces/nroggendorff/train-llama?duplicate=true)
The first run must use `RESUME=false` and `INST=false`.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please adhere to [CONTRIBUTING](./CONTRIBUTING.md).

## Notes

- The pipeline uses Git LFS for handling large files.
- Models are automatically pushed to Hugging Face Hub when running as a Space.
- When `INPUT_TOKENIZER` is set, tokenizer creation is skipped on the first run, and only the required data is loaded.
- Each run saves a small `training_state.json` alongside the model (recording the global step and whether training fully finished). On the next `RESUME=true` run, this lets the learning-rate schedule continue smoothly instead of re-warming-up from scratch every time the Space restarts.
- The Space variables `RESUME` and `INST` are managed automatically once a run pushes a checkpoint: if a run is stopped early by the timeout, `RESUME` is flipped to `true` so the next restart continues the same phase; once a phase (pretraining or instruction fine-tuning) genuinely finishes all its steps, the pipeline advances on its own (pretraining hands off to instruction fine-tuning with `RESUME` reset to `false`; instruction fine-tuning pauses the Space when done).
