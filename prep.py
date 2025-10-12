from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

from datasets import load_dataset, Dataset, DownloadConfig
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
)
import torch.distributed as dist
import threading
from collections import deque

from config import Config
from util import *
from quant_utils import prepare_model_with_quantization, cleanup_memory

config = Config()
download_config = DownloadConfig(max_retries=config.MAX_RETRIES)


def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = (
            config.INPUT_REPO + "-it"
            if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
            else config.INPUT_REPO
        )

        print(f"Loading model from: {model_path}")

        def load_model_from_pretrained():
            return prepare_model_with_quantization(
                model_path, config.USE_QUANTIZATION, tokenizer
            )

        model = retry_on_failure(load_model_from_pretrained)
        print(f"Model loaded successfully")

    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        import traceback

        traceback.print_exc()
        raise

    if dist.is_initialized():
        dist.barrier()

    return model


def create_model(tokenizer):
    print("Creating new model from scratch...")

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    hidden_size = int(config.FACTOR)
    preferred_head_dim = 128
    num_attention_heads = max(1, hidden_size // preferred_head_dim)

    while num_attention_heads > 1 and (hidden_size % num_attention_heads) != 0:
        num_attention_heads -= 1

    actual_head_dim = hidden_size // num_attention_heads
    num_hidden_layers = max(1, hidden_size // 128)
    intermediate_size = hidden_size * 4

    print(f"Model architecture:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  Head dimension: {actual_head_dim}")
    print(f"  Intermediate size: {intermediate_size}")

    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=config.MAX_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        bos_token_id=getattr(tokenizer, "bos_token_id", 1),
        eos_token_id=getattr(tokenizer, "eos_token_id", 2),
        tie_word_embeddings=False,
    )

    try:
        torch_dtype = torch.bfloat16 if config.USE_QUANTIZATION else torch.float16
        model = LlamaForCausalLM(model_config)
        model = model.to(torch_dtype)
        print(f"Model created with dtype {torch_dtype}")
    except Exception as e:
        print(f"Failed to create model: {e}")
        raise

    try:
        old_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        new_size = model.get_input_embeddings().weight.shape[0]
        print(f"Resized embeddings: {old_size} → {new_size}")
    except Exception as e:
        print(f"Warning: Could not resize token embeddings: {e}")

    return model


def load_data():
    dataset_name = (
        config.INSTRUCT_DATASET
        if config.INSTRUCT_FINETUNE_BOOL
        else config.INPUT_DATASET
    )

    print(f"Loading dataset: {dataset_name}")
    print(f"  Skip: {config.SKIP_SAMPLES}")
    print(f"  Take: {config.SHARD_SIZE}")

    try:
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            download_config=download_config,
        )
        dataset = dataset.skip(config.SKIP_SAMPLES).take(config.SHARD_SIZE)
        return dataset
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        raise


def load_full_dataset():
    print(f"Loading full dataset for tokenizer training: {config.INPUT_DATASET}")
    try:
        return load_dataset(
            config.INPUT_DATASET,
            split="train",
            streaming=True,
            download_config=download_config,
        )
    except Exception as e:
        print(f"Failed to load full dataset: {e}")
        raise


def format_prompts(examples, tokenizer, isinst):
    def process_text(text):
        try:
            if not text or len(text.strip()) == 0:
                return None

            if isinst:
                conversation = []
                parts = text.split("<|end|>")
                for i in range(0, len(parts) - 1, 2):
                    if i + 1 < len(parts):
                        prompt = parts[i].replace("<|user|>", "").strip()
                        response = parts[i + 1].replace("<|bot|>", "").strip()

                        if prompt and response:
                            conversation.append({"role": "user", "content": prompt})
                            conversation.append(
                                {"role": "assistant", "content": response}
                            )

                if conversation:
                    return tokenizer.apply_chat_template(conversation, tokenize=False)
            else:
                return tokenizer.bos_token + text + tokenizer.eos_token

            return None
        except Exception as e:
            print(f"Warning: Error processing text: {e}")
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        texts = list(executor.map(process_text, examples["text"]))

    texts = [t for t in texts if t is not None]

    if len(texts) == 0:
        raise ValueError("No valid texts found in batch")

    return {"text": texts}


def create_tokenizer(training_corpus):
    print("Training new tokenizer...")
    try:
        tokenizer = ByteLevelBPETokenizer()
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

        tokenizer.train_from_iterator(
            training_corpus,
            vocab_size=config.VOCAB_SIZE,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True,
        )

        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
        print(f"Tokenizer trained with vocab size {len(fast_tokenizer)}")
        return fast_tokenizer
    except Exception as e:
        print(f"Failed to create tokenizer: {e}")
        raise


def load_tokenizer():
    tokenizer_path = (
        config.INPUT_REPO + "-it"
        if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
        else config.INPUT_REPO
    )

    print(f"Loading tokenizer from: {tokenizer_path}")

    def do_load():
        return AutoTokenizer.from_pretrained(tokenizer_path)

    tokenizer = retry_on_failure(do_load)
    print(f"Tokenizer loaded")
    return tokenizer


def get_training_corpus(dataset):
    buffer = []
    count = 0
    for example in dataset:
        if example["text"] and len(example["text"].strip()) > 0:
            buffer.append(example["text"])
            count += 1
            if len(buffer) >= 1000:
                yield buffer
                buffer = []
    if buffer:
        yield buffer
    print(f"  Processed {count} examples for tokenizer training")


def configure_tokenizer(tokenizer):
    print("Configuring tokenizer special tokens...")

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "additional_special_tokens": [],
    }

    if config.INSTRUCT_FINETUNE_BOOL:
        special_tokens["additional_special_tokens"] = ["<|user|>", "<|bot|>", "<|end|>"]

    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"  Added {num_added} special tokens")

    if config.INSTRUCT_FINETUNE_BOOL:
        tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

        chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        tokenizer.chat_template = chat_template
        print("Chat template configured")


def main(is_inst=config.INSTRUCT_FINETUNE_BOOL):
    print("=" * 60)
    print("PREPARATION PHASE")
    print("=" * 60)
    print(f"Mode: {'Instruction Tuning' if is_inst else 'Pre-training'}")
    print(f"Quantization: {config.USE_QUANTIZATION}")
    print(f"Init level: {config.INIT}")
    print("=" * 60)

    print("\n[1/4] Tokenizer Setup")
    if config.INSTRUCT_FINETUNE_BOOL or config.INIT > 0:
        tokenizer = load_tokenizer()
        print(f"Loaded existing tokenizer: {len(tokenizer)} tokens")

        print("\n[2/4] Data Loading")
        dataset = load_data()
        print("Dataset loaded")
    else:
        print("Creating new tokenizer from scratch...")
        full_dataset = load_full_dataset()
        training_corpus = get_training_corpus(full_dataset)
        tokenizer = create_tokenizer(training_corpus)

        print("\n[2/4] Data Loading")
        dataset = load_data()
        print("Dataset loaded")

    print("\n[3/4] Tokenizer Configuration")
    configure_tokenizer(tokenizer)
    print("Tokenizer configured")

    print("\n[4/4] Data Processing")
    print("Processing dataset in parallel...")

    def process_batch(batch, tokenizer, instruct_finetune_bool):
        batch_dict = {"text": [item["text"] for item in batch]}
        formatted_batch = format_prompts(batch_dict, tokenizer, instruct_finetune_bool)
        return [{"text": text} for text in formatted_batch["text"]]

    def batch_generator(dataset, batch_size):
        current_batch = []
        for example in dataset:
            current_batch.append(example)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

    processed_data = []
    batch_size = 100
    max_workers = 4
    max_queued_batches = max_workers * 2

    batch_gen = batch_generator(dataset, batch_size)
    result_lock = threading.Lock()

    def process_and_store(batch):
        result = process_batch(batch, tokenizer, config.INSTRUCT_FINETUNE_BOOL)
        with result_lock:
            processed_data.extend(result)
        return len(result)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = deque()
        total_processed = 0

        with tqdm(desc="Processing batches", unit="example") as pbar:
            for batch in batch_gen:
                if len(futures) >= max_queued_batches:
                    future = futures.popleft()
                    count = future.result()
                    total_processed += count
                    pbar.update(count)

                future = executor.submit(process_and_store, batch)
                futures.append(future)

            while futures:
                future = futures.popleft()
                count = future.result()
                total_processed += count
                pbar.update(count)

    print(f"Processed {len(processed_data)} examples")

    print("\nCreating HuggingFace dataset...")
    final_dataset = Dataset.from_list(processed_data)
    processed_data = None
    cleanup_memory()
    print(f"Dataset created: {len(final_dataset)} examples")

    print("\nModel Setup")
    try:
        if is_inst or config.INIT > 0:
            print("Loading existing model...")
            model = load_model(tokenizer)
        else:
            print("Creating new model...")
            model = create_model(tokenizer)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model ready: {param_count:,} parameters")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        import traceback

        traceback.print_exc()
        raise

    if not config.USE_QUANTIZATION and config.FP16:
        print("Converting model to FP16...")
        model = model.half()

    print("\nSaving Prepared Artifacts")
    print("  Saving dataset...")
    final_dataset.save_to_disk("prepared_dataset")
    print("Dataset saved")

    print("  Saving tokenizer...")
    tokenizer.save_pretrained("prepared_tokenizer")
    print("Tokenizer saved")

    print("  Saving model...")
    model.save_pretrained("prepared_model")
    print("Model saved")

    cleanup_memory()

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"PREPARATION FAILED")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")

        import traceback

        traceback.print_exc()

        try:
            from util import Space

            Space().stop(e)
        except Exception as space_error:
            print(f"Failed to stop space: {space_error}")

        raise
