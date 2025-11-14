from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import warnings

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

config = Config()
download_config = DownloadConfig(max_retries=config.MAX_RETRIES)

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.utils.checkpoint"
)
warnings.filterwarnings("ignore", message=".*Flash Attention 2.0.*")


def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = (
            config.INPUT_REPO + f"-{config.INST_SUFFIX}"
            if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
            else config.INPUT_REPO
        )

        def load_model_from_pretrained():
            return AutoModelForCausalLM.from_pretrained(model_path, use_cache=False)

        model = retry_on_failure(load_model_from_pretrained)
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise

    if dist.is_initialized():
        dist.barrier()

    return model


def create_model(tokenizer):
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    hidden_size = int(config.FACTOR)

    preferred_head_dim = 128
    num_attention_heads = max(1, hidden_size // preferred_head_dim)

    while num_attention_heads > 1 and (hidden_size % num_attention_heads) != 0:
        num_attention_heads -= 1

    actual_head_dim = hidden_size // num_attention_heads

    num_hidden_layers = max(1, hidden_size // 128)

    intermediate_size = hidden_size * 4
    print(
        f"Creating model with hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}, num_attention_heads={num_attention_heads}, head_dim={actual_head_dim}, intermediate_size={intermediate_size}"
    )
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=config.MAX_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        bos_token_id=getattr(tokenizer, "bos_token_id", 1),
        eos_token_id=getattr(tokenizer, "eos_token_id", 2),
        tie_word_embeddings=False,
    )

    try:
        model = LlamaForCausalLM(model_config)
    except Exception as e:
        print(f"Failed to create model with the derived topology: {e}")
        raise

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    return model


def load_data():
    dataset = load_dataset(
        (
            config.INSTRUCT_DATASET
            if config.INSTRUCT_FINETUNE_BOOL
            else config.INPUT_DATASET
        ),
        split="train",
        streaming=True,
        download_config=download_config,
    )
    dataset = dataset.skip(config.SKIP_SAMPLES).take(config.SHARD_SIZE)
    return dataset


def load_full_dataset():
    return load_dataset(
        config.INPUT_DATASET,
        split="train",
        streaming=True,
        download_config=download_config,
    )


def format_prompts(examples, tokenizer, isinst):
    custom_processor = config.get_custom_processor()

    def process_text(text):
        if not text or len(text.strip()) == 0:
            return None
        try:
            return custom_processor(text, tokenizer, isinst)
        except Exception as e:
            print(f"Custom processor error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        texts = list(executor.map(process_text, examples["text"]))

    texts = [t for t in texts if t is not None]

    if len(texts) == 0:
        raise ValueError("No valid texts found in examples for formatting.")

    return {"text": texts}


def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=config.VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens,
    )
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    return fast_tokenizer


def load_tokenizer():
    def do_load():
        if config.INPUT_TOKENIZER != config.INPUT_REPO and config.INIT == 0:
            return AutoTokenizer.from_pretrained(config.INPUT_TOKENIZER)
        return AutoTokenizer.from_pretrained(
            config.INPUT_REPO + f"-{config.INST_SUFFIX}"
            if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
            else config.INPUT_REPO
        )

    return retry_on_failure(do_load)


def get_training_corpus(dataset):
    buffer = []
    for i, example in enumerate(dataset):
        if example["text"] and len(example["text"].strip()) > 0:
            buffer.append(example["text"])
            if len(buffer) >= 1000:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def configure_tokenizer(tokenizer):
    if config.INSTRUCT_FINETUNE_BOOL and check_tokenizer_has_instruct_config(tokenizer):
        print("Skipping tokenizer configuration - already configured for instructions")
        return

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
    tokenizer.add_special_tokens(special_tokens)

    if config.INSTRUCT_FINETUNE_BOOL:
        tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

        chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        tokenizer.chat_template = chat_template


def main(is_inst=config.INSTRUCT_FINETUNE_BOOL):
    print("Getting Tokenizer..")
    if (
        config.INSTRUCT_FINETUNE_BOOL
        or config.INIT > 0
        or config.INPUT_TOKENIZER != config.INPUT_REPO
    ):
        tokenizer = load_tokenizer()
        print(f"Got Tokenizer with size {len(tokenizer)}.")

        print("Loading Data..")
        dataset = load_data()
        print("Loaded data.")
    else:
        print("Loading full dataset for tokenizer creation..")
        full_dataset = load_full_dataset()
        print("Making Corpus from full dataset..")
        training_corpus = get_training_corpus(full_dataset)
        print("Made Corpus.")
        tokenizer = create_tokenizer(training_corpus)
        print(f"Created Tokenizer with size {len(tokenizer)}.")

        print("Loading Data..")
        dataset = load_data()
        print("Loaded data.")

    print("Adding Special Tokens..")
    configure_tokenizer(tokenizer)
    print("Added Tokens.")

    print("Processing and saving data in streaming mode..")

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

        if config.INIT == 0 and not config.INSTRUCT_FINETUNE_BOOL:
            total_samples = None
        else:
            total_samples = config.SHARD_SIZE
        with tqdm(desc="Processing examples", total=total_samples) as pbar:
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

    print(f"Creating dataset from {len(processed_data)} processed examples...")
    final_dataset = Dataset.from_list(processed_data)
    processed_data = None

    print("Getting Model..")
    try:
        model = (
            load_model(tokenizer)
            if is_inst or config.INIT > 0
            else create_model(tokenizer)
        )
        print("Got Model.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise

    if config.FP16:
        model = model.half()

    print("Saving Prepared Data..")
    final_dataset.save_to_disk("prepared_dataset")
    tokenizer.save_pretrained("prepared_tokenizer")
    model.save_pretrained("prepared_model")
    print("Prepared data saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in main: {e}")
        import traceback

        traceback.print_exc()
        try:
            from util import Space

            Space().stop(e)
        except Exception:
            pass
        raise
