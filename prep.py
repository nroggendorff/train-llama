from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
)
import torch.distributed as dist
import threading
from collections import deque

from config import Config
from util import *

config = Config()


def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = (
            config.INPUT_REPO + "-it"
            if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
            else config.INPUT_REPO
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
        )
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
    head_dim = 128

    if hidden_size < head_dim:
        head_dim = max(32, (hidden_size // 32) * 32)

    if hidden_size % head_dim != 0:
        hidden_size = (hidden_size // head_dim) * head_dim
        if hidden_size == 0:
            hidden_size = head_dim

    num_attention_heads = hidden_size // head_dim
    num_key_value_heads = num_attention_heads

    v_head_dim = head_dim
    qk_rope_head_dim = head_dim
    qk_nope_head_dim = 0

    intermediate_size = 4 * hidden_size
    intermediate_size = ((intermediate_size + 255) // 256) * 256

    num_hidden_layers = max(2, min(256, hidden_size // 128))

    n_group = 8
    base_experts = max(16, min(64, hidden_size // 128))
    n_routed_experts = ((base_experts + n_group - 1) // n_group) * n_group

    num_experts_per_tok = min(2, n_routed_experts // n_group)

    model_config = DeepseekV3Config(
        vocab_size=getattr(tokenizer, "vocab_size", 32000),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_position_embeddings=getattr(config, "MAX_LENGTH", 512),
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=getattr(config, "USE_CACHE", False),
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        bos_token_id=getattr(tokenizer, "bos_token_id", 0),
        eos_token_id=getattr(tokenizer, "eos_token_id", 1),
        tie_word_embeddings=False,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_layer_freq=getattr(config, "MOE_LAYER_FREQ", 2),
        first_k_dense_replace=getattr(config, "FIRST_K_DENSE_REPLACE", 0),
        torch_dtype=(
            torch.bfloat16
            if getattr(config, "BF16", False)
            else (torch.float16 if getattr(config, "FP16", False) else torch.float32)
        ),
    )

    model = DeepseekV3ForCausalLM(model_config)
    model.resize_token_embeddings(getattr(tokenizer, "vocab_size", 32000))

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
    )

    dataset = dataset.skip(config.SKIP_SAMPLES).take(config.SHARD_SIZE)
    return dataset


def load_full_dataset():
    dataset = load_dataset(
        config.INPUT_DATASET,
        split="train",
        streaming=True,
    )
    return dataset


def format_prompts(examples, tokenizer, isinst):
    def process_text(text):
        if not text or len(text.strip()) == 0:
            return None

        if isinst:
            conversation = []
            parts = text.split("<|end|>")
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    prompt = parts[i].replace("<|user|>", "").strip()
                    response = parts[i + 1].replace("<|bot|>", "").strip()
                    conversation.append({"role": "user", "content": prompt})
                    conversation.append({"role": "assistant", "content": response})

            if conversation:
                return tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            return tokenizer.bos_token + text + tokenizer.eos_token

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
    return AutoTokenizer.from_pretrained(
        config.INPUT_REPO + "-it"
        if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
        else config.INPUT_REPO
    )


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
    if config.INSTRUCT_FINETUNE_BOOL or config.INIT > 0:
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

        with tqdm(desc="Processing examples") as pbar:
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
