from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
)
import torch.distributed as dist
from torch.utils.data import DataLoader

from config import Config
from util import *

config = Config()


def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = (
            config.OUTPUT_REPO + "-it"
            if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
            else config.OUTPUT_REPO
        )
        model = LlamaForCausalLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise

    if dist.is_initialized():
        dist.barrier()

    return model


def create_model(tokenizer):
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
        use_cache=True,
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
    )

    dataset = dataset.skip(config.SKIP_SAMPLES).take(config.SHARD_SIZE)

    dataloader = DataLoader(
        dataset,
        batch_size=1000,
        num_workers=8,
        pin_memory=True,
    )

    shard_data = []
    for batch in tqdm(dataloader, desc="Loading data with parallel workers"):
        shard_data.extend(batch["text"])

    print(f"Shard set loaded with size {len(shard_data)}, realizing shard data..")

    shard = Dataset.from_dict({"text": shard_data})
    return shard


def load_full_dataset():
    dataset = load_dataset(
        config.INPUT_DATASET,
        split="train",
        streaming=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1000,
        num_workers=8,
        pin_memory=True,
    )

    data = []
    for batch in tqdm(dataloader, desc="Loading data with parallel workers"):
        data.extend(batch["text"])

    print(f"Shard set loaded with size {len(data)}, realizing shard data..")

    dataset = Dataset.from_dict({"text": data})
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

    with ThreadPoolExecutor(max_workers=8) as executor:
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
        config.OUTPUT_REPO + "-it"
        if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
        else config.OUTPUT_REPO
    )


def get_training_corpus(dataset):
    buffer = []
    for i, text in enumerate(dataset):
        buffer.append(text["text"])
        if (i + 1) % 1000 == 0:
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

    print("Mapping Data..")
    dataset = dataset.map(
        lambda examples: format_prompts(
            examples, tokenizer, config.INSTRUCT_FINETUNE_BOOL
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print("Mapped Data.")

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
    dataset.save_to_disk("prepared_dataset")
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
