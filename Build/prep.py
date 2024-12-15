from tqdm import tqdm
import torch

from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
import torch.distributed as dist

from config import Config
from util import *

config = Config()

def load_model(tokenizer):
    if dist.is_initialized():
        dist.barrier()

    try:
        model_path = config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0 else config.OUTPUT_REPO
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

    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.FACTOR,
        intermediate_size=config.FACTOR * 4,
        num_hidden_layers=config.FACTOR // 2 ** 5,
        num_attention_heads=config.FACTOR // 2 ** 4,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False
    )

    try:
        model = LlamaForCausalLM(model_config)
    except Exception as e:
        print(f"Failed to create model: {e}")
        raise

    return model

def load_data():
    dataset = load_dataset(
        config.INSTRUCT_DATASET if config.INSTRUCT_FINETUNE_BOOL else config.INPUT_DATASET,
        split="train",
        streaming=True
    )

    shard_data = list(tqdm(dataset.skip(config.INIT * config.SHARD_SIZE).take(config.SHARD_SIZE), total=config.SHARD_SIZE, desc="Creating shard"))
    print(f'Shard set loaded with size {len(shard_data)}, realizing shard data..')
    shard = Dataset.from_list(shard_data)

    return shard

def format_prompts(examples, tokenizer, isinst):
    texts = []
    for text in examples['text']:
        if text and len(text.strip()) > 0:
            if isinst:
                conversation = []
                parts = text.split('<|end|>')
                for i in range(0, len(parts) - 1, 2):
                    prompt = parts[i].replace("<|user|>", "").strip()
                    response = parts[i + 1].replace("<|bot|>", "").strip()
                    conversation.append({"role": "user", "content": prompt})
                    conversation.append({"role": "assistant", "content": response})
                formatted_conversation = tokenizer.apply_chat_template(conversation, tokenize=False)

                texts.append(formatted_conversation)
            else:
                texts.append(tokenizer.bos_token +text + tokenizer.eos_token)
        else:
            print('Found empty entry in examples. Moving on..')
            continue

    if len(texts) == 0:
        raise ValueError("No valid texts found in examples for formatting.")

    return {'text': texts}

def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=config.VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens
    )
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    return fast_tokenizer

def load_tokenizer():
    return AutoTokenizer.from_pretrained(config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0 else config.OUTPUT_REPO)

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
        "additional_special_tokens": []
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
    print("Loading Data..")
    dataset = load_data()
    print("Loaded data.")

    print("Making Corpus..")
    training_corpus = get_training_corpus(dataset)
    print("Made Corpus.")

    print("Getting Tokenizer..")
    tokenizer = load_tokenizer() if config.INSTRUCT_FINETUNE_BOOL or config.INIT > 0 else create_tokenizer(training_corpus)
    print(f"Got Tokenizer with size {len(tokenizer)}.")

    print("Adding Special Tokens..")
    configure_tokenizer(tokenizer)
    print("Added Tokens.")

    print("Mapping Data..")
    dataset = dataset.map(lambda examples: format_prompts(examples, tokenizer, config.INSTRUCT_FINETUNE_BOOL), batched=True, remove_columns=dataset.column_names)
    print("Mapped Data.")

    print("Getting Model..")
    try:
        model = load_model(tokenizer) if is_inst or config.INIT > 0 else create_model(tokenizer)
        print("Got Model.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise

    print("Saving Prepared Data..")
    dataset.save_to_disk("/tmp/prepared_dataset")
    print("Saving pretrained tokenizer..")
    tokenizer.save_pretrained("/tmp/prepared_tokenizer")
    print("Saving model..")
    model.save_pretrained("/tmp/prepared_model")
    print("Prepared data saved.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        Space().stop(e)
