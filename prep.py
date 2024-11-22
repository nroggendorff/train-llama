import os
from itertools import islice
from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from config import Config

config = Config()

def load_data():
    if not config.INSTRUCT_FINETUNE_BOOL:
        dataset = load_dataset(config.INPUT_DATASET, "cosmopedia-v2", split="train", streaming=True)
    else:
        dataset = load_dataset(config.INSTRUCT_DATASET, split="train", streaming=True)

    start = config.INIT * config.SHARD_SIZE
    data_list = list(islice(dataset, start, start + config.SHARD_SIZE))
    
    dataset = Dataset.from_dict({'text': [example['text'] for example in data_list]})
    return dataset

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
    return AutoTokenizer.from_pretrained(config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL else config.OUTPUT_REPO)

def get_training_corpus(dataset):
    for i in range(0, len(dataset['text']), 1000):
        yield dataset['text'][i : i + 1000]

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

def save_prepared_data(dataset, tokenizer):
    dataset.save_to_disk("prepared_dataset")
    tokenizer.save_pretrained("prepared_tokenizer")

def main():
    print("Loading Data..")
    dataset = load_data()
    print("Loaded data.")
    
    print("Making Corpus..")
    training_corpus = get_training_corpus(dataset)
    print("Made Corpus.")

    print("Making Tokenizer..")
    tokenizer = create_tokenizer(training_corpus)
    print(f"Made Tokenizer with size {len(tokenizer)}.")

    print("Adding Special Tokens..")
    configure_tokenizer(tokenizer)
    print("Added Tokens.")

    print("Saving Prepared Data..")
    save_prepared_data(dataset, tokenizer)
    print("Prepared data saved.")

if __name__ == "__main__":
    main()