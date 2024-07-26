import os

import torch
import trl

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, TrainingArguments, PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 256
EPOCHS = 1
LEARNING_RATE = 1e-4
FP16 = True
FACTOR = 1
VOCAB_SIZE = 3200
INPUT_DATASET = "nroggendorff/elephant"
OUTPUT_REPO = "smallama"

def load_data():
    dataset = load_dataset(INPUT_DATASET, split="train")
    return dataset

def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|user|>", "<|bot|>", "<|end|>"]
    )

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    return fast_tokenizer

def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

def format_prompts(examples, tokenizer):
    texts = []
    for text in examples['text']:
        conversation = []
        parts = text.split('<|end|>')
        for i in range(0, len(parts) - 1, 2):
            prompt = parts[i].replace("<|user|>", "")
            response = parts[i + 1].replace("<|bot|>", "")
            conversation.append({"role": "user", "content": prompt})
            conversation.append({"role": "assistant", "content": response})
        formatted_conversation = tokenizer.apply_chat_template(conversation, tokenize=False)
        texts.append(formatted_conversation)
    return {"text": texts}

def create_model(tokenizer):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=FACTOR,
        intermediate_size=FACTOR * 2,
        num_hidden_layers=max(1, FACTOR // 64),
        num_attention_heads=max(1, FACTOR // 64),
        max_position_embeddings=MAX_SEQ_LENGTH,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    
    model = LlamaForCausalLM(config)
    return model

def configure_tokenizer(tokenizer):
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
    tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")
    
    chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{{ eos_token }}"
    tokenizer.chat_template = chat_template

def train_model(model, tokenizer, dataset):
    args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        optim="sgd"
    )
    dataset = dataset.map(lambda examples: format_prompts(examples, tokenizer), batched=True)
    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        dataset_text_field='text',
        max_seq_length=MAX_SEQ_LENGTH
    )
    trainer.train()
    
    trained_model = trainer.model
    trained_tokenizer = trainer.tokenizer
    
    repo_id = OUTPUT_REPO
    trained_model.push_to_hub(repo_id)
    trained_tokenizer.push_to_hub(repo_id)

def main():
    dataset = load_data()
    training_corpus = get_training_corpus(dataset)
    tokenizer = create_tokenizer(training_corpus)
    configure_tokenizer(tokenizer)
    model = create_model(tokenizer)
    train_model(model, tokenizer, dataset)

if __name__ == "__main__":
    main()
    raise RuntimeError("The script is finished.")