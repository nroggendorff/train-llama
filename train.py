import os

import torch
import trl

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, TrainingArguments, PreTrainedTokenizerFast, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 192
EPOCHS = 30
LEARNING_RATE = 2e-2
FACTOR = 64
VOCAB_SIZE = 32000
INPUT_DATASET = "nroggendorff/openhermes"
OUTPUT_REPO = "smallama"
FP16 = True
WARMUP_STEPS = 200
DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
CLIPPING = 1.0
PUSH_TO_HUB = True

def load_data():
    dataset = load_dataset(INPUT_DATASET, split="train").select(range(int(2e+3)))
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
        intermediate_size=FACTOR * 4,
        num_hidden_layers=max(1, FACTOR // 32),
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

def train_model(model, tokenizer, dataset, push):
    args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch",
        warmup_steps=WARMUP_STEPS,
        weight_decay=DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=FP16,
        max_grad_norm=CLIPPING
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=(len(dataset) // args.per_device_train_batch_size) * args.num_train_epochs
    )
    
    dataset = dataset.map(lambda examples: format_prompts(examples, tokenizer), batched=True)
    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        dataset_text_field='text',
        max_seq_length=MAX_SEQ_LENGTH,
        optimizers=(optimizer, scheduler)
    )
    
    trainer.train()
    
    trained_model = trainer.model
    trained_tokenizer = trainer.tokenizer
    
    if push:
        repo_id = OUTPUT_REPO
        trained_model.push_to_hub(repo_id)
        trained_tokenizer.push_to_hub(repo_id)
    else:
        trained_model.save_pretrained("model")
        trained_tokenizer.save_pretrained("tokenizer")

def main(push_to_hub=True):
    dataset = load_data()
    training_corpus = get_training_corpus(dataset)
    tokenizer = create_tokenizer(training_corpus)
    configure_tokenizer(tokenizer)
    model = create_model(tokenizer)
    train_model(model, tokenizer, dataset, push_to_hub)

if __name__ == "__main__":
    main(PUSH_TO_HUB)
    raise RuntimeError("The script is finished.")