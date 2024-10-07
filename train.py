import os
import torch
import trl
from transformers import (
    AutoTokenizer, LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM,
    TrainingArguments, PreTrainedTokenizerFast, AdamW, get_cosine_schedule_with_warmup
)
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-4
FACTOR = 768
MAX_SEQ_LENGTH = 128
VOCAB_SIZE = 32000
INPUT_DATASET = "HuggingFaceTB/smollm-corpus"
INSTRUCT_DATASET = "nroggendorff/elephant"
OUTPUT_REPO = "nroggendorff/smallama"
INSTRUCT_FINETUNE_BOOL = False
INIT = 0
FP16 = True
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // 4
PUSH_TO_HUB = True
NUM_WORKERS = 4

def load_data():
    if not INSTRUCT_FINETUNE_BOOL:
        dataset = load_dataset(INPUT_DATASET, "cosmopedia-v2", split="train", streaming=True)
        dataset = custom_shard_stream(dataset)
    else:
        dataset = load_dataset(INSTRUCT_DATASET, split="train")
    return dataset

def custom_shard_stream(dataset, shard_size=5e5, shard_index=0):
    def shard_generator():
        count = 0
        for example in dataset:
            if count % shard_size == shard_index:
                yield example
            count += 1
    return shard_generator()

def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    if INSTRUCT_FINETUNE_BOOL:
        special_tokens.extend(["<|user|>", "<|bot|>", "<|end|>"])
    tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens
    )
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    return fast_tokenizer

def load_tokenizer():
    return AutoTokenizer.from_pretrained(OUTPUT_REPO)

def get_training_corpus(dataset):
    for example in dataset:
        yield example['text']

def format_prompts(examples, tokenizer, isinst):
    texts = []
    for text in examples['text']:
        if isinst:
            conversation = []
            parts = text.split('<|end|>')
            for i in range(0, len(parts) - 1, 2):
                prompt = parts[i].replace("<|user|>", "")
                response = parts[i + 1].replace("<|bot|>", "")
                conversation.append({"role": "user", "content": prompt})
                conversation.append({"role": "assistant", "content": response})
            formatted_conversation = tokenizer.apply_chat_template(conversation, tokenize=False)
            texts.append(formatted_conversation)
        else:
            texts.append(tokenizer.bos_token + text + tokenizer.eos_token)
    return {"text": texts}

def create_model(tokenizer):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=FACTOR,
        intermediate_size=FACTOR * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=MAX_SEQ_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)

def load_model():
    return AutoModelForCausalLM.from_pretrained(OUTPUT_REPO)

def configure_tokenizer(tokenizer):
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    }
    if INSTRUCT_FINETUNE_BOOL:
        special_tokens["additional_special_tokens"] = ["<|user|>", "<|bot|>", "<|end|>"]
    tokenizer.add_special_tokens(special_tokens)

    if INSTRUCT_FINETUNE_BOOL:
        tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")
    
        chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        tokenizer.chat_template = chat_template

def train_model(model, tokenizer, dataset, push, isinst):
    args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=FP16,
        save_steps=int(1e+10),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=args.num_train_epochs
    )

    dataset = dataset.map(lambda examples: format_prompts(examples, tokenizer, isinst), batched=True, remove_columns=dataset.column_names)

    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler),
        max_seq_length=MAX_SEQ_LENGTH
    )

    train_result = trainer.train()

    if push:
        repo_id = OUTPUT_REPO + "-it" if INSTRUCT_FINETUNE_BOOL else OUTPUT_REPO
        msg = f"Training loss: {train_result.training_loss:.4f}"
        trainer.model.push_to_hub(repo_id, commit_message=msg, force=True)
        trainer.tokenizer.push_to_hub(repo_id, commit_message=msg, force=True)
    else:
        trainer.model.save_pretrained("model")
        trainer.tokenizer.save_pretrained("tokenizer")

def main(push_to_hub=True, is_inst_finetune=False):
    dataset = load_data()
    if not is_inst_finetune and INIT == 0:
        training_corpus = get_training_corpus(dataset)
        tokenizer = create_tokenizer(training_corpus)
    else:
        tokenizer = load_tokenizer()
    
    configure_tokenizer(tokenizer)
    
    if is_inst_finetune:
        model = load_model()
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = create_model(tokenizer) if INIT == 0 else load_model()
    
    train_model(model, tokenizer, dataset, push_to_hub, is_inst_finetune)

if __name__ == "__main__":
    main(PUSH_TO_HUB, INSTRUCT_FINETUNE_BOOL)
    raise Exception("Done baking!")