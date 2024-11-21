import os
from sys import exit
import torch
import trl
from transformers import (
    AutoTokenizer, LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM,
    TrainingArguments, PreTrainedTokenizerFast, AdamW, get_cosine_schedule_with_warmup
)
from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from itertools import islice

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
FACTOR = 12 ** 3 // 3
MAX_SEQ_LENGTH = 512
VOCAB_SIZE = 32000
INPUT_DATASET = "HuggingFaceTB/smollm-corpus"
INSTRUCT_DATASET = "nroggendorff/elephant"
OUTPUT_REPO = "nroggendorff/smallama"
INSTRUCT_FINETUNE_BOOL = False
INIT = 0
SHARD_SIZE = int(2e+5)
FP16 = True
WEIGHT_DECAY = 1e-3
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // 4
WARMUP_STEPS = ((SHARD_SIZE // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS) // 10
PUSH_TO_HUB = True

total_steps = WARMUP_STEPS * 10

class Space:
    def __init__(self):
        self.api = HfApi()
        self.pause = lambda: self.api.pause_space("nroggendorff/train-llama")

space = Space()

class FineError(Exception):
    def __init__(self, message="Script execution has completed."):
        self.message = message
        super().__init__(self.message)

def load_data():
    if not INSTRUCT_FINETUNE_BOOL:
        dataset = load_dataset(INPUT_DATASET, "cosmopedia-v2", split="train", streaming=True)
    else:
        dataset = load_dataset(INSTRUCT_DATASET, split="train", streaming=True)

    start = INIT * SHARD_SIZE
    data_list = list(islice(dataset, start, start + SHARD_SIZE))
    
    dataset = Dataset.from_dict({'text': [example['text'] for example in data_list]})
    return dataset

def encode_decode(texts, tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    tokenized_texts = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    ).input_ids

    if tokenized_texts.dim() >= 1:
        decoded_texts = tok.batch_decode(tokenized_texts)
    else:
        print('Found invalid entry in examples. Returning dummy..')
        decoded_texts = [tokenizer.pad_token * MAX_SEQ_LENGTH]
    
    islist = not len(decoded_texts) == 1
    
    return decoded_texts if islist else decoded_texts[0]

def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer.train_from_iterator(
        training_corpus,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens
    )
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)
    return fast_tokenizer

def load_tokenizer():
    return AutoTokenizer.from_pretrained(OUTPUT_REPO + '-it' if INSTRUCT_FINETUNE_BOOL else OUTPUT_REPO)

def get_training_corpus(dataset):
    for i in range(0, len(dataset['text']), 1000):
        yield dataset['text'][i : i + 1000]

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
                coded_text = tokenizer.code(formatted_conversation)
                texts.append(coded_text)
            else:
                texts.append(tokenizer.bos_token + tokenizer.code(text) + tokenizer.eos_token)
        else:
            print('Found empty entry in examples. Moving on..')
            continue

    if len(texts) == 0:
        raise ValueError("No valid texts found in examples for formatting.")

    coded_texts = tokenizer.code(texts)
    return {'text': coded_texts}

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
    return AutoModelForCausalLM.from_pretrained(OUTPUT_REPO + '-it' if INSTRUCT_FINETUNE_BOOL else OUTPUT_REPO)

def configure_tokenizer(tokenizer):
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "additional_special_tokens": []
    }
    if INSTRUCT_FINETUNE_BOOL:
        special_tokens["additional_special_tokens"] = ["<|user|>", "<|bot|>", "<|end|>"]
    tokenizer.add_special_tokens(special_tokens)

    if INSTRUCT_FINETUNE_BOOL:
        tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")
    
        chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        tokenizer.chat_template = chat_template

    tokenizer.code = lambda example: encode_decode(example, tokenizer)

def update_tokenizer(tokenizer, dataset, batch_size=1000):
    existing_vocab = tokenizer.get_vocab()
    oov_tokens = set()
    
    for i in range(0, len(dataset['text']), batch_size):
        batch = dataset['text'][i:i + batch_size]
        
        for text in batch:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            for token_id in token_ids:
                token = tokenizer.decode([token_id])
                if token.strip() and token not in existing_vocab:
                    oov_tokens.add(token)
    
    if oov_tokens:
        num_added = tokenizer.add_tokens(list(oov_tokens))
        return num_added
    
    return 0

def train_model(model, tokenizer, dataset, push, isinst):
    args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        per_gpu_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=FP16,
        save_steps=WARMUP_STEPS * 5,
        logging_steps=WARMUP_STEPS,
        eval_strategy="no",
        # eval_steps=WARMUP_STEPS,
        save_total_limit=2,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    dataset = dataset.map(lambda examples: format_prompts(examples, tokenizer, isinst), batched=True, remove_columns=dataset.column_names)

    if 'text' not in dataset.column_names:
        raise ValueError("Dataset transformation failed: 'text' column missing after mapping.")
    
    print("Mapped dataset sample length:", len(dataset[0]['text']))

    try:
        test_input = tokenizer(
            ["This is a test input."],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        test_output = model(**test_input)
        print("Model test output shape:", test_output.logits.shape)
    except RuntimeError as e:
        print(f"Error processing test batch: {e}")
    
    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        # dataset_text_field='text',
        max_seq_length=MAX_SEQ_LENGTH,
        optimizers=(optimizer, scheduler)
    )
    
    train = trainer.train()
    
    trained_model = trainer.model
    trained_tokenizer = trainer.tokenizer
    
    if push:
        repo_id = OUTPUT_REPO + "-it" if INSTRUCT_FINETUNE_BOOL else OUTPUT_REPO
        msg = f"Training loss: {train.training_loss:.4f}"
        trained_model.push_to_hub(repo_id, commit_message=msg, force=True)
        trained_tokenizer.push_to_hub(repo_id, commit_message=msg, force=True)
    else:
        trained_model.save_pretrained("model")
        trained_tokenizer.save_pretrained("tokenizer")

def main(push_to_hub=True, is_inst_finetune=False):
    print("Loading Data..")
    dataset = load_data()
    print("Loaded data.")
    
    if is_inst_finetune and INIT > 0:
        print("Loading Tokenizer..")
        tokenizer = load_tokenizer()
        print("Loaded Tokenizer.")
    else:
        print("Making Corpus..")
        training_corpus = get_training_corpus(dataset)
        print("Made Corpus.")

        print("Making Tokenizer..")
        tokenizer = create_tokenizer(training_corpus)
        print(f"Made Tokenizer with size {len(tokenizer)}.")

        # print("Adding Tokens..")
        # num_new_tokens = update_tokenizer(tokenizer, dataset)
        # print(f"Added {num_new_tokens} new tokens to the vocabulary")

    if INIT == 0:
        print("Adding Special Tokens..")
        configure_tokenizer(tokenizer)
        print("Added Tokens.")
    
    if is_inst_finetune or INIT > 0:
        print("Loading Model..")
        model = load_model()
        print("Loaded Model.")
    else:
        print("Creating Model..")
        model = create_model(tokenizer)
        print("Created Model.")

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    print("Resizing Token Embeddings..")
    try:
        model.resize_token_embeddings(len(tokenizer))
    except RuntimeError as e:
        raise RuntimeError(f"Error resizing token embeddings: {e}")
    print("Resized Embeddings.")

    print("Training Model..")
    train_model(model, tokenizer, dataset, push_to_hub, is_inst_finetune)
    raise FineError("Trained Model.")

if __name__ == "__main__":
    try:
        main(PUSH_TO_HUB, INSTRUCT_FINETUNE_BOOL)
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        space.pause()