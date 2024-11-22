import os
from sys import exit
import torch
import trl
from transformers import (
    AutoTokenizer, LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM,
    PreTrainedTokenizerFast, AdamW, get_cosine_schedule_with_warmup
)
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset
from tokenizers import ByteLevelBPETokenizer
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from itertools import islice
from typing import Optional
from logging import getLogger, StreamHandler, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)
handler = StreamHandler()
logger.addHandler(handler)

class Config:
    def __init__(self):
        # Model and training hyperparameters
        self.BATCH_SIZE = 16
        self.EPOCHS = 3
        self.LEARNING_RATE = 2e-4
        self.MAX_SEQ_LENGTH = 512
        self.VOCAB_SIZE = 32000
        self.FP16 = True
        self.WEIGHT_DECAY = 1e-3
        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // 4
    
        # Dataset configurations
        self.INPUT_DATASET = "HuggingFaceTB/smollm-corpus"
        self.INSTRUCT_DATASET = "nroggendorff/elephant"
        self.SHARD_SIZE = int(2e+5)
    
        # Output and repo settings
        self.OUTPUT_REPO = "nroggendorff/smallama"
        self.PUSH_TO_HUB = True
        self.INSTRUCT_FINETUNE_BOOL = False
    
        # Training steps and warmup
        self.FACTOR = 12 ** 3 // 3
        self.TOTAL_STEPS = (self.SHARD_SIZE * self.EPOCHS) // (self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS)
        self.WARMUP_STEPS = int(self.TOTAL_STEPS * 0.1)
    
        # Initial state for shard offset
        self.INIT = 0

        # ignore
        self.getConfig = lambda: self._args()

    # @staticmethod
    def _args(self):
        return SFTConfig(
            output_dir="model",
            num_train_epochs=self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            learning_rate=self.LEARNING_RATE,
            warmup_steps=self.WARMUP_STEPS,
            weight_decay=self.WEIGHT_DECAY,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            fp16=self.FP16,
            save_steps=int(self.WARMUP_STEPS * 5),
            logging_steps=int(self.WARMUP_STEPS),
            save_total_limit=2,
            report_to="none",
        )

config = Config().getConfig()

class Space:
    def __init__(self):
        self.api = HfApi()
        self.pause = lambda: self.api.pause_space("nroggendorff/train-llama")

space = Space()

class FineError(Exception):
    def __init__(self, message="Training completed successfully."):
        self.message = message
        super().__init__(self.message)

def load_data(dataset_name: str, split: str, shard_size: int, init_offset: int = 0) -> Dataset:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    shard_start = init_offset * shard_size
    data_list = list(islice(dataset, shard_start, shard_start + shard_size))
    return Dataset.from_dict({'text': [example.get('text', '') for example in data_list]})

def encode_decode(texts, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_texts = tokenizer(
        texts, padding="max_length", truncation=True, max_length=config.MAX_SEQ_LENGTH, return_tensors="pt"
    ).input_ids
    return tokenizer.batch_decode(tokenized_texts) if tokenized_texts.dim() >= 1 else [tokenizer.pad_token * config.MAX_SEQ_LENGTH]

def create_tokenizer(training_corpus):
    tokenizer = ByteLevelBPETokenizer()
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokenizer.train_from_iterator(training_corpus, vocab_size=config.VOCAB_SIZE, min_frequency=2, special_tokens=special_tokens)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer._tokenizer)

def load_tokenizer(repo: str):
    return AutoTokenizer.from_pretrained(repo)

def get_training_corpus(dataset):
    for i in range(0, len(dataset['text']), 1000):
        yield dataset['text'][i : i + 1000]

def format_prompts(examples, tokenizer, is_instructional):
    texts = []
    for text in examples['text']:
        if text and len(text.strip()) > 0:
            if is_instructional:
                conversation = []
                parts = text.split('<|end|>')
                for i in range(0, len(parts) - 1, 2):
                    prompt = parts[i].replace("<|user|>", "").strip()
                    response = parts[i + 1].replace("<|bot|>", "").strip()
                    conversation.append({"role": "user", "content": prompt})
                    conversation.append({"role": "assistant", "content": response})
                coded_text = tokenizer.code(tokenizer.apply_chat_template(conversation, tokenize=False))
                texts.append(coded_text)
            else:
                texts.append(tokenizer.bos_token + tokenizer.code(text) + tokenizer.eos_token)
    if not texts:
        raise ValueError("No valid texts found in examples for formatting.")
    return {'text': tokenizer.code(texts)}

def create_model(tokenizer):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.FACTOR,
        intermediate_size=config.FACTOR * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)

def train_model(model, tokenizer, dataset, push_to_hub, is_instructional):
    config = 
    dataset = dataset.map(
        lambda examples: format_prompts(examples, tokenizer, is_instructional), 
        batched=True, 
        remove_columns=dataset.column_names
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset
    )
    train_result = trainer.train()

    if push_to_hub:
        repo_id = config.OUTPUT_REPO + "-it" if config.INSTRUCT_FINETUNE_BOOL else config.OUTPUT_REPO
        trainer.model.push_to_hub(repo_id, commit_message=f"Training loss: {train_result.training_loss:.4f}", force=True)
        trainer.tokenizer.push_to_hub(repo_id, commit_message=f"Training loss: {train_result.training_loss:.4f}", force=True)
    else:
        trainer.model.save_pretrained("model")
        trainer.tokenizer.save_pretrained("tokenizer")

def main():
    dataset = load_data(config.INPUT_DATASET, "train", config.SHARD_SIZE, config.INIT)
    tokenizer = (
        load_tokenizer(config.OUTPUT_REPO)
        if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0
        else create_tokenizer(get_training_corpus(dataset))
    )
    model = (
        load_model()
        if config.INSTRUCT_FINETUNE_BOOL or config.INIT > 0
        else create_model(tokenizer)
    )
    train_model(model, tokenizer, dataset, config.PUSH_TO_HUB, config.INSTRUCT_FINETUNE_BOOL)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")
        space.pause()