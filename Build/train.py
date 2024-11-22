import torch
from transformers import (
    AdamW, get_cosine_schedule_with_warmup, AutoTokenizer, LlamaForCausalLM, LlamaConfig
)
from trl import SFTTrainer
from datasets import load_from_disk
from config import Config
from util import *

config = Config()

def load_model(tokenizer):
    model = LlamaForCausalLM.from_pretrained(config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL and config.INIT > 0 else config.OUTPUT_REPO)
    model.resize_token_embeddings(len(tokenizer))
    return model

def create_model(tokenizer):
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
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(model_config)

def train_model(model, tokenizer, dataset, push):
    args = config.getConfig()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=config.TOTAL_STEPS
    )

    try:
        test_input = tokenizer(
            ["This is a test input."], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=config.MAX_SEQ_LENGTH
        )
        test_output = model(**test_input)
        print("Model test output shape:", test_output.logits.shape)
    except RuntimeError as e:
        print(f"Error processing test batch: {e}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        max_seq_length=config.MAX_SEQ_LENGTH,
        optimizers=(optimizer, scheduler)
    )
    
    train = trainer.train()
    
    if push:
        repo_id = config.OUTPUT_REPO + "-it" if config.INSTRUCT_FINETUNE_BOOL else config.OUTPUT_REPO
        msg = f"Training loss: {train.training_loss:.4f}"
        trainer.model.push_to_hub(repo_id, commit_message=msg, force=True)
        trainer.tokenizer.push_to_hub(repo_id, commit_message=msg, force=True)
    else:
        trainer.model.save_pretrained("trained_model")
        trainer.tokenizer.save_pretrained("trained_tokenizer")

def main(push_to_hub=True, is_inst=config.INSTRUCT_FINETUNE_BOOL):
    print("Loading Prepared Data..")
    dataset = load_from_disk("prepared_dataset")
    tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")
    print("Loaded Prepared Data.")

    print("Getting Model..")
    model = load_model(tokenizer) if is_inst or config.INIT > 0 else create_model(tokenizer)
    print("Got Model.")

    print("Training Model..")
    train_model(model, tokenizer, dataset, push_to_hub)
    raise Conclusion("Trained Model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        Space().pause()