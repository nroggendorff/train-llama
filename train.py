import torch
from transformers import AutoModelForCausalLM, AdamW, get_cosine_schedule_with_warmup
from trl import SFTTrainer
from datasets import load_from_disk
from config import Config

config = Config()

class FineError(Exception):
    def __init__(self, message="Script execution has completed."):
        self.message = message
        super().__init__(self.message)

def load_model(tokenizer):
    model = AutoModelForCausalLM.from_pretrained(config.OUTPUT_REPO + '-it' if config.INSTRUCT_FINETUNE_BOOL else config.OUTPUT_REPO)
    model.resize_token_embeddings(len(tokenizer))
    return model

def train_model(model, tokenizer, dataset, push):
    args = config.getConfig()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=args.num_training_steps
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
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

def main(push_to_hub=True):
    print("Loading Prepared Data..")
    dataset = load_from_disk("prepared_dataset")
    tokenizer = AutoTokenizer.from_pretrained("prepared_tokenizer")
    print("Loaded Prepared Data.")

    print("Loading Model..")
    model = load_model(tokenizer)
    print("Loaded Model.")

    print("Training Model..")
    train_model(model, tokenizer, dataset, push_to_hub)
    raise FineError("Trained Model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f'{type(e).__name__}: {e}')