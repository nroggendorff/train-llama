import gc

import numpy as np
import requests as rq
import torch

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, TrainingArguments
from datasets import load_dataset

from tokenizers import ByteLevelBPETokenizer
import trl

dataset = load_dataset("nroggendorff/openhermes", split="train").select(range(int(4e+4)))

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

training_corpus = get_training_corpus()

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    training_corpus,
    vocab_size=3200,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|user|>", "<|bot|>", "<|end|>"]
)

tokenizer.save("/tmp/custom_tokenizer.json")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/tmp/custom_tokenizer.json")

tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.unk_token = "<unk>"
tokenizer.pad_token = "<pad>"
tokenizer.mask_token = "<mask>"

tokenizer.additional_special_tokens = ["<|user|>", "<|bot|>", "<|end|>"]

tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

chat_template = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|bot|>\n' + message['content'] + '<|end|>\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{{ eos_token }}"

tokenizer.chat_template = chat_template

tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"]
})

tokenizer.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
tokenizer.assistant_token_id = tokenizer.convert_tokens_to_ids("<|bot|>")

tokenizer.save_pretrained("/tmp/llama-tokenizer")

tokenizer = AutoTokenizer.from_pretrained("/tmp/llama-tokenizer")
print(tokenizer.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}, {"role": "assistant", "content": "Due to rayleigh scattering."}, {"role": "user", "content": "That's cool."}, {"role": "assistant", "content": "Yeah, I agree."}], tokenize=False))

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False,
)

model = LlamaForCausalLM(config)

def format_prompts(examples):
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
    output = {}
    output['text'] = texts
    return output

dataset = dataset.map(format_prompts, batched=True)

print(dataset['text'][2])

args = TrainingArguments(
    output_dir="mayo",
    num_train_epochs=32,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    fp16=True,
    optim="sgd",
    optim_target_modules=["attn", "mlp"]
)

trainer = trl.SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=512
)

torch.cuda.set_device(0)

gc.collect()
torch.cuda.empty_cache()

trainer.train()

#trainer.push_to_hub()
trained_model = trainer.model
trained_tokenizer = trainer.tokenizer

repo_id = "makeshift-mayo"
trained_model.push_to_hub(repo_id)
trained_tokenizer.push_to_hub(repo_id)

raise RuntimeError("The script is finished.")