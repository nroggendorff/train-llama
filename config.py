import os
import json

from trl import SFTConfig

class Config:
    def __init__(self):
        config_data = self._load_json('config.json')

        self.BATCH_SIZE = int(os.environ.get('BATCH_SIZE', config_data.get("batch-size")))
        self.EPOCHS = config_data.get("epochs")
        self.LEARNING_RATE = config_data.get("learning-rate")
        self.MAX_SEQ_LENGTH = config_data.get("max-seq-length")
        self.VOCAB_SIZE = config_data.get("vocab-size")
        self.FP16 = config_data.get("fp16")
        self.WEIGHT_DECAY = config_data.get("weight-decay")
        self.GRADIENT_ACCUMULATION_STEPS = config_data.get("gradient-accumulation-steps")

        self.INPUT_DATASET = config_data.get("input-dataset")
        self.INSTRUCT_DATASET = config_data.get("instruct-dataset")
        self.SHARD_SIZE = int(os.environ.get('SHARD_SIZE', config_data.get("shard-size")))

        self.OUTPUT_REPO = config_data.get("output-repo")
        self.PUSH_TO_HUB = config_data.get("push-to-hub")
        self.INSTRUCT_FINETUNE_BOOL = os.environ.get('INSTRUCT', 'false').lower() == 'true'

        self.FACTOR = int(os.environ.get('FACTOR', config_data.get("factor")))
        self.TOTAL_STEPS = (self.SHARD_SIZE * self.EPOCHS) // (self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS)
        self.WARMUP_STEPS = int(self.TOTAL_STEPS * 0.1)

        self.INIT = int(os.environ.get('INIT', 0))
        self.SEED = 42

    @staticmethod
    def _load_json(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    class _AutoDict(dict):
        def __init__(self, config):
            self.config = config

        def __getitem__(self, key):
            cuda_settings = {
                "zero_optimization": {"stage": 2},
                "fp16": {"enabled": self.config.FP16},
                "accelerator": {"type": "cuda"}
            }
            return cuda_settings.get(key, self.get(key, "auto"))

    def getConfig(self):
        ds_config = self._AutoDict(self)
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
            logging_steps=max(self.BATCH_SIZE, int(self.WARMUP_STEPS)),
            save_total_limit=2,
            report_to="none",
            deepspeed=ds_config,
            use_liger=True,
            max_seq_length=self.MAX_SEQ_LENGTH
        )
