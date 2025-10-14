import os
import re
import math
from trl import SFTConfig


class Config:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._initialized:
            return

        epochs = float(os.environ.get("EPOCHS", 3))
        lr = float(os.environ.get("LEARNING_RATE", 3e-4))
        space_timeout = os.environ.get("STARTUP_DURATION_TIMEOUT", "350m")
        int_space_timeout = int(re.sub(r"\D", "", space_timeout))
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
        self.INIT = int(os.environ.get("INIT", 0))
        self.INSTRUCT_FINETUNE_BOOL = os.environ.get("INST", "false").lower() == "true"
        self.EPOCHS = (
            epochs
            if self.INIT > 1 and self.INSTRUCT_FINETUNE_BOOL == True
            else epochs / 2
        )
        self.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 2048))
        self.VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 52000))
        self.FP16 = True
        self.WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-2))
        self.GRADIENT_ACCUMULATION_STEPS = int(
            os.environ.get("GRADIENT_ACCUMULATION_STEPS", 2)
        )
        self.INPUT_DATASET = os.environ.get("INPUT_DS", "nroggendorff/microrpus")
        self.INSTRUCT_DATASET = os.environ.get("INST_DS", "nroggendorff/elephant")
        self.SHARD_SIZE = int(os.environ.get("SHARD_SIZE", 131072))
        if self.INIT == 0:
            self.SHARD_INDEX = 0
            self.SKIP_SAMPLES = 0
            self.EPOCHS = epochs / 2
        elif self.INIT == 1:
            self.SHARD_INDEX = 0
            self.SKIP_SAMPLES = 0
            self.EPOCHS = epochs / 2
        else:
            self.SHARD_INDEX = self.INIT - 1 if self.INSTRUCT_FINETUNE_BOOL else 0
            self.SKIP_SAMPLES = self.SHARD_INDEX * self.SHARD_SIZE
            self.EPOCHS = epochs
        self.LEARNING_RATE = (
            lr
            / math.sqrt(self.INIT + 1)
            ** (
                ((self.SHARD_SIZE / self.EPOCHS) * (self.INIT + self.EPOCHS))
                / (self.SHARD_SIZE + self.EPOCHS)
            )
            if self.INIT > 1
            else lr
        )
        self.OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "nroggendorff/smallama")
        self.INPUT_REPO = os.environ.get("INPUT_REPO", self.OUTPUT_REPO)
        self.FACTOR = int(os.environ.get("FACTOR", 12288))
        self.TOTAL_STEPS = (self.SHARD_SIZE * self.EPOCHS) // (
            self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS
        )
        self.WARMUP_STEPS = int(self.TOTAL_STEPS * 0.1)
        self.SPACE_TIMEOUT = (
            int_space_timeout
            if space_timeout.endswith("m")
            else (
                int_space_timeout * 60
                if space_timeout.endswith("h")
                else (
                    print(
                        "Invalid STARTUP_DURATION_TIMEOUT format.",
                        "Use 'm' for minutes or 'h' for hours.",
                        "Falling back to 30 minutes.",
                    )
                    or 30
                )
            )
        )
        self.TIMEOUT_BUFFER = int(os.environ.get("TIMEOUT_BUFFER", "20"))
        self.TIMEOUT = self.SPACE_TIMEOUT - self.TIMEOUT_BUFFER
        self.MAX_RETRIES = 5
        self.SEED = 42

        Config._initialized = True

    def getDeepSpeedConfig(self):
        return {
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
            },
            "fp16": {
                "enabled": self.FP16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "gradient_clipping": 1.0,
            "train_micro_batch_size_per_gpu": self.BATCH_SIZE,
            "gradient_accumulation_steps": self.GRADIENT_ACCUMULATION_STEPS,
            "wall_clock_breakdown": False,
            "communication_data_type": "fp16",
            "aio": {
                "block_size": 262144,
                "queue_depth": 4,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            },
            "zero_force_ds_cpu_optimizer": False,
        }

    def getConfig(self):
        return SFTConfig(
            output_dir="model",
            num_train_epochs=self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            learning_rate=self.LEARNING_RATE,
            warmup_steps=self.WARMUP_STEPS,
            weight_decay=self.WEIGHT_DECAY,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            fp16=self.FP16,
            save_steps=max(1, int(self.WARMUP_STEPS * 5)),
            save_strategy="steps",
            logging_steps=max(self.BATCH_SIZE, int(self.WARMUP_STEPS)),
            save_total_limit=1,
            report_to="none",
            deepspeed=self.getDeepSpeedConfig(),
            use_liger_kernel=True,
            max_length=self.MAX_LENGTH,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            lr_scheduler_type="cosine",
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            dataloader_persistent_workers=False,
            prediction_loss_only=True,
            save_safetensors=True,
        )
