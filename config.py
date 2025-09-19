import os
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
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
        self.EPOCHS = float(os.environ.get("EPOCHS", 3))
        self.LEARNING_RATE = 3e-4
        self.MAX_LENGTH = 512
        self.VOCAB_SIZE = 52000
        self.FP16 = True
        self.WEIGHT_DECAY = 1e-2
        self.GRADIENT_ACCUMULATION_STEPS = 8
        self.INPUT_DATASET = os.environ.get("INPUT_DS", "nroggendorff/micropus")
        self.INSTRUCT_DATASET = os.environ.get("INST_DS", "nroggendorff/elephant")
        self.SHARD_SIZE = int(os.environ.get("SHARD_SIZE", 131072))
        self.OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "nroggendorff/smallama")
        self.PUSH_TO_HUB = True
        self.INSTRUCT_FINETUNE_BOOL = os.environ.get("INST", "false").lower() == "true"
        self.FACTOR = int(os.environ.get("FACTOR", 12288))
        self.TOTAL_STEPS = (self.SHARD_SIZE * self.EPOCHS) // (
            self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS
        )
        self.WARMUP_STEPS = int(self.TOTAL_STEPS * 0.1)
        self.INIT = int(os.environ.get("INIT", 0))
        self.SEED = 42

        Config._initialized = True

    def getDeepSpeedConfig(self):
        return {
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
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
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
            "communication_data_type": "fp16",
            "aio": {
                "block_size": 1048576,
                "queue_depth": 8,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True,
            },
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
            logging_steps=max(self.BATCH_SIZE, int(self.WARMUP_STEPS)),
            save_total_limit=2,
            report_to="none",
            deepspeed=self.getDeepSpeedConfig(),
            use_liger_kernel=True,
            max_length=self.MAX_LENGTH,
            gradient_checkpointing=True,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            lr_scheduler_type="cosine",
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
        )
