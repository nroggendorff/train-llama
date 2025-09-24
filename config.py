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

        epochs = float(os.environ.get("EPOCHS", 3))
        self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
        self.INIT = int(os.environ.get("INIT", 0))
        self.EPOCHS = epochs if self.INIT >= 2 else epochs / 2
        self.LEARNING_RATE = 3e-4
        self.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 2048))
        self.VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 52000))
        self.FP16 = True
        self.WEIGHT_DECAY = 1e-2
        self.GRADIENT_ACCUMULATION_STEPS = 8
        self.INPUT_DATASET = os.environ.get("INPUT_DS", "nroggendorff/micropus")
        self.INSTRUCT_DATASET = os.environ.get("INST_DS", "nroggendorff/elephant")
        self.SHARD_SIZE = int(os.environ.get("SHARD_SIZE", 131072))
        self.SHARD_INDEX = 0 if self.INIT < 2 else (self.INIT - 1)
        self.SKIP_SAMPLES = self.SHARD_INDEX * self.SHARD_SIZE
        self.OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "nroggendorff/smallama")
        self.PUSH_TO_HUB = True
        self.INSTRUCT_FINETUNE_BOOL = os.environ.get("INST", "false").lower() == "true"
        self.FACTOR = int(os.environ.get("FACTOR", 12288))
        self.TOTAL_STEPS = (self.SHARD_SIZE * self.EPOCHS) // (
            self.BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS
        )
        self.WARMUP_STEPS = int(self.TOTAL_STEPS * 0.1)
        self.SEED = 42

        Config._initialized = True

    def getDeepSpeedConfig(self):
        return {
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 2e6,
                "stage3_prefetch_bucket_size": 2e6,
                "stage3_param_persistence_threshold": 1e3,
                "stage3_max_live_parameters": 1e6,
                "stage3_max_reuse_distance": 1e6,
                "stage3_gather_16bit_weights_on_model_save": True,
                "sub_group_size": 1e6,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e6,
                "reduce_scatter": True,
                "stage3_use_all_reduce_for_fetch_params": True,
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
            "comms_logger": {
                "enabled": False,
                "verbose": False,
                "prof_all": False,
                "debug": False
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
            save_strategy="steps",
            logging_steps=max(self.BATCH_SIZE, int(self.WARMUP_STEPS)),
            save_total_limit=1,
            report_to="none",
            deepspeed=self.getDeepSpeedConfig(),
            use_liger_kernel=True,
            max_length=self.MAX_LENGTH,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            lr_scheduler_type="cosine",
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            dataloader_persistent_workers=False,
            prediction_loss_only=True,
            save_safetensors=True,
            ddp_timeout=1800,
        )
