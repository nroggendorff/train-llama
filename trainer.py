import os
import math
import time
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from typing import Dict, Any, Optional
import deepspeed
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from tqdm.auto import tqdm
import torch.distributed as dist


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_accumulation_steps: int
    fp16: bool
    logging_steps: int
    deepspeed: Dict[str, Any]
    use_liger_kernel: bool
    max_length: int
    gradient_checkpointing: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    remove_unused_columns: bool
    lr_scheduler_type: str
    adam_beta1: float
    adam_beta2: float
    max_grad_norm: float
    dataloader_persistent_workers: bool
    dataloader_prefetch_factor: Optional[int] = 2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Trainer:
    def __init__(
        self,
        model: LlamaForCausalLM,
        processing_class: PreTrainedTokenizerFast,
        args: TrainingConfig,
        train_dataset,
        callbacks=None,
    ):
        self.model = model
        self.processing_class = processing_class
        self.args = args
        self.train_dataset = train_dataset
        self.callbacks = callbacks or []

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.global_step = 0
        self.current_epoch = 0
        self.total_loss = 0.0

        self.model_engine = None
        self.optimizer = None
        self.scheduler = None

    def is_world_process_zero(self):
        return self.local_rank in [-1, 0]

    def _create_dataloader(self):
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True,
            seed=42,
            drop_last=True,
        )

        def tokenize_collate_fn(examples):
            texts = [example["text"] for example in examples]

            tokenized = self.processing_class(
                texts,
                truncation=True,
                max_length=self.args.max_length,
                padding="longest",
                pad_to_multiple_of=8,
                return_tensors="pt",
            )

            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        dataloader_kwargs = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": tokenize_collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
        }

        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                self.args.dataloader_persistent_workers
            )
            if (
                hasattr(self.args, "dataloader_prefetch_factor")
                and self.args.dataloader_prefetch_factor
            ):
                dataloader_kwargs["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )

        return DataLoader(self.train_dataset, **dataloader_kwargs)

    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            fused=True,
        )

    def _get_cosine_schedule_with_warmup(
        self, optimizer, num_warmup_steps, num_training_steps
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def _create_scheduler(self, num_training_steps):
        return self._get_cosine_schedule_with_warmup(
            self.optimizer,
            self.args.warmup_steps,
            num_training_steps,
        )

    def _initialize_deepspeed(self, num_training_steps):
        if self.args.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(num_training_steps)

        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=self.args.deepspeed,
            dist_init_required=False,
        )

        if self.is_world_process_zero():
            print(f"\nDeepSpeed Configuration:")
            print(f"  World size: {self.world_size}")
            print(f"  Local rank: {self.local_rank}")
            print(f"  FP16 enabled: {self.args.fp16}")
            print(
                f"  Zero stage: {self.args.deepspeed.get('zero_optimization', {}).get('stage', 'N/A')}"
            )
            print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
            print(
                f"  Effective batch size: {self.args.per_device_train_batch_size * self.world_size * self.args.gradient_accumulation_steps}"
            )

    def training_step(self, batch):
        self.model_engine.train()

        inputs = {k: v.to(self.model_engine.device) for k, v in batch.items()}

        outputs = self.model_engine(**inputs)
        loss = outputs.loss

        self.model_engine.backward(loss)

        loss_value = loss.item()
        self.total_loss += loss_value

        return loss_value

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.01 and value != 0:
                    formatted.append(f"{key}: {value:.2e}")
                else:
                    formatted.append(f"{key}: {value:.4f}")
            else:
                formatted.append(f"{key}: {value}")
        return " | ".join(formatted)

    def train(self):
        if self.is_world_process_zero():
            print("\n" + "=" * 70)
            print("Starting Training")
            print("=" * 70)

        train_dataloader = self._create_dataloader()

        num_update_steps_per_epoch = (
            len(train_dataloader) // self.args.gradient_accumulation_steps
        )
        num_training_steps = int(
            self.args.num_train_epochs * num_update_steps_per_epoch
        )

        self._initialize_deepspeed(num_training_steps)

        if self.is_world_process_zero():
            print(f"\nTraining Configuration:")
            print(f"  Num examples: {len(self.train_dataset)}")
            print(f"  Num epochs: {self.args.num_train_epochs}")
            print(f"  Batch size per device: {self.args.per_device_train_batch_size}")
            print(
                f"  Total batch size: {self.args.per_device_train_batch_size * self.world_size * self.args.gradient_accumulation_steps}"
            )
            print(
                f"  Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
            )
            print(f"  Total optimization steps: {num_training_steps}")
            print(f"  Warmup steps: {self.args.warmup_steps}")
            print(f"  Learning rate: {self.args.learning_rate}")
            print(f"  Dataloader workers: {self.args.dataloader_num_workers}")
            print(
                f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
            )
            print("=" * 70 + "\n")

        self.global_step = 0
        log_loss = 0.0
        log_steps = 0
        start_time = time.time()
        epoch_start_time = start_time
        last_log_time = start_time

        for epoch in range(int(self.args.num_train_epochs)):
            self.current_epoch = epoch

            train_dataloader.sampler.set_epoch(epoch)

            if self.is_world_process_zero():
                epoch_pbar = tqdm(
                    total=len(train_dataloader),
                    desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}",
                    position=0,
                    leave=True,
                    mininterval=1.0,
                )

            for step, batch in enumerate(train_dataloader):
                loss = self.training_step(batch)
                log_loss += loss
                log_steps += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.model_engine.step()
                    self.global_step += 1

                    if (
                        self.global_step % self.args.logging_steps == 0
                        and log_steps > 0
                    ):
                        avg_loss = log_loss / log_steps
                        current_lr = self.scheduler.get_last_lr()[0]
                        current_time = time.time()
                        log_elapsed = current_time - last_log_time

                        steps_per_second = (
                            self.args.logging_steps / log_elapsed
                            if log_elapsed > 0
                            else 0
                        )
                        samples_per_second = (
                            steps_per_second
                            * self.args.per_device_train_batch_size
                            * self.world_size
                            * self.args.gradient_accumulation_steps
                        )

                        if self.is_world_process_zero():
                            metrics = {
                                "loss": avg_loss,
                                "learning_rate": current_lr,
                                "epoch": epoch + (step / len(train_dataloader)),
                            }

                            epoch_pbar.set_postfix_str(self._format_metrics(metrics))

                            if self.global_step % (self.args.logging_steps * 5) == 0:
                                print(f"\n{'='*70}")
                                print(f"Step {self.global_step}/{num_training_steps}")
                                print(f"{self._format_metrics(metrics)}")
                                print(
                                    f"Speed: {samples_per_second:.2f} samples/s | {steps_per_second:.2f} steps/s"
                                )
                                print(f"{'='*70}")

                        log_loss = 0.0
                        log_steps = 0
                        last_log_time = current_time

                    for callback in self.callbacks:
                        if hasattr(callback, "on_step_end"):
                            state = type(
                                "State",
                                (),
                                {
                                    "global_step": self.global_step,
                                    "epoch": epoch,
                                },
                            )()
                            control = type(
                                "Control", (), {"should_training_stop": False}
                            )()

                            callback.on_step_end(
                                self.args,
                                state,
                                control,
                                trainer=self,
                            )

                            if control.should_training_stop:
                                if self.is_world_process_zero():
                                    print("\n" + "=" * 70)
                                    print("Training stopped by callback")
                                    print("=" * 70)
                                    epoch_pbar.close()
                                training_loss = self.total_loss / max(
                                    1, self.global_step
                                )
                                return type(
                                    "TrainOutput", (), {"training_loss": training_loss}
                                )()

                if self.is_world_process_zero():
                    epoch_pbar.update(1)

            if self.is_world_process_zero():
                epoch_pbar.close()
                epoch_time = time.time() - epoch_start_time
                print(f"\n{'='*70}")
                print(f"Epoch {epoch + 1} completed in {epoch_time/60:.2f} minutes")
                print(f"{'='*70}\n")
                epoch_start_time = time.time()

        training_loss = self.total_loss / max(1, self.global_step)
        total_time = time.time() - start_time

        if self.is_world_process_zero():
            print("\n" + "=" * 70)
            print("Training completed")
            print("=" * 70)
            print(f"Total training time: {total_time/60:.2f} minutes")
            print(f"Average loss: {training_loss:.4f}")
            print(f"Final learning rate: {self.scheduler.get_last_lr()[0]:.2e}")
            print("=" * 70 + "\n")

        return type("TrainOutput", (), {"training_loss": training_loss})()

    def save_pretrained(self, output_dir):
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            self.model_engine.module.save_pretrained(output_dir)
            self.processing_class.save_pretrained(output_dir)
