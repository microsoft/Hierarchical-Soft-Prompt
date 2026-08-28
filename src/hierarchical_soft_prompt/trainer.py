"""Training and evaluation for the public USPTO-ORD pipeline."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .logging_utils import get_logger, log_metrics, save_results
from .metrics import compute_all_metrics
from .model import PromptTunedModel
from .parser import parse_json_output


@torch.no_grad()
def evaluate_model(
    model: PromptTunedModel,
    loader,
    tokenizer,
    schema_spec,
    device: str,
    max_new_tokens: int = 512,
    include_predictions: bool = False,
) -> tuple[dict, list[dict]]:
    """Generate JSON predictions and compute the paper's public metrics."""
    model.eval()
    model.to(device)
    parsed_predictions: list[dict] = []
    parsed_golds: list[dict] = []
    all_predictions: list[dict] = []
    all_golds: list[dict] = []
    records: list[dict] = []
    parse_failures = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        full_mask = batch["attention_mask"].to(device)
        prompt_lengths = batch["prompt_lengths"].tolist()
        batch_size = input_ids.shape[0]

        if model.n_soft_tokens:
            text_mask = full_mask[:, model.n_soft_tokens :]
        else:
            text_mask = full_mask

        max_prompt = max(prompt_lengths)
        prompt_ids = torch.full(
            (batch_size, max_prompt),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        prompt_mask = torch.zeros(
            batch_size,
            max_prompt,
            dtype=text_mask.dtype,
            device=device,
        )
        for row, length in enumerate(prompt_lengths):
            prompt_ids[row, max_prompt - length :] = input_ids[row, :length]
            prompt_mask[row, max_prompt - length :] = 1

        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for text, gold_text in zip(texts, batch["target_json_str"]):
            predicted, ok = parse_json_output(text)
            gold = json.loads(gold_text)
            all_predictions.append(predicted if ok else {})
            all_golds.append(gold)
            if ok:
                parsed_predictions.append(predicted)
                parsed_golds.append(gold)
            else:
                parse_failures += 1
            if include_predictions:
                records.append(
                    {
                        "prediction": text,
                        "gold": gold,
                        "parse_ok": ok,
                    }
                )

    metrics = compute_all_metrics(
        parsed_predictions,
        parsed_golds,
        schema_spec,
        parse_failures=parse_failures,
    )
    all_sample_metrics = compute_all_metrics(
        all_predictions,
        all_golds,
        schema_spec,
        parse_failures=0,
    )
    metrics["field_em_all_samples"] = all_sample_metrics["field_em"]
    metrics["field_f1_all_samples"] = all_sample_metrics["field_f1"]
    return metrics, records


class Trainer:
    """Three-phase optimizer for LoRA, soft prompts, and joint tuning."""

    def __init__(
        self,
        model: PromptTunedModel,
        train_loader,
        val_loader,
        schema_spec,
        tokenizer,
        lora_warmup_steps: int = 0,
        soft_prompt_steps: int = 0,
        joint_steps: int = 0,
        lr_lora: float = 1e-4,
        lr_soft: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        eval_interval: int = 0,
        checkpoint_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
        run_name: str = "experiment",
        device: str = "cuda",
        max_new_tokens: int = 512,
        run_config: Optional[dict] = None,
        log_file: Optional[str] = None,
    ):
        if lora_warmup_steps + soft_prompt_steps + joint_steps <= 0:
            raise ValueError("At least one training phase must contain a step")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.schema_spec = schema_spec
        self.tokenizer = tokenizer
        self.phase_steps = [
            ("lora_warmup", lora_warmup_steps),
            ("soft_prompt", soft_prompt_steps),
            ("joint", joint_steps),
        ]
        self.lr_lora = lr_lora
        self.lr_soft = lr_soft
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.run_name = run_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.run_config = run_config or {}
        self.global_step = 0
        self.best_field_em = -1.0
        self.logger = get_logger("hierarchical_soft_prompt.trainer", log_file)

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def _build_optimizer(self, phase: str) -> AdamW:
        self.model.set_phase(phase)
        groups = []
        lora_parameters = [
            parameter
            for name, parameter in self.model.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if lora_parameters:
            groups.append(
                {
                    "params": lora_parameters,
                    "lr": self.lr_lora,
                    "weight_decay": self.weight_decay,
                }
            )
        if self.model.soft_prompt is not None:
            soft_parameters = [
                parameter
                for parameter in self.model.soft_prompt.parameters()
                if parameter.requires_grad
            ]
            if soft_parameters:
                groups.append(
                    {
                        "params": soft_parameters,
                        "lr": self.lr_soft,
                        "weight_decay": 0.0,
                    }
                )
        if not groups:
            raise RuntimeError(f"No trainable parameters for phase {phase}")
        return AdamW(groups)

    def _write_checkpoint_metadata(self, directory: str) -> None:
        with open(
            os.path.join(directory, "training_meta.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_field_em": self.best_field_em,
                },
                handle,
                indent=2,
            )
        with open(
            os.path.join(directory, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(self.run_config, handle, indent=2)

    def _save_checkpoint(self, suffix: str) -> str:
        directory = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_{suffix}",
        )
        self.model.save(directory)
        self._write_checkpoint_metadata(directory)
        return directory

    def evaluate(self) -> dict:
        metrics, _ = evaluate_model(
            self.model,
            self.val_loader,
            self.tokenizer,
            self.schema_spec,
            self.device,
            max_new_tokens=self.max_new_tokens,
        )
        return metrics

    def _train_phase(self, phase: str, total_steps: int) -> None:
        if total_steps == 0:
            return
        self.logger.info("Starting %s for %d optimizer steps", phase, total_steps)
        optimizer = self._build_optimizer(phase)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        optimizer.zero_grad(set_to_none=True)
        self.model.train()
        self.model.to(self.device)

        data_iterator = iter(self.train_loader)
        optimizer_steps = 0
        accumulated = 0
        running_loss = 0.0
        running_updates = 0

        while optimizer_steps < total_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_loader)
                batch = next(data_iterator)

            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in phase {phase}: {loss.detach().item()}"
                )
            (loss / self.gradient_accumulation_steps).backward()
            running_loss += loss.detach().item()
            accumulated += 1

            if accumulated < self.gradient_accumulation_steps:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            self.global_step += 1
            running_updates += 1
            accumulated = 0

            if optimizer_steps % 50 == 0 or optimizer_steps == total_steps:
                log_metrics(
                    {"loss": running_loss / running_updates},
                    self.global_step,
                    self.logger,
                    prefix=phase,
                )
                running_loss = 0.0
                running_updates = 0

            if (
                self.checkpoint_interval > 0
                and self.global_step % self.checkpoint_interval == 0
            ):
                self._save_checkpoint("latest")

            if self.eval_interval > 0 and self.global_step % self.eval_interval == 0:
                metrics = self.evaluate()
                field_em = metrics["field_em"]
                self.logger.info(
                    "step=%d field_em=%.4f field_f1=%.4f parse_success=%.4f",
                    self.global_step,
                    field_em,
                    metrics["field_f1"],
                    metrics["parse_success_rate"],
                )
                if field_em > self.best_field_em:
                    self.best_field_em = field_em
                    self._save_checkpoint("best")
                self.model.train()

    def train(self) -> dict:
        for phase, steps in self.phase_steps:
            self._train_phase(phase, steps)

        self._save_checkpoint("last")
        final_metrics = self.evaluate()
        if final_metrics["field_em"] > self.best_field_em:
            self.best_field_em = final_metrics["field_em"]
            self._save_checkpoint("best")

        results_path = os.path.join(
            self.results_dir,
            f"{self.run_name}_final_metrics.json",
        )
        save_results(final_metrics, results_path)
        self.logger.info(
            "Training complete: field_em=%.4f field_f1=%.4f",
            final_metrics["field_em"],
            final_metrics["field_f1"],
        )
        return final_metrics
