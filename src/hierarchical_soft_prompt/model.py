"""Causal language model integration for soft prompts and LoRA."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


_LORA_TARGET_REGISTRY: list[tuple[str, list[str]]] = [
    ("Phi-4-mini", ["qkv_proj", "o_proj"]),
    ("phi-4", ["qkv_proj", "o_proj"]),
    ("Phi-3", ["qkv_proj", "o_proj"]),
    ("Phi-3.5", ["qkv_proj", "o_proj"]),
    ("Qwen2", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("Qwen3", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("Llama", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("Mistral", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("Mixtral", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("gemma", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("falcon", ["query_key_value", "dense"]),
    ("neox", ["query_key_value", "dense"]),
]


def resolve_dtype(name: str) -> torch.dtype:
    """Map a CLI dtype name to a torch dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _verify_modules(model, modules: list[str], model_name: str) -> None:
    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [name for name in modules if name not in leaf_names]
    if missing:
        raise ValueError(
            f"LoRA target modules {missing} were not found in {model_name}. "
            "Pass explicit target modules for this architecture."
        )


def get_lora_target_modules(model_name: str, model) -> list[str]:
    """Resolve and validate LoRA projection names for a model family."""
    lower_name = model_name.lower()
    for key, modules in _LORA_TARGET_REGISTRY:
        if key.lower() in lower_name:
            _verify_modules(model, modules, model_name)
            return modules

    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    if "qkv_proj" in leaf_names:
        modules = ["qkv_proj"]
    elif "kv_proj" in leaf_names and "q_proj" in leaf_names:
        modules = ["q_proj", "kv_proj"]
    else:
        modules = ["q_proj", "k_proj", "v_proj"]
    modules.append("o_proj" if "o_proj" in leaf_names else "dense")
    _verify_modules(model, modules, model_name)
    return modules


def _enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def build_base_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = True,
):
    """Load a frozen causal LM for soft-prompt-only training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    return model


def build_lora_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = True,
):
    """Load a causal LM and attach a trainable LoRA adapter."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if target_modules is None:
        target_modules = get_lora_target_modules(model_name, base_model)
    if gradient_checkpointing:
        _enable_gradient_checkpointing(base_model)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(base_model, config)


class PromptTunedModel(nn.Module):
    """Prepend an optional learned prompt to a frozen or LoRA-tuned LM."""

    def __init__(
        self,
        backbone,
        soft_prompt: Optional[nn.Module] = None,
        lora_enabled: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.soft_prompt = soft_prompt
        self.lora_enabled = lora_enabled
        self._embed_layer = backbone.get_input_embeddings()

    @property
    def n_soft_tokens(self) -> int:
        return self.soft_prompt.num_tokens if self.soft_prompt is not None else 0

    def _prepend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed_layer(input_ids)
        if self.soft_prompt is None:
            return embeddings, attention_mask

        batch_size = input_ids.shape[0]
        soft = self.soft_prompt().to(embeddings.dtype)
        soft = soft.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft, embeddings], dim=1), attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        embeddings, attention_mask = self._prepend(input_ids, attention_mask)
        return self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **generation_kwargs,
    ) -> torch.Tensor:
        embeddings = self._embed_layer(input_ids)
        if self.soft_prompt is not None:
            batch_size = input_ids.shape[0]
            soft = self.soft_prompt().to(embeddings.dtype)
            soft = soft.unsqueeze(0).expand(batch_size, -1, -1)
            embeddings = torch.cat([soft, embeddings], dim=1)
            soft_mask = torch.ones(
                batch_size,
                self.n_soft_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        eos_id = self.backbone.config.eos_token_id
        pad_id = self.backbone.config.pad_token_id
        if pad_id is None:
            pad_id = eos_id
        return self.backbone.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
            **generation_kwargs,
        )

    def set_phase(self, phase: str) -> None:
        """Select trainable parameters for a LoRA, soft-prompt, or joint phase."""
        if phase not in {"lora_warmup", "soft_prompt", "joint"}:
            raise ValueError(f"Unknown training phase: {phase}")

        train_lora = phase in {"lora_warmup", "joint"}
        train_soft = phase in {"soft_prompt", "joint"}

        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = train_lora and "lora_" in name
        if self.soft_prompt is not None:
            self.soft_prompt.requires_grad_(train_soft)

    def get_trainable_param_count(self) -> dict[str, int]:
        return {
            "lora": sum(
                parameter.numel()
                for name, parameter in self.backbone.named_parameters()
                if parameter.requires_grad and "lora_" in name
            ),
            "soft_prompt": (
                sum(p.numel() for p in self.soft_prompt.parameters() if p.requires_grad)
                if self.soft_prompt is not None
                else 0
            ),
        }

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self.lora_enabled:
            self.backbone.save_pretrained(output_dir)
        if self.soft_prompt is not None:
            torch.save(
                self.soft_prompt.state_dict(),
                os.path.join(output_dir, "soft_prompt.pt"),
            )

    def load(
        self,
        checkpoint_dir: str,
        is_trainable: bool = False,
    ) -> None:
        if self.lora_enabled:
            if isinstance(self.backbone, PeftModel):
                base_model = self.backbone.unload()
            else:
                base_model = self.backbone
            self.backbone = PeftModel.from_pretrained(
                base_model,
                checkpoint_dir,
                is_trainable=is_trainable,
            )
            self._embed_layer = self.backbone.get_input_embeddings()

        if self.soft_prompt is not None:
            path = os.path.join(checkpoint_dir, "soft_prompt.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing soft prompt checkpoint: {path}")
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            self.soft_prompt.load_state_dict(state)


JointModel = PromptTunedModel
