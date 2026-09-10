"""Command-line entry points for training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .collator import build_dataloader
from .data import PROMPTS, load_uspto_ord
from .model import (
    PromptTunedModel,
    build_base_model,
    build_lora_model,
    resolve_dtype,
)
from .schema import USPTO_ORD_SPEC
from .schema_utils import dfs_keys
from .seed import set_seed
from .soft_prompt import FlatSoftPrompt, SchemaHierarchicalSoftPrompt
from .trainer import Trainer, evaluate_model


MODEL_ALIASES = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}


def _model_id(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _configure_tokenizer(model_id: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer defines neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_soft_prompt(config: dict, embed_dim: int):
    if config["mode"] not in {"sp", "joint"}:
        return None
    if config["soft_type"] == "flat":
        token_count = config["kg"] + len(dfs_keys(USPTO_ORD_SPEC.schema)) * config["t"]
        return FlatSoftPrompt(token_count, embed_dim)
    return SchemaHierarchicalSoftPrompt(
        schema=USPTO_ORD_SPEC.schema,
        embed_dim=embed_dim,
        kg=config["kg"],
        t=config["t"],
        use_depth=config["use_depth"],
        key_order=config["key_order"],
        order_seed=config["seed"],
    )


def _build_model(
    config: dict,
    initialize: bool,
    tokenizer=None,
    semantic_text: str | None = None,
):
    model_id = config["model_id"]
    dtype = resolve_dtype(config["dtype"])
    if config["mode"] == "sp":
        backbone = build_base_model(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=config["trust_remote_code"],
            gradient_checkpointing=config["gradient_checkpointing"],
        )
        lora_enabled = False
    else:
        backbone = build_lora_model(
            model_id,
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            torch_dtype=dtype,
            trust_remote_code=config["trust_remote_code"],
            gradient_checkpointing=config["gradient_checkpointing"],
        )
        lora_enabled = True

    embed_dim = backbone.get_input_embeddings().embedding_dim
    soft_prompt = _build_soft_prompt(config, embed_dim)
    if initialize and soft_prompt is not None and config["init"] == "semantic":
        if tokenizer is None or not semantic_text:
            raise ValueError("Semantic initialization requires a tokenizer and sample text")
        soft_prompt.initialize_from_embeddings(
            tokenizer,
            backbone.get_input_embeddings(),
            semantic_text,
        )
    return PromptTunedModel(
        backbone=backbone,
        soft_prompt=soft_prompt,
        lora_enabled=lora_enabled,
    )


def _common_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["sp", "lora", "joint"], default="sp")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPTS),
        default="none",
    )
    parser.add_argument(
        "--soft-type",
        choices=["hierarchical", "flat"],
        default="hierarchical",
    )
    parser.add_argument("--kg", type=int, default=20)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--init", choices=["semantic", "random"], default="semantic")
    parser.add_argument("--key-order", choices=["dfs", "shuffle"], default="dfs")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")


def _resolve_phase_steps(args) -> tuple[int, int, int]:
    if args.mode == "sp":
        return 0, args.soft_prompt_steps or 4500, 0
    if args.mode == "lora":
        return (
            args.lora_warmup_steps if args.lora_warmup_steps is not None else 1000,
            0,
            args.joint_steps if args.joint_steps is not None else 2000,
        )
    return (
        args.lora_warmup_steps if args.lora_warmup_steps is not None else 500,
        args.soft_prompt_steps if args.soft_prompt_steps is not None else 500,
        args.joint_steps if args.joint_steps is not None else 2000,
    )


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train public USPTO-ORD soft-prompt and LoRA baselines."
    )
    _common_train_arguments(parser)
    parser.add_argument("--lora-warmup-steps", type=int)
    parser.add_argument("--soft-prompt-steps", type=int)
    parser.add_argument("--joint-steps", type=int)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-soft", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    model_id = _model_id(args.model)
    max_length = args.max_length or (3072 if args.mode == "sp" else 5120)
    run_name = args.run_name or (
        f"{args.mode}_{args.soft_type}_{args.prompt_style}_seed{args.seed}"
    )
    config = {
        "mode": args.mode,
        "model_id": model_id,
        "prompt_style": args.prompt_style,
        "soft_type": args.soft_type,
        "kg": args.kg,
        "t": args.t,
        "init": args.init,
        "key_order": args.key_order,
        "use_depth": not args.no_depth,
        "seed": args.seed,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": max_length,
        "max_new_tokens": args.max_new_tokens,
    }

    train_samples = load_uspto_ord(
        "train",
        data_dir=args.data_dir,
        max_samples=args.max_train_samples,
        prompt_style=args.prompt_style,
    )
    val_samples = load_uspto_ord(
        "valid",
        data_dir=args.data_dir,
        max_samples=args.max_val_samples,
        prompt_style=args.prompt_style,
    )
    tokenizer = _configure_tokenizer(model_id, args.trust_remote_code)
    model = _build_model(
        config,
        initialize=True,
        tokenizer=tokenizer,
        semantic_text=train_samples[0]["input_text"][:500],
    )
    train_loader = build_dataloader(
        train_samples,
        tokenizer,
        n_soft_tokens=model.n_soft_tokens,
        batch_size=args.batch_size,
        shuffle=True,
        max_length=max_length,
    )
    val_loader = build_dataloader(
        val_samples,
        tokenizer,
        n_soft_tokens=model.n_soft_tokens,
        batch_size=args.batch_size,
        shuffle=False,
        max_length=max_length,
        include_targets=False,
        max_prompt_length=max(1, max_length - args.max_new_tokens),
    )

    lora_steps, soft_steps, joint_steps = _resolve_phase_steps(args)
    output_dir = Path(args.output_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        schema_spec=USPTO_ORD_SPEC,
        tokenizer=tokenizer,
        lora_warmup_steps=lora_steps,
        soft_prompt_steps=soft_steps,
        joint_steps=joint_steps,
        lr_lora=args.lr_lora,
        lr_soft=args.lr_soft,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=str(output_dir / "checkpoints"),
        results_dir=str(output_dir / "results"),
        run_name=run_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        run_config=config,
        log_file=str(output_dir / "logs" / f"{run_name}.log"),
    )
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved public checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--predictions")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    config_path = checkpoint / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    set_seed(config["seed"])
    tokenizer = _configure_tokenizer(
        config["model_id"],
        config["trust_remote_code"],
    )
    model = _build_model(config, initialize=False)
    model.load(str(checkpoint), is_trainable=False)
    samples = load_uspto_ord(
        args.split,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        prompt_style=config["prompt_style"],
    )
    loader = build_dataloader(
        samples,
        tokenizer,
        n_soft_tokens=model.n_soft_tokens,
        batch_size=args.batch_size,
        shuffle=False,
        max_length=config["max_length"],
        include_targets=False,
        max_prompt_length=max(
            1,
            config["max_length"] - config["max_new_tokens"],
        ),
    )
    metrics, predictions = evaluate_model(
        model,
        loader,
        tokenizer,
        USPTO_ORD_SPEC,
        args.device,
        max_new_tokens=config["max_new_tokens"],
        include_predictions=bool(args.predictions),
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if args.predictions:
        with open(args.predictions, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)
    print(json.dumps(metrics, indent=2))
