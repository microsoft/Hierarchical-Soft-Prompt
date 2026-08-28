"""Data collation for causal JSON extraction with prepended soft prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class SoftPromptCollator:
    tokenizer: PreTrainedTokenizerBase
    n_soft_tokens: int = 0
    max_length: int = 3072
    label_pad_token_id: int = -100
    include_targets: bool = True
    max_prompt_length: Optional[int] = None

    def _encode(self, feature: dict) -> tuple[list[int], int]:
        prompt_ids = list(
            self.tokenizer(
                feature["input_text"],
                add_special_tokens=True,
            )["input_ids"]
        )
        if not self.include_targets:
            prompt_limit = self.max_prompt_length or self.max_length
            prompt_ids = prompt_ids[:prompt_limit]
            return prompt_ids, len(prompt_ids)

        target_ids = list(
            self.tokenizer(
                feature["target_json_str"],
                add_special_tokens=False,
            )["input_ids"]
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids.append(eos_id)

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length - 1]
            if eos_id is not None:
                target_ids.append(eos_id)

        keep_prompt = max(1, self.max_length - len(target_ids))
        prompt_ids = prompt_ids[:keep_prompt]
        full_ids = prompt_ids + target_ids
        return full_ids, len(prompt_ids)

    def __call__(self, features: list[dict]) -> dict:
        encoded = [self._encode(feature) for feature in features]
        full_ids = [item[0] for item in encoded]
        prompt_lengths = [item[1] for item in encoded]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id")

        max_len = max(len(ids) for ids in full_ids)
        input_rows = []
        mask_rows = []
        for ids in full_ids:
            padding = max_len - len(ids)
            input_rows.append(ids + [pad_id] * padding)
            mask_rows.append([1] * len(ids) + [0] * padding)

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long)
        labels = input_ids.clone()
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, :prompt_length] = self.label_pad_token_id
        labels[attention_mask == 0] = self.label_pad_token_id

        if self.n_soft_tokens:
            batch_size = input_ids.shape[0]
            soft_mask = torch.ones(
                batch_size,
                self.n_soft_tokens,
                dtype=attention_mask.dtype,
            )
            soft_labels = torch.full(
                (batch_size, self.n_soft_tokens),
                self.label_pad_token_id,
                dtype=labels.dtype,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)
            labels = torch.cat([soft_labels, labels], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "target_json_str": [feature["target_json_str"] for feature in features],
            "target_json": [feature["target_json"] for feature in features],
        }


class ListDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def build_dataloader(
    samples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    n_soft_tokens: int = 0,
    batch_size: int = 4,
    shuffle: bool = True,
    max_length: int = 3072,
    num_workers: int = 0,
    include_targets: bool = True,
    max_prompt_length: Optional[int] = None,
) -> DataLoader:
    if not samples:
        raise ValueError("Cannot build a dataloader from an empty sample list")
    return DataLoader(
        ListDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SoftPromptCollator(
            tokenizer=tokenizer,
            n_soft_tokens=n_soft_tokens,
            max_length=max_length,
            include_targets=include_targets,
            max_prompt_length=max_prompt_length,
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
