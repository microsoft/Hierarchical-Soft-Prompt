# Hierarchical Soft Prompt

Public implementation for **Schema-Hierarchical Soft Prompts for Structured
Information Extraction**.

This repository contains only the paper's public USPTO-ORD pipeline:

- schema-hierarchical and equal-token flat soft prompts;
- semantic initialization, DFS ordering, depth embeddings, and global-token
  component ablations;
- soft-prompt-only, LoRA-only, and joint training;
- USPTO-ORD loading, role-based canonicalization, JSON parsing, and evaluation.

It intentionally excludes proprietary datasets, task schemas, prompts,
predictions, experiment logs, checkpoints, credentials, and prior repository
history. No training data or model weights are distributed here.

## Install

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Prepare USPTO-ORD

Follow [docs/data.md](docs/data.md), then point the code at the directory
containing `train.json`, `valid.json`, and `test.json`:

```bash
export USPTO_ORD_DATA=/path/to/USPTO-n100k-t2048_exp1
```

The dataset is obtained separately from its upstream source and remains subject
to its own license. Do not commit the downloaded data.

## Train

The paper's public soft-prompt-only configuration uses a hierarchical prompt,
no hard schema prompt, seed 42, 20 global tokens, and two tokens per schema key:

```bash
hsp-train \
  --mode sp \
  --model qwen3-4b \
  --data-dir "$USPTO_ORD_DATA" \
  --prompt-style none \
  --soft-type hierarchical \
  --seed 42 \
  --run-name sp_uspto_seed42
```

The defaults use per-device batch size 4, gradient accumulation 4, evaluation
every 500 optimizer steps, and greedy decoding with 1536 new tokens. SP-only
uses 4500 steps; LoRA-only and joint training use the phase schedules reported
in the paper.

LoRA-only and joint variants:

```bash
hsp-train --mode lora --data-dir "$USPTO_ORD_DATA" --prompt-style none
hsp-train --mode joint --data-dir "$USPTO_ORD_DATA" --prompt-style none
```

Run the component ablations with:

```bash
bash scripts/run_component_ablations.sh
```

Outputs are written under `outputs/`, which is ignored by Git.

## Evaluate

```bash
hsp-evaluate \
  --checkpoint outputs/checkpoints/sp_uspto_seed42_best \
  --data-dir "$USPTO_ORD_DATA" \
  --split test \
  --output test_metrics.json
```

The evaluator reports field exact match, token F1, schema compliance,
parse-success rate, depth-stratified exact match, and confusable-field scores.
For paper compatibility, `field_em` and `field_f1` are conditioned on successful
JSON parsing. The additional `field_em_all_samples` and
`field_f1_all_samples` values count parse failures as empty predictions.

## Interpret learned prompts

The repository includes a backbone-free prompt-atlas analysis for locally
trained USPTO-ORD hierarchical soft prompts. It produces schema-group word
clouds, a PCA/depth map, cosine-similarity views, per-field metrics, and a
quantitative summary without loading the language-model backbone or training
data.

```bash
python -m pip install -e ".[interpretability]"
python scripts/analyze_prompt_atlas.py \
  --checkpoint outputs/checkpoints/sp_uspto_seed42_best \
  --output-dir outputs/interpretability/uspto_prompt_atlas
```

See [docs/interpretability.md](docs/interpretability.md) for example results,
methodology, and interpretation limits. Generated analysis remains under
`outputs/` and should not be committed.

## Safety and release scope

Keep all datasets, checkpoints, adapters, generated predictions, local
configuration, and credentials outside the repository. The `.gitignore`
contains defensive exclusions, but contributors must inspect staged files and
run a secret scanner before every release.

See [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

This project is licensed under the [MIT License](LICENSE).
