#!/usr/bin/env bash
set -euo pipefail

: "${USPTO_ORD_DATA:?Set USPTO_ORD_DATA to the directory containing train.json, valid.json, and test.json}"

COMMON=(
  --mode sp
  --model qwen3-4b
  --data-dir "$USPTO_ORD_DATA"
  --prompt-style none
  --soft-type hierarchical
  --seed 42
)

hsp-train "${COMMON[@]}" --run-name full
hsp-train "${COMMON[@]}" --init random --run-name random_init
hsp-train "${COMMON[@]}" --key-order shuffle --run-name shuffled_order
hsp-train "${COMMON[@]}" --no-depth --run-name no_depth
hsp-train "${COMMON[@]}" --kg 0 --run-name no_global_tokens
hsp-train "${COMMON[@]}" --soft-type flat --run-name flat_equal_token
