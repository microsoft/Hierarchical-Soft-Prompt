"""
Evaluation metrics for structured JSON extraction.

All metrics operate on (predicted_json, gold_json) pairs and a list of
leaf key paths in DFS order.

Primary metrics:
  - field_em:          field-level exact match (% of fields exactly correct)
  - field_f1:          field-level token F1 (partial credit for near-misses)
  - schema_compliance: % predictions that are valid JSON matching schema structure
  - hierarchical_em:   EM broken down by nesting depth (dict of depth → score)

Diagnostic:
  - disambiguation_precision: EM on confusable field pairs only
  - per_key_em:               per-key accuracy dict
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import Any

from .schema_utils import dfs_leaf_keys, get_nested


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: Any) -> str:
    """Normalize a value to a lowercase, stripped string."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # Normalize common separators
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for F1 computation."""
    return re.findall(r"\b\w+\b", s.lower())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = defaultdict(int)
    gold_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    for t in gold_tokens:
        gold_set[t] += 1
    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set)
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def sample_field_em(
    predicted: dict,
    gold: dict,
    leaf_key_paths: list[str],
) -> tuple[float, dict[str, float]]:
    """
    Compute field-level exact match for a single sample.

    Returns:
        (mean_em, per_key_em_dict)

    Only counts keys that exist in gold (non-null). Predicted null for a gold
    non-null key counts as 0.
    """
    per_key = {}
    for kp in leaf_key_paths:
        gold_val = get_nested(gold, kp)
        if gold_val is None:
            continue
        pred_val = get_nested(predicted, kp)
        match = int(_normalize(pred_val) == _normalize(gold_val))
        per_key[kp] = float(match)

    mean_em = sum(per_key.values()) / len(per_key) if per_key else 0.0
    return mean_em, per_key


def sample_field_f1(
    predicted: dict,
    gold: dict,
    leaf_key_paths: list[str],
) -> tuple[float, dict[str, float]]:
    """
    Compute field-level token F1 for a single sample.
    """
    per_key = {}
    for kp in leaf_key_paths:
        gold_val = get_nested(gold, kp)
        if gold_val is None:
            continue
        pred_val = get_nested(predicted, kp)
        f1 = _token_f1(str(pred_val) if pred_val is not None else "", str(gold_val))
        per_key[kp] = f1

    mean_f1 = sum(per_key.values()) / len(per_key) if per_key else 0.0
    return mean_f1, per_key


def sample_schema_compliance(
    predicted: dict,
    schema: dict,
) -> float:
    """
    Check if predicted dict has valid structure w.r.t. schema.
    Returns 1.0 if all top-level schema keys that appear in predicted are correctly typed.
    Lenient: extra keys in predicted are not penalized.
    """
    if not isinstance(predicted, dict):
        return 0.0
    score = 0
    total = 0
    for key, val_schema in schema.items():
        if key not in predicted:
            continue
        total += 1
        pred_val = predicted[key]
        if isinstance(val_schema, dict):
            score += int(isinstance(pred_val, dict))
        elif isinstance(val_schema, list):
            score += int(isinstance(pred_val, list))
        else:
            score += int(isinstance(pred_val, (str, int, float, type(None))))
    return score / total if total > 0 else 1.0


# ---------------------------------------------------------------------------
# Corpus-level metrics
# ---------------------------------------------------------------------------

def corpus_field_em(
    predictions: list[dict],
    golds: list[dict],
    leaf_key_paths: list[str],
) -> dict[str, float]:
    """
    Compute corpus-level field EM metrics.

    Returns:
        {
            "field_em": float,           # mean over all samples and fields
            "per_key_em": dict[str, float],  # per-key mean over samples
            "n_samples": int,
        }
    """
    all_per_key = defaultdict(list)
    sample_ems = []

    for pred, gold in zip(predictions, golds):
        em, per_key = sample_field_em(pred, gold, leaf_key_paths)
        sample_ems.append(em)
        for kp, v in per_key.items():
            all_per_key[kp].append(v)

    per_key_mean = {kp: sum(vals) / len(vals) for kp, vals in all_per_key.items() if vals}
    return {
        "field_em": sum(sample_ems) / len(sample_ems) if sample_ems else 0.0,
        "per_key_em": per_key_mean,
        "n_samples": len(predictions),
    }


def corpus_field_f1(
    predictions: list[dict],
    golds: list[dict],
    leaf_key_paths: list[str],
) -> dict[str, float]:
    all_per_key = defaultdict(list)
    sample_f1s = []

    for pred, gold in zip(predictions, golds):
        f1, per_key = sample_field_f1(pred, gold, leaf_key_paths)
        sample_f1s.append(f1)
        for kp, v in per_key.items():
            all_per_key[kp].append(v)

    per_key_mean = {kp: sum(vals) / len(vals) for kp, vals in all_per_key.items() if vals}
    return {
        "field_f1": sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0,
        "per_key_f1": per_key_mean,
        "n_samples": len(predictions),
    }


def corpus_schema_compliance(
    predictions: list[dict],
    schema: dict,
) -> float:
    scores = [sample_schema_compliance(p, schema) for p in predictions]
    return sum(scores) / len(scores) if scores else 0.0


def corpus_hierarchical_em(
    predictions: list[dict],
    golds: list[dict],
    schema: dict,
) -> dict[int, float]:
    """
    Compute EM broken down by nesting depth.

    Returns: {depth: mean_em}
    """
    depth_ems = defaultdict(list)
    leaf_keys_with_depth = dfs_leaf_keys(schema)

    for pred, gold in zip(predictions, golds):
        for kp, depth in leaf_keys_with_depth:
            gold_val = get_nested(gold, kp)
            if gold_val is None:
                continue
            pred_val = get_nested(pred, kp)
            match = int(_normalize(pred_val) == _normalize(gold_val))
            depth_ems[depth].append(float(match))

    return {
        depth: sum(ems) / len(ems)
        for depth, ems in depth_ems.items()
        if ems
    }


def corpus_disambiguation_em(
    predictions: list[dict],
    golds: list[dict],
    confusable_groups: list[tuple[str, ...]],
) -> dict[str, float]:
    """
    Compute EM specifically on confusable field pairs.
    Returns per-key EM for each confusable field.
    """
    per_key = defaultdict(list)
    confusable_keys = set()
    for group in confusable_groups:
        for k in group:
            confusable_keys.add(k)

    for pred, gold in zip(predictions, golds):
        for kp in confusable_keys:
            gold_val = get_nested(gold, kp)
            if gold_val is None:
                continue
            pred_val = get_nested(pred, kp)
            match = int(_normalize(pred_val) == _normalize(gold_val))
            per_key[kp].append(float(match))

    return {
        kp: sum(vals) / len(vals)
        for kp, vals in per_key.items()
        if vals
    }


# ---------------------------------------------------------------------------
# Aggregate — returns the full metrics dict used in results JSON
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: list[dict],
    golds: list[dict],
    schema_spec,
    parse_failures: int = 0,
) -> dict:
    """
    Compute all metrics and return a single results dict.
    """
    leaf_key_paths = schema_spec.get_leaf_key_paths()
    schema = schema_spec.schema

    em_results = corpus_field_em(predictions, golds, leaf_key_paths)
    f1_results = corpus_field_f1(predictions, golds, leaf_key_paths)
    compliance = corpus_schema_compliance(predictions, schema)
    hier_em = corpus_hierarchical_em(predictions, golds, schema)
    disambig = corpus_disambiguation_em(predictions, golds, schema_spec.confusable_groups)

    total = len(predictions) + parse_failures

    return {
        "field_em": em_results["field_em"],
        "field_f1": f1_results["field_f1"],
        "schema_compliance": compliance,
        "parse_success_rate": len(predictions) / total if total > 0 else 0.0,
        "hierarchical_em": hier_em,
        "disambiguation_em": disambig,
        "per_key_em": em_results["per_key_em"],
        "per_key_f1": f1_results["per_key_f1"],
        "n_evaluated": len(predictions),
        "n_parse_failures": parse_failures,
    }
