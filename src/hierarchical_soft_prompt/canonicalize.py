"""
USPTO-ORD semantic canonicalization.

The ORD proto stores reaction inputs in a ``map<string, ReactionInput>`` where the
string key is free-form author-chosen ("reactant_1", "solvent", "foo_42"). That is
poor material for a schema-hierarchical soft prompt, because the hierarchy
ends up nominal — "first input vs second input" — not semantic.

Each component inside an input carries a ``reaction_role`` enum (REACTANT, SOLVENT,
CATALYST, REAGENT, WORKUP, INTERNAL_STANDARD, PRODUCT, BYPRODUCT, SIDE_PRODUCT).
That enum is the semantically real role label. Canonicalization re-buckets every
component by its role, producing a fixed 4-bucket inputs schema and flattened
outcomes — the structure our schema registry then describes.

Canonical output schema (what gold / pred both look like after canonicalization):

    {
      "inputs": {
        "reactants":  [ {identifiers, amount, reaction_role?}, ... ],
        "solvents":   [ ... ],
        "catalysts":  [ ... ],
        "reagents":   [ ... ],
      },
      "conditions": { "temperature": {...}, "pressure": {...},
                      "stirring": {...}, "ph": {...}, "details": "str" },
      "outcomes":   { "products": [ {identifiers, amount, measurements}, ... ] },
      "workups":    [ {type, details, duration}, ... ],
    }

Missing sections are omitted (a reaction with no workup has no "workups" key), so
scoring focuses on fields that actually exist in gold.
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Role bucketing
# ---------------------------------------------------------------------------

# Map ORD reaction_role enum → canonical bucket name.
# Covers the ~10 enum values that appear in USPTO-ORD; anything unknown → "reagents".
_ROLE_TO_BUCKET = {
    "REACTANT":          "reactants",
    "SOLVENT":           "solvents",
    "CATALYST":          "catalysts",
    "REAGENT":           "reagents",
    "INTERNAL_STANDARD": "reagents",
    "AUTHENTIC_STANDARD": "reagents",
    # PRODUCT / BYPRODUCT / SIDE_PRODUCT / WORKUP flow through outcomes / workups
    # paths, not inputs, so they shouldn't appear on input components — but if they
    # do (bad data), treat them as reagents rather than dropping silently.
    "PRODUCT":           "reagents",
    "BYPRODUCT":         "reagents",
    "SIDE_PRODUCT":      "reagents",
    "WORKUP":            "reagents",
}

# When the proto field ``reaction_role`` is absent, infer from the map-key name.
# ORD convention: map keys often encode role prefix. Order matters: longer
# keywords first so "solvent_1" hits SOLVENT before the "n" / "t" tails.
_MAP_KEY_TO_ROLE = [
    ("catalyst",          "CATALYST"),
    ("ligand",            "CATALYST"),
    ("solvent",           "SOLVENT"),
    ("base",              "REAGENT"),
    ("acid",              "REAGENT"),
    ("quench",            "REAGENT"),
    ("reactant",          "REACTANT"),
    ("substrate",         "REACTANT"),
    ("starting_material", "REACTANT"),
    ("starting-material", "REACTANT"),
]


def _infer_role_from_key(map_key: str) -> str:
    k = map_key.lower()
    for needle, role in _MAP_KEY_TO_ROLE:
        if needle in k:
            return role
    return "REAGENT"


def _component_role(component: dict, fallback_map_key: str) -> str:
    role = component.get("reaction_role")
    if isinstance(role, str) and role.strip():
        return role.strip().upper()
    return _infer_role_from_key(fallback_map_key)


# ---------------------------------------------------------------------------
# Section canonicalizers
# ---------------------------------------------------------------------------

def _pick_first_identifier(identifiers: Any) -> dict:
    """
    ORD allows multiple identifier entries per compound (SMILES, InChI, name, ...).
    For schema-stable leaf paths we pick the first SMILES, falling back to the
    first entry. Preserves both fields so scoring can match whichever the model
    emits.
    """
    if not isinstance(identifiers, list) or not identifiers:
        return {}
    smiles_entry = next(
        (i for i in identifiers
         if isinstance(i, dict) and str(i.get("type", "")).upper() == "SMILES"),
        None,
    )
    chosen = smiles_entry if smiles_entry is not None else identifiers[0]
    if not isinstance(chosen, dict):
        return {}
    out = {}
    if "value" in chosen:
        out["value"] = chosen["value"]
    if "type" in chosen:
        out["type"] = chosen["type"]
    return out


def _canonicalize_amount(amount: Any) -> dict:
    """ORD Amount has one-of mass / moles / volume / unmeasured. Keep the populated leg."""
    if not isinstance(amount, dict):
        return {}
    out = {}
    for leg in ("mass", "moles", "volume"):
        leg_val = amount.get(leg)
        if isinstance(leg_val, dict):
            sub = {}
            if "value" in leg_val:
                sub["value"] = leg_val["value"]
            if "units" in leg_val:
                sub["units"] = leg_val["units"]
            if sub:
                out[leg] = sub
    return out


def _canonicalize_component(comp: dict) -> dict:
    if not isinstance(comp, dict):
        return {}
    out = {}
    ident = _pick_first_identifier(comp.get("identifiers", []))
    if ident:
        out["identifiers"] = ident
    amt = _canonicalize_amount(comp.get("amount"))
    if amt:
        out["amount"] = amt
    return out


def _canonicalize_inputs(raw_inputs: Any) -> dict:
    """Bucket all input components by reaction_role."""
    buckets: dict[str, list] = {
        "reactants": [], "solvents": [], "catalysts": [], "reagents": [],
    }
    if not isinstance(raw_inputs, dict):
        return {}

    for map_key, inp in raw_inputs.items():
        if not isinstance(inp, dict):
            continue
        for comp in inp.get("components", []) or []:
            role = _component_role(comp, fallback_map_key=str(map_key))
            bucket = _ROLE_TO_BUCKET.get(role, "reagents")
            canon = _canonicalize_component(comp)
            if canon:
                buckets[bucket].append(canon)

    # Drop empty buckets so get_nested / dfs_leaf_keys see only keys with content.
    return {k: v for k, v in buckets.items() if v}


def _canonicalize_conditions(raw_conditions: Any) -> dict:
    if not isinstance(raw_conditions, dict):
        return {}
    out = {}

    temp = raw_conditions.get("temperature")
    if isinstance(temp, dict) and isinstance(temp.get("setpoint"), dict):
        sp = temp["setpoint"]
        val = {k: sp[k] for k in ("value", "units") if k in sp}
        if val:
            out["temperature"] = {"setpoint": val}

    pres = raw_conditions.get("pressure")
    if isinstance(pres, dict) and isinstance(pres.get("setpoint"), dict):
        sp = pres["setpoint"]
        val = {k: sp[k] for k in ("value", "units") if k in sp}
        if val:
            out["pressure"] = {"setpoint": val}

    stir = raw_conditions.get("stirring")
    if isinstance(stir, dict) and "type" in stir:
        out["stirring"] = {"type": stir["type"]}

    ph = raw_conditions.get("ph")
    if ph is not None and not (isinstance(ph, dict) and not ph):
        out["ph"] = {"value": ph} if not isinstance(ph, dict) else ph

    details = raw_conditions.get("details")
    if isinstance(details, str) and details.strip():
        out["details"] = details.strip()

    return out


def _canonicalize_outcomes(raw_outcomes: Any) -> dict:
    """Flatten outcomes[] → products list; filter to role=PRODUCT / default-product."""
    if not isinstance(raw_outcomes, list) or not raw_outcomes:
        return {}
    products = []
    for outcome in raw_outcomes:
        if not isinstance(outcome, dict):
            continue
        for prod in outcome.get("products", []) or []:
            if not isinstance(prod, dict):
                continue
            role = str(prod.get("reaction_role", "PRODUCT")).upper()
            if role not in ("PRODUCT", ""):
                continue   # byproducts / side-products excluded from canonical.products
            canon = {}
            ident = _pick_first_identifier(prod.get("identifiers", []))
            if ident:
                canon["identifiers"] = ident
            amt = _canonicalize_amount(prod.get("amount"))
            if amt:
                canon["amount"] = amt

            # Measurements — keep first yield/purity numeric leaf if present.
            measurements = prod.get("measurements", []) or []
            yield_val = None
            for m in measurements:
                if not isinstance(m, dict):
                    continue
                mtype = str(m.get("type", "")).upper()
                if mtype == "YIELD" and isinstance(m.get("percentage"), dict):
                    yield_val = m["percentage"].get("value")
                    break
            if yield_val is not None:
                canon["yield"] = {"value": yield_val}
            if canon:
                products.append(canon)
    return {"products": products} if products else {}


def _canonicalize_workups(raw_workups: Any) -> list:
    if not isinstance(raw_workups, list):
        return []
    out = []
    for w in raw_workups:
        if not isinstance(w, dict):
            continue
        canon = {}
        if isinstance(w.get("type"), str):
            canon["type"] = w["type"]
        if isinstance(w.get("details"), str) and w["details"].strip():
            canon["details"] = w["details"].strip()
        dur = w.get("duration")
        if isinstance(dur, dict):
            sub = {k: dur[k] for k in ("value", "units") if k in dur}
            if sub:
                canon["duration"] = sub
        if canon:
            out.append(canon)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def canonicalize_reaction(raw: dict) -> dict:
    """
    Produce the canonical, role-bucketed form of one ORD reaction dict.
    Empty top-level sections are omitted.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    inputs = _canonicalize_inputs(raw.get("inputs"))
    if inputs:
        out["inputs"] = inputs
    conditions = _canonicalize_conditions(raw.get("conditions"))
    if conditions:
        out["conditions"] = conditions
    outcomes = _canonicalize_outcomes(raw.get("outcomes"))
    if outcomes:
        out["outcomes"] = outcomes
    workups = _canonicalize_workups(raw.get("workups"))
    if workups:
        out["workups"] = workups
    return out
