"""
USPTO-ORD-100K reaction-extraction adapter.

Reads the Alpaca-format JSON files produced by the qai222/LLM_organic_synthesis
prep pipeline (see docs/data.md) and converts them to the
(input_text, target_json) pair shape our raw-concat collator expects.

Not on HuggingFace — data must be prepped on the server first. Points via
$USPTO_ORD_DATA env var or the data_dir argument.
"""

from __future__ import annotations
import json
import os
from typing import Optional

from .canonicalize import canonicalize_reaction


_FULL_PLUS_SCHEMA_BLOCK = """\
You are extracting structured data from an organic chemistry procedure into a fixed JSON schema.

Schema (use the subset of top-level sections that the procedure describes — omit any section with no content):
{{
  "inputs": {{
    "reactants":  [ {{ "identifiers": {{"value": "<SMILES_or_name>", "type": "SMILES"|"NAME"|"INCHI"|"CAS"}},
                      "amount": {{ "mass":   {{"value": <float>, "units": "GRAM"|"MILLIGRAM"}},
                                  "moles":  {{"value": <float>, "units": "MILLIMOLE"|"MOLE"}},
                                  "volume": {{"value": <float>, "units": "MILLILITER"|"LITER"}} }} }} ],
    "solvents":   [ <same shape as reactants; typically volume only> ],
    "catalysts":  [ <same shape as reactants> ],
    "reagents":   [ <same shape as reactants> ]
  }},
  "conditions": {{
    "temperature": {{"setpoint": {{"value": <float>, "units": "CELSIUS"|"KELVIN"|"FAHRENHEIT"}}}},
    "pressure":    {{"setpoint": {{"value": <float>, "units": "BAR"|"PSI"|"ATMOSPHERE"|"TORR"}}}},
    "stirring":    {{"type": "STIR_BAR"|"OVERHEAD_MIXER"|"AGITATION"|"NONE"}},
    "ph":          {{"value": <float>}},
    "details":     "<free text for any non-enumerated condition>"
  }},
  "workups": [
    {{ "type": "EXTRACTION"|"FILTRATION"|"WASH"|"DRY_WITH_MATERIAL"|"CONCENTRATION"|"ADDITION"|"TEMPERATURE"|"CUSTOM",
      "details": "<free text>",
      "duration": {{"value": <float>, "units": "SECOND"|"MINUTE"|"HOUR"|"DAY"}} }}
  ],
  "outcomes": {{
    "products": [ {{ "identifiers": {{"value": "<SMILES_or_name>", "type": "SMILES"|"NAME"}},
                    "amount":      {{"mass": {{"value": <float>, "units": "GRAM"|"MILLIGRAM"}}}},
                    "yield":       {{"value": <float, percentage>}} }} ]
  }}
}}

Rules:
1. Bucket each input component by its role in the reaction: REACTANTS participate in bond-forming, SOLVENTS are the bulk medium, CATALYSTS are sub-stoichiometric mediators (Pd, Ni, ligand complexes, acids/bases used catalytically), REAGENTS are stoichiometric helpers (bases like K2CO3, oxidants, coupling partners that are not the main substrate).
2. Amount: keep only the populated leg of {{mass, moles, volume}}. If multiple are stated (e.g., "2.00 g, 10.8 mmol"), keep both.
3. Identifiers: prefer SMILES if inferable from the compound name; otherwise use "NAME" with the literal name.
4. Omit sections not described. Omit empty arrays. A reaction with no explicit workup has no "workups" key.
5. Yield is a percentage (e.g., 88.0, not 0.88).
6. Output ONLY the JSON object — no prose, no markdown fences."""


_FULL_PLUS_ONE_SHOT_EXAMPLE = """\

Example —
Procedure:
A mixture of 4-bromobenzaldehyde (2.00 g, 10.8 mmol) and phenylboronic acid (1.58 g, 13.0 mmol) in dioxane (30 mL) and aqueous K2CO3 (2 M, 15 mL) was degassed with N2. Pd(PPh3)4 (0.25 g, 0.22 mmol) was added and the mixture was stirred at 85 C for 12 h. The reaction was cooled, diluted with EtOAc (50 mL) and washed with brine (2 x 30 mL). The organic layer was dried over MgSO4, filtered, and concentrated. The residue was purified by silica gel chromatography (hexane/EtOAc 4:1) to afford the title compound as a white solid (1.73 g, 88%).
JSON:
{{"inputs":{{"reactants":[{{"identifiers":{{"value":"O=Cc1ccc(Br)cc1","type":"SMILES"}},"amount":{{"mass":{{"value":2.0,"units":"GRAM"}},"moles":{{"value":10.8,"units":"MILLIMOLE"}}}}}},{{"identifiers":{{"value":"OB(O)c1ccccc1","type":"SMILES"}},"amount":{{"mass":{{"value":1.58,"units":"GRAM"}},"moles":{{"value":13.0,"units":"MILLIMOLE"}}}}}}],"solvents":[{{"identifiers":{{"value":"C1CCOCC1","type":"SMILES"}},"amount":{{"volume":{{"value":30.0,"units":"MILLILITER"}}}}}}],"catalysts":[{{"identifiers":{{"value":"[Pd](PPh3)4","type":"SMILES"}},"amount":{{"mass":{{"value":0.25,"units":"GRAM"}},"moles":{{"value":0.22,"units":"MILLIMOLE"}}}}}}],"reagents":[{{"identifiers":{{"value":"[K+].[K+].[O-]C([O-])=O","type":"SMILES"}}}}]}},"conditions":{{"temperature":{{"setpoint":{{"value":85.0,"units":"CELSIUS"}}}},"stirring":{{"type":"STIR_BAR"}}}},"workups":[{{"type":"EXTRACTION","details":"diluted with EtOAc and washed with brine"}},{{"type":"DRY_WITH_MATERIAL","details":"MgSO4"}},{{"type":"FILTRATION"}},{{"type":"CONCENTRATION"}},{{"type":"CUSTOM","details":"silica gel chromatography hexane/EtOAc 4:1"}}],"outcomes":{{"products":[{{"identifiers":{{"value":"O=Cc1ccc(-c2ccccc2)cc1","type":"SMILES"}},"amount":{{"mass":{{"value":1.73,"units":"GRAM"}}}},"yield":{{"value":88.0}}}}]}}}}
"""


PROMPTS = {
    "full": """\
Extract structured information from the organic synthesis procedure below.
Output ONLY a valid JSON object with these top-level keys (use the subset that applies):
- inputs: reagents and starting materials (each with components, amount, role)
- conditions: reaction conditions (temperature, time, pressure, stirring)
- workups: post-reaction steps (extractions, washes, filtrations)
- outcomes: products with analyses and measured amounts

Procedure:
{procedure_text}

Extracted JSON:""",

    "minimal": """\
Extract structured information from the procedure below.
Output ONLY a valid JSON object.

Procedure:
{procedure_text}

JSON:""",

    "none": "{procedure_text}\n",

    # Strong zero-shot baseline: explicit schema + field descriptions + enum values + rules
    "full_plus": _FULL_PLUS_SCHEMA_BLOCK + """

Procedure:
{procedure_text}

JSON:""",

    # Same as full_plus plus one worked Suzuki coupling example — crosses into 1-shot ICL
    "full_plus_1shot": _FULL_PLUS_SCHEMA_BLOCK + _FULL_PLUS_ONE_SHOT_EXAMPLE + """
Procedure:
{procedure_text}
JSON:""",
}


PROCEDURE_MARKER = "### Procedure:\n"
JSON_MARKER = "\n\n### ORD JSON:"


def _extract_procedure_text(instruction: str) -> str:
    """
    Recover the raw procedure paragraph from the Alpaca-wrapped instruction.

    The upstream prep script wraps every sample in:
        "Below is a description... ### Procedure:\\n<text>\\n\\n### ORD JSON:\\n"
    We slice between the two markers. Returns "" if markers aren't found.
    """
    start = instruction.find(PROCEDURE_MARKER)
    if start == -1:
        return ""
    start += len(PROCEDURE_MARKER)
    end = instruction.find(JSON_MARKER, start)
    if end == -1:
        return instruction[start:].strip()
    return instruction[start:end].strip()


def _resolve_data_dir(data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir
    env = os.environ.get("USPTO_ORD_DATA")
    if env:
        return env
    raise RuntimeError(
        "USPTO-ORD data directory not found. Either pass data_dir=... or set "
        "$USPTO_ORD_DATA. See docs/data.md for preparation steps."
    )


def load_uspto_ord(
    split: str = "train",              # "train" | "valid" | "test"
    data_dir: Optional[str] = None,    # parent of {train,valid,test}.json
    max_samples: Optional[int] = None,
    prompt_style: str = "full",        # "full" | "minimal" | "none"
    canonicalize: bool = True,         # apply reaction_role-based canonicalization
) -> list[dict]:
    """
    Load USPTO-ORD-100K reactions and return samples for the raw-concat collator.

    Each sample:
        {
            "input_text":     str,    # hard prompt + procedure paragraph
            "target_json":    dict,   # nested reaction JSON (inputs/conditions/outcomes/workups)
            "target_json_str": str,
            "reaction_id":    str,    # row index within split (no upstream id in Alpaca rows)
            "dataset":        "uspto_ord",
        }

    Malformed rows (missing procedure text, unparseable output JSON) are skipped.
    """
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be one of train/valid/test, got {split!r}")

    dir_ = _resolve_data_dir(data_dir)
    path = os.path.join(dir_, f"{split}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Expected {split}.json inside {dir_}. "
            "Run the steps in docs/data.md."
        )

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a JSON list")

    prompt_template = PROMPTS.get(prompt_style, PROMPTS["full"])
    samples = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        instruction = row.get("instruction")
        output_str = row.get("output")
        if not isinstance(instruction, str) or not isinstance(output_str, str):
            continue

        procedure_text = _extract_procedure_text(instruction)
        if not procedure_text:
            continue

        try:
            target_json = json.loads(output_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(target_json, dict) or not target_json:
            continue

        if canonicalize:
            target_json = canonicalize_reaction(target_json)
            if not target_json:
                continue  # canonicalization stripped everything — skip sample

        samples.append({
            "input_text": prompt_template.format(procedure_text=procedure_text),
            "target_json": target_json,
            "target_json_str": json.dumps(target_json, ensure_ascii=False),
            "reaction_id": str(idx),
            "dataset": "uspto_ord",
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
