"""Public schema definition for the canonical USPTO-ORD task."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema_utils import dfs_keys, dfs_leaf_keys, max_depth


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    schema: dict
    confusable_groups: list[tuple[str, ...]] = field(default_factory=list)
    field_weights: dict[str, float] = field(default_factory=dict)

    def get_key_paths(self) -> list[str]:
        return [path for path, _ in dfs_keys(self.schema)]

    def get_keys_with_depth(self) -> list[tuple[str, int]]:
        return dfs_keys(self.schema)

    def get_leaf_key_paths(self) -> list[str]:
        return [path for path, _ in dfs_leaf_keys(self.schema)]

    def get_max_depth(self) -> int:
        return max_depth(self.schema)

    def get_key_weights(self) -> dict[str, float]:
        return {
            path: self.field_weights.get(path, 1.0)
            for path in self.get_leaf_key_paths()
        }


USPTO_ORD_SCHEMA = {
    "inputs": {
        "reactants": [
            {
                "identifiers": {"value": "str", "type": "str"},
                "amount": {
                    "mass": {"value": "float", "units": "str"},
                    "moles": {"value": "float", "units": "str"},
                    "volume": {"value": "float", "units": "str"},
                },
            }
        ],
        "solvents": [
            {
                "identifiers": {"value": "str", "type": "str"},
                "amount": {
                    "mass": {"value": "float", "units": "str"},
                    "moles": {"value": "float", "units": "str"},
                    "volume": {"value": "float", "units": "str"},
                },
            }
        ],
        "catalysts": [
            {
                "identifiers": {"value": "str", "type": "str"},
                "amount": {
                    "mass": {"value": "float", "units": "str"},
                    "moles": {"value": "float", "units": "str"},
                    "volume": {"value": "float", "units": "str"},
                },
            }
        ],
        "reagents": [
            {
                "identifiers": {"value": "str", "type": "str"},
                "amount": {
                    "mass": {"value": "float", "units": "str"},
                    "moles": {"value": "float", "units": "str"},
                    "volume": {"value": "float", "units": "str"},
                },
            }
        ],
    },
    "conditions": {
        "temperature": {"setpoint": {"value": "float", "units": "str"}},
        "pressure": {"setpoint": {"value": "float", "units": "str"}},
        "stirring": {"type": "str"},
        "ph": {"value": "float"},
        "details": "str",
    },
    "outcomes": {
        "products": [
            {
                "identifiers": {"value": "str", "type": "str"},
                "amount": {
                    "mass": {"value": "float", "units": "str"},
                    "moles": {"value": "float", "units": "str"},
                    "volume": {"value": "float", "units": "str"},
                },
                "yield": {"value": "float"},
            }
        ]
    },
    "workups": [
        {
            "type": "str",
            "details": "str",
            "duration": {"value": "float", "units": "str"},
        }
    ],
}

USPTO_ORD_CONFUSABLES = [
    (
        "conditions.temperature.setpoint.value",
        "conditions.pressure.setpoint.value",
    ),
    (
        "inputs.reactants.amount.mass.value",
        "inputs.reactants.amount.moles.value",
    ),
    (
        "inputs.solvents.amount.volume.value",
        "inputs.reactants.amount.volume.value",
    ),
]

USPTO_ORD_SPEC = SchemaSpec(
    name="uspto_ord",
    schema=USPTO_ORD_SCHEMA,
    confusable_groups=USPTO_ORD_CONFUSABLES,
)
