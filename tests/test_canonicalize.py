from hierarchical_soft_prompt.canonicalize import canonicalize_reaction


def test_canonicalize_buckets_components_by_role():
    raw = {
        "inputs": {
            "input_1": {
                "components": [
                    {
                        "reaction_role": "REACTANT",
                        "identifiers": [{"value": "CCO", "type": "SMILES"}],
                        "amount": {"moles": {"value": 1.0, "units": "MOLE"}},
                    },
                    {
                        "reaction_role": "SOLVENT",
                        "identifiers": [{"value": "O", "type": "SMILES"}],
                    },
                ]
            }
        }
    }

    canonical = canonicalize_reaction(raw)

    assert canonical["inputs"]["reactants"][0]["identifiers"]["value"] == "CCO"
    assert canonical["inputs"]["solvents"][0]["identifiers"]["value"] == "O"


def test_canonicalize_omits_empty_sections():
    assert canonicalize_reaction({"inputs": {}, "conditions": {}}) == {}
