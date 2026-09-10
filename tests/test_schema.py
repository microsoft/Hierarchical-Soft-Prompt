from hierarchical_soft_prompt.schema import USPTO_ORD_SPEC
from hierarchical_soft_prompt.schema_utils import dfs_keys, dfs_leaf_keys


def test_schema_has_nested_keys_and_leaves():
    keys = dfs_keys(USPTO_ORD_SPEC.schema)
    leaves = dfs_leaf_keys(USPTO_ORD_SPEC.schema)

    assert ("inputs", 0) in keys
    assert ("inputs.reactants.amount.mass.value", 4) in leaves
    assert ("outcomes.products.yield.value", 3) in leaves
    assert len(keys) > len(leaves)


def test_schema_spec_accessors_are_consistent():
    assert USPTO_ORD_SPEC.get_key_paths() == [
        path for path, _ in USPTO_ORD_SPEC.get_keys_with_depth()
    ]
    assert USPTO_ORD_SPEC.get_max_depth() >= 4
