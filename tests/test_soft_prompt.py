import torch

from hierarchical_soft_prompt.schema import USPTO_ORD_SPEC
from hierarchical_soft_prompt.soft_prompt import (
    FlatSoftPrompt,
    SchemaHierarchicalSoftPrompt,
)


def test_hierarchical_prompt_shape_matches_declared_token_count():
    prompt = SchemaHierarchicalSoftPrompt(
        USPTO_ORD_SPEC.schema,
        embed_dim=16,
        kg=3,
        t=2,
    )
    assert prompt().shape == (prompt.num_tokens, 16)


def test_flat_prompt_is_trainable():
    prompt = FlatSoftPrompt(n_tokens=5, embed_dim=8)
    assert prompt().shape == (5, 8)
    assert isinstance(prompt.tokens, torch.nn.Parameter)
