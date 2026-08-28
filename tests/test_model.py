import torch
from transformers import GPT2Config, GPT2LMHeadModel

from hierarchical_soft_prompt.model import PromptTunedModel
from hierarchical_soft_prompt.soft_prompt import FlatSoftPrompt


def test_prompt_model_forward_and_generate_with_inputs_embeds():
    config = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = PromptTunedModel(
        GPT2LMHeadModel(config),
        soft_prompt=FlatSoftPrompt(n_tokens=2, embed_dim=16),
    )
    input_ids = torch.tensor([[1, 3, 4, 5]])
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 3, 4, 5]])

    output = model(input_ids, attention_mask, labels)
    generated = model.generate(
        input_ids,
        torch.ones_like(input_ids),
        max_new_tokens=2,
    )

    assert torch.isfinite(output.loss)
    assert generated.shape == (1, 2)
