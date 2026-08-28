"""
Soft prompt modules.

FlatSoftPrompt:
    Equal-token flat baseline with no schema structure.
    Same token count as hierarchical for fair comparison.

SchemaHierarchicalSoftPrompt:
    Tokens whose topology mirrors the JSON schema tree.
    DFS pre-order: global tokens first, then per-key tokens with depth encoding.

Both expose the same interface:
    forward() → Tensor[n_tokens, embed_dim]
    num_tokens: int (property)
    initialize_from_embeddings(tokenizer, embed_layer, hard_prompt_text)
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .schema_utils import dfs_keys


# ---------------------------------------------------------------------------
# Flat soft prompt baseline
# ---------------------------------------------------------------------------

class FlatSoftPrompt(nn.Module):
    """
    Flat learnable prefix — no schema structure.
    Used as an equal-token ablation baseline.
    """

    def __init__(self, n_tokens: int, embed_dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.tokens = nn.Parameter(torch.empty(n_tokens, embed_dim))
        nn.init.normal_(self.tokens, mean=0.0, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self.n_tokens

    def forward(self) -> torch.Tensor:
        """Returns (n_tokens, embed_dim)."""
        return self.tokens

    def initialize_from_embeddings(
        self,
        tokenizer,
        embed_layer: nn.Embedding,
        hard_prompt_text: str,
    ):
        """Initialize from mean-pooled hard prompt embeddings."""
        with torch.no_grad():
            prompt_ids = tokenizer(
                hard_prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(embed_layer.weight.device)

            prompt_embeds = embed_layer(prompt_ids)[0]  # (L, D)
            mean_embed = prompt_embeds.mean(dim=0)      # (D,)

            # Initialize all tokens to mean + small noise
            self.tokens.data = mean_embed.unsqueeze(0).expand(self.n_tokens, -1).clone()
            self.tokens.data += 0.01 * torch.randn_like(self.tokens)


# ---------------------------------------------------------------------------
# Schema-hierarchical soft prompt
# ---------------------------------------------------------------------------

class SchemaHierarchicalSoftPrompt(nn.Module):
    """
    Learnable soft prompt whose token topology mirrors the JSON schema tree.

    Structure (DFS pre-order):
        [G_1 ... G_kg]           <- Global tokens (kg tokens total)
        [K1_1 ... K1_t]          <- Per-key tokens for key 1 (t tokens)
        [K2_1 ... K2_t]          <- Per-key tokens for key 2
        ...
        [Km_1 ... Km_t]          <- Per-key tokens for key m (last in DFS order)

    Total tokens: kg + (num_schema_keys × t)

    Depth encoding: each per-key token group receives an additive depth embedding
    corresponding to that key's nesting depth in the schema tree.
    This encodes parent-child structure without changing token count.
    """

    def __init__(
        self,
        schema: dict,
        embed_dim: int,
        kg: int = 20,           # global tokens
        t: int = 2,             # tokens per schema key
        max_depth_override: Optional[int] = None,
        use_depth: bool = True,     # False = depth-embedding ablation
        key_order: str = "dfs",     # "shuffle" = key-ordering ablation
        order_seed: int = 42,       # seed for the shuffle ablation
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kg = kg
        self.t = t
        self.use_depth = use_depth
        self.key_order = key_order

        # DFS traversal — defines token ordering (FIXED unless ablated)
        self._keys_with_depth = dfs_keys(schema)   # list of (key_path, depth)
        if key_order == "shuffle":
            import random as _random
            _random.Random(order_seed).shuffle(self._keys_with_depth)
        self.num_keys = len(self._keys_with_depth)
        self.key_paths = [kp for kp, _ in self._keys_with_depth]
        self.key_depths = [d for _, d in self._keys_with_depth]

        max_d = max_depth_override or (max(self.key_depths) if self.key_depths else 0)

        # Learnable parameters
        self.global_tokens = nn.Parameter(torch.empty(kg, embed_dim))
        # per_key_tokens: shape (num_keys, t, embed_dim)
        self.per_key_tokens = nn.Parameter(torch.empty(self.num_keys, t, embed_dim))
        # Depth embedding table: (max_depth+1, embed_dim)
        self.depth_embeddings = nn.Embedding(max_d + 1, embed_dim)

        # Initialize
        nn.init.normal_(self.global_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.per_key_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.depth_embeddings.weight, mean=0.0, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self.kg + self.num_keys * self.t

    def forward(self) -> torch.Tensor:
        """
        Assemble and return the full soft prompt tensor: (num_tokens, embed_dim).

        Order: global tokens, then per-key tokens in DFS pre-order.
        Each per-key group has depth encoding added (additive, same dim as embed).
        """
        # Global tokens — no depth encoding
        global_part = self.global_tokens  # (kg, D)

        # Per-key tokens with depth encoding
        # depth_embeddings lookup: (num_keys, D)
        depths = torch.tensor(
            self.key_depths, dtype=torch.long, device=self.per_key_tokens.device
        )
        depth_emb = self.depth_embeddings(depths)           # (num_keys, D)
        depth_emb = depth_emb.unsqueeze(1)                  # (num_keys, 1, D)

        # per_key_tokens (+ depth encoding unless ablated via use_depth=False)
        keyed = self.per_key_tokens + (depth_emb if self.use_depth else 0)  # (num_keys, t, D)
        keyed = keyed.reshape(self.num_keys * self.t, self.embed_dim)  # (num_keys*t, D)

        # Concatenate: global first (DFS: parent before children), then per-key in DFS order
        return torch.cat([global_part, keyed], dim=0)       # (kg + num_keys*t, D)

    def initialize_from_embeddings(
        self,
        tokenizer,
        embed_layer: nn.Embedding,
        hard_prompt_text: str,
    ):
        """
        Semantic initialization:
        - Global tokens: mean-pooled embeddings of hard prompt text
        - Per-key tokens: embedding of the leaf key name string

        This gives each token a semantically meaningful starting point
        rather than random noise, which accelerates convergence.
        """
        device = embed_layer.weight.device

        with torch.no_grad():
            # --- Global tokens from hard prompt ---
            prompt_ids = tokenizer(
                hard_prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)

            prompt_embeds = embed_layer(prompt_ids)[0]  # (L, D)
            mean_embed = prompt_embeds.mean(dim=0)       # (D,)

            self.global_tokens.data = (
                mean_embed.unsqueeze(0).expand(self.kg, -1).clone()
                + 0.01 * torch.randn(self.kg, self.embed_dim, device=device)
            )

            # --- Per-key tokens from key name ---
            for i, (key_path, _) in enumerate(self._keys_with_depth):
                # Use the leaf key name (last component)
                key_name = key_path.split(".")[-1]
                key_ids = tokenizer(
                    key_name,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(device)

                if key_ids.shape[1] == 0:
                    # Fallback: use mean embed for unknown tokens
                    key_mean = mean_embed
                else:
                    key_embeds = embed_layer(key_ids)[0]  # (L_key, D)
                    key_mean = key_embeds.mean(dim=0)     # (D,)

                self.per_key_tokens.data[i] = (
                    key_mean.unsqueeze(0).expand(self.t, -1).clone()
                    + 0.01 * torch.randn(self.t, self.embed_dim, device=device)
                )
