"""
Create backbone-free interpretability views of a learned USPTO-ORD soft prompt.

The analysis uses only ``soft_prompt.pt`` and the selected schema. It does not
load training data or the language-model backbone.

Outputs
-------
prompt_key_metrics.csv
    Per-schema-node geometry and salience metrics.
prompt_summary.json / prompt_summary.md
    Aggregate statistics and ranked schema nodes.
prompt_wordclouds.{png,pdf}
    One field-name cloud per top-level schema subtree. Word size is a
    transparent heuristic combining slot norm and slot distinctiveness.
prompt_atlas.{png,pdf}
    Four panels:
      1. PCA displacement from learned slot content to depth-aware slot vectors.
      2. Pairwise cosine similarity in DFS order.
      3. Norm, distinctiveness, and depth influence by schema depth.
      4. Top-level subtree similarity/coherence.

These are descriptive probes, not causal attribution. In particular, the word
clouds show which schema-node slots are geometrically prominent; they do not
decode a soft prompt into natural-language instructions.

Example
-------
python scripts/analyze_prompt_atlas.py \
    --checkpoint outputs/checkpoints/sp_uspto_seed42_best \
    --output-dir outputs/interpretability/uspto_prompt_atlas \
    --title "USPTO-ORD SP-only (Qwen3-4B, seed 42)"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from wordcloud import WordCloud, get_single_color_func

from hierarchical_soft_prompt.schema import USPTO_ORD_SPEC
from hierarchical_soft_prompt.soft_prompt import SchemaHierarchicalSoftPrompt


GROUP_COLORS = {
    "inputs": "#4C78A8",
    "conditions": "#F58518",
    "outcomes": "#54A24B",
    "workups": "#B279A2",
    "flight": "#4C78A8",
    "hotel": "#F58518",
    "transport": "#54A24B",
}
FALLBACK_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
]
DEPTH_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def load_prompt_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    path = checkpoint / "soft_prompt.pt" if checkpoint.is_dir() else checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Missing soft-prompt checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"global_tokens", "per_key_tokens", "depth_embeddings.weight"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Checkpoint is missing tensors: {missing}")
    return {name: tensor.float() for name, tensor in state.items()}


def percentile_scores(values: torch.Tensor) -> torch.Tensor:
    """Return stable percentile ranks in (0, 1], preserving tied ordering."""
    values = values.detach().cpu()
    order = torch.argsort(values, stable=True)
    scores = torch.empty_like(values, dtype=torch.float32)
    scores[order] = torch.arange(
        1,
        values.numel() + 1,
        dtype=torch.float32,
    ) / values.numel()
    return scores


def off_diagonal_mean(matrix: torch.Tensor) -> float:
    if matrix.shape[0] < 2:
        return float("nan")
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return float(matrix[mask].mean().item())


def compact_label(key_path: str) -> str:
    """Make a unique, readable word-cloud phrase from a dotted key path."""
    parts = key_path.split(".")
    if len(parts) > 1:
        parts = parts[1:]
    cleaned = []
    for part in parts or key_path.split("."):
        part = CAMEL_BOUNDARY.sub(" ", part).replace("_", " ")
        cleaned.append(part)
    return " ".join(cleaned)


def group_palette(groups: list[str]) -> dict[str, str]:
    return {
        group: GROUP_COLORS.get(group, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        for i, group in enumerate(groups)
    }


def compute_prompt_geometry(
    state: dict[str, torch.Tensor],
    task: str,
    *,
    key_order: str = "dfs",
    order_seed: int = 42,
    use_depth: bool = True,
) -> tuple[list[dict], dict]:
    per_key = state["per_key_tokens"]
    globals_ = state["global_tokens"]
    depth_table = state["depth_embeddings.weight"]

    if task != "uspto_ord":
        raise ValueError(f"Unsupported public task: {task!r}")
    schema_spec = USPTO_ORD_SPEC
    template = SchemaHierarchicalSoftPrompt(
        schema=schema_spec.schema,
        embed_dim=per_key.shape[-1],
        kg=globals_.shape[0],
        t=per_key.shape[1],
        max_depth_override=depth_table.shape[0] - 1,
        key_order=key_order,
        order_seed=order_seed,
        use_depth=use_depth,
    )
    key_paths = list(template.key_paths)
    key_depths = torch.tensor(template.key_depths, dtype=torch.long)

    if per_key.shape[0] != len(key_paths):
        raise ValueError(
            f"Schema/checkpoint mismatch: checkpoint has {per_key.shape[0]} key "
            f"slots, but task={task!r} produces {len(key_paths)} keys."
        )
    if key_depths.max().item() >= depth_table.shape[0]:
        raise ValueError(
            f"Depth table has {depth_table.shape[0]} rows, but schema requires "
            f"depth {key_depths.max().item()}."
        )

    content_vectors = per_key.mean(dim=1)
    depth_vectors = depth_table[key_depths]
    visible_vectors = content_vectors + depth_vectors if use_depth else content_vectors

    normalized = F.normalize(visible_vectors, dim=-1)
    similarity = normalized @ normalized.T
    peer_mask = ~torch.eye(len(key_paths), dtype=torch.bool)
    peer_values = similarity.masked_fill(~peer_mask, float("-inf"))
    max_peer_cos = peer_values.max(dim=1).values
    mean_peer_cos = (
        similarity.masked_fill(~peer_mask, 0.0).sum(dim=1)
        / max(1, len(key_paths) - 1)
    )
    specialization = 1.0 - max_peer_cos

    global_similarity = normalized @ F.normalize(globals_, dim=-1).T
    mean_global_cos = global_similarity.mean(dim=1)
    max_global_cos = global_similarity.max(dim=1).values

    depth_influence = 1.0 - F.cosine_similarity(
        content_vectors,
        visible_vectors,
        dim=-1,
    )
    slot_norm = visible_vectors.norm(dim=-1)
    content_norm = content_vectors.norm(dim=-1)
    depth_norm = depth_vectors.norm(dim=-1)

    if per_key.shape[1] > 1:
        token_norm = F.normalize(per_key, dim=-1)
        intra_similarity = token_norm @ token_norm.transpose(1, 2)
        intra_mask = ~torch.eye(per_key.shape[1], dtype=torch.bool)
        intra_token_cos = intra_similarity[:, intra_mask].mean(dim=1)
    else:
        intra_token_cos = torch.ones(len(key_paths))

    groups = [path.split(".")[0] for path in key_paths]
    subtree_margin = torch.zeros(len(key_paths))
    for i, group in enumerate(groups):
        same = torch.tensor(
            [j != i and other == group for j, other in enumerate(groups)],
            dtype=torch.bool,
        )
        other = torch.tensor(
            [other != group for other in groups],
            dtype=torch.bool,
        )
        same_mean = similarity[i, same].mean() if same.any() else torch.tensor(0.0)
        other_mean = similarity[i, other].mean() if other.any() else torch.tensor(0.0)
        subtree_margin[i] = same_mean - other_mean

    norm_score = percentile_scores(slot_norm)
    specialization_score = percentile_scores(specialization)
    salience = torch.sqrt(norm_score * specialization_score)

    leaf_paths = set(schema_spec.get_leaf_key_paths())
    rows = []
    for i, key_path in enumerate(key_paths):
        rows.append(
            {
                "index": i,
                "key_path": key_path,
                "label": compact_label(key_path),
                "group": groups[i],
                "depth": int(key_depths[i].item()),
                "is_leaf": key_path in leaf_paths,
                "slot_norm": float(slot_norm[i].item()),
                "content_norm": float(content_norm[i].item()),
                "depth_norm": float(depth_norm[i].item()),
                "depth_influence": float(depth_influence[i].item()),
                "mean_peer_cos": float(mean_peer_cos[i].item()),
                "max_peer_cos": float(max_peer_cos[i].item()),
                "specialization": float(specialization[i].item()),
                "mean_global_cos": float(mean_global_cos[i].item()),
                "max_global_cos": float(max_global_cos[i].item()),
                "intra_token_cos": float(intra_token_cos[i].item()),
                "subtree_margin": float(subtree_margin[i].item()),
                "salience": float(salience[i].item()),
            }
        )

    ordered_groups = list(dict.fromkeys(groups))
    group_indices = {
        group: [i for i, value in enumerate(groups) if value == group]
        for group in ordered_groups
    }
    group_within_cosine = {}
    group_centroids = {}
    for group, indices in group_indices.items():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        block = similarity[index_tensor][:, index_tensor]
        group_within_cosine[group] = off_diagonal_mean(block)
        centroid = normalized[indices].mean(dim=0)
        group_centroids[group] = F.normalize(centroid, dim=0)

    group_centroid_cosine = {
        group_i: {
            group_j: float(
                torch.dot(group_centroids[group_i], group_centroids[group_j]).item()
            )
            for group_j in ordered_groups
        }
        for group_i in ordered_groups
    }

    within_values = []
    between_values = []
    for i in range(len(key_paths)):
        for j in range(i + 1, len(key_paths)):
            target = within_values if groups[i] == groups[j] else between_values
            target.append(float(similarity[i, j].item()))

    summary = {
        "task": task,
        "num_keys": len(key_paths),
        "num_global_tokens": int(globals_.shape[0]),
        "tokens_per_key": int(per_key.shape[1]),
        "embedding_dim": int(per_key.shape[2]),
        "groups": ordered_groups,
        "group_within_cosine": group_within_cosine,
        "group_centroid_cosine": group_centroid_cosine,
        "mean_pairwise_cosine": off_diagonal_mean(similarity),
        "mean_within_group_cosine": float(np.mean(within_values)),
        "mean_between_group_cosine": float(np.mean(between_values)),
        "within_minus_between_cosine": float(
            np.mean(within_values) - np.mean(between_values)
        ),
        "mean_global_affinity": float(mean_global_cos.mean().item()),
        "mean_depth_influence": float(depth_influence.mean().item()),
        "mean_intra_token_cosine": float(intra_token_cos.mean().item()),
        "_content_vectors": content_vectors,
        "_visible_vectors": visible_vectors,
        "_global_tokens": globals_,
        "_similarity": similarity,
    }
    return rows, summary


def write_metrics(rows: list[dict], output_dir: Path) -> None:
    columns = [
        "index",
        "key_path",
        "label",
        "group",
        "depth",
        "is_leaf",
        "slot_norm",
        "content_norm",
        "depth_norm",
        "depth_influence",
        "mean_peer_cos",
        "max_peer_cos",
        "specialization",
        "mean_global_cos",
        "max_global_cos",
        "intra_token_cos",
        "subtree_margin",
        "salience",
    ]
    with (output_dir / "prompt_key_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ranked(rows: list[dict], metric: str, n: int = 10) -> list[dict]:
    return sorted(rows, key=lambda row: row[metric], reverse=True)[:n]


def checkpoint_label(checkpoint: Path) -> str:
    if checkpoint.is_dir():
        return checkpoint.name
    if checkpoint.name == "soft_prompt.pt" and checkpoint.parent.name:
        return checkpoint.parent.name
    return checkpoint.name


def write_summary(
    rows: list[dict],
    summary: dict,
    output_dir: Path,
    checkpoint: Path,
    title: str,
) -> None:
    public_summary = {
        key: value
        for key, value in summary.items()
        if not key.startswith("_")
    }
    public_summary["checkpoint"] = checkpoint_label(checkpoint)
    public_summary["title"] = title
    public_summary["top_salience"] = [
        {"key_path": row["key_path"], "value": row["salience"]}
        for row in ranked(rows, "salience")
    ]
    public_summary["top_specialization"] = [
        {"key_path": row["key_path"], "value": row["specialization"]}
        for row in ranked(rows, "specialization")
    ]
    public_summary["top_depth_influence"] = [
        {"key_path": row["key_path"], "value": row["depth_influence"]}
        for row in ranked(rows, "depth_influence")
    ]
    public_summary["top_global_affinity"] = [
        {"key_path": row["key_path"], "value": row["mean_global_cos"]}
        for row in ranked(rows, "mean_global_cos")
    ]

    with (output_dir / "prompt_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(public_summary, handle, indent=2)

    def table(metric: str, heading: str) -> str:
        lines = [f"### {heading}", "", "| schema node | value |", "|---|---:|"]
        for row in ranked(rows, metric):
            lines.append(f"| `{row['key_path']}` | {row[metric]:.4f} |")
        return "\n".join(lines)

    text = [
        f"# {title}",
        "",
        f"- Checkpoint label: `{checkpoint_label(checkpoint)}`",
        f"- Schema nodes: {summary['num_keys']}",
        f"- Global tokens: {summary['num_global_tokens']}",
        f"- Tokens per schema node: {summary['tokens_per_key']}",
        f"- Embedding dimension: {summary['embedding_dim']}",
        (
            "- Mean within/between-top-level cosine: "
            f"{summary['mean_within_group_cosine']:.4f} / "
            f"{summary['mean_between_group_cosine']:.4f} "
            f"(margin {summary['within_minus_between_cosine']:+.4f})"
        ),
        (
            "- Per-group within-subtree cosine: "
            + ", ".join(
                f"{group}={value:.4f}"
                for group, value in summary["group_within_cosine"].items()
            )
        ),
        f"- Mean global affinity: {summary['mean_global_affinity']:.4f}",
        f"- Mean depth influence (1 - cosine): {summary['mean_depth_influence']:.4f}",
        f"- Mean within-slot token cosine: {summary['mean_intra_token_cosine']:.4f}",
        "",
        (
            "> Interpretation note: salience is a descriptive geometric heuristic "
            "(percentile slot norm x percentile distinctiveness)^0.5. It is not "
            "gradient attribution, feature importance, or a causal score."
        ),
        "",
        table("salience", "Highest geometric salience"),
        "",
        table("specialization", "Most distinctive slots"),
        "",
        table("depth_influence", "Largest depth-embedding influence"),
        "",
        table("mean_global_cos", "Highest global-token affinity"),
        "",
    ]
    (output_dir / "prompt_summary.md").write_text(
        "\n".join(text),
        encoding="utf-8",
    )


def plot_wordclouds(
    rows: list[dict],
    output_dir: Path,
    title: str,
    *,
    leaves_only: bool,
) -> None:
    selected = [row for row in rows if row["is_leaf"] or not leaves_only]
    groups = list(dict.fromkeys(row["group"] for row in selected))
    palette = group_palette(groups)
    ncols = 2
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, group in zip(axes, groups):
        group_rows = [row for row in selected if row["group"] == group]
        frequencies = {
            row["label"]: 1.0 + 99.0 * row["salience"]
            for row in group_rows
        }
        cloud = WordCloud(
            width=1200,
            height=650,
            background_color="white",
            prefer_horizontal=0.85,
            relative_scaling=0.55,
            random_state=42,
            collocations=False,
            margin=3,
        ).generate_from_frequencies(frequencies)
        cloud.recolor(color_func=get_single_color_func(palette[group]))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(
            f"{group} ({len(group_rows)} schema nodes)",
            fontsize=14,
            color=palette[group],
            fontweight="bold",
        )
        ax.axis("off")

    for ax in axes[len(groups) :]:
        ax.axis("off")

    node_scope = "leaf fields" if leaves_only else "all schema nodes"
    fig.suptitle(
        f"{title}\nSchema-node word clouds ({node_scope}); "
        "size = geometric salience",
        fontsize=17,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Salience combines learned slot norm and distinctiveness. "
        "The labels are schema fields, not decoded latent tokens.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prompt_wordclouds.{extension}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_prompt_atlas(
    rows: list[dict],
    summary: dict,
    output_dir: Path,
    title: str,
    annotate_top: int,
) -> None:
    content = summary["_content_vectors"].numpy()
    visible = summary["_visible_vectors"].numpy()
    globals_ = summary["_global_tokens"].numpy()
    similarity = summary["_similarity"].numpy()
    groups = summary["groups"]
    palette = group_palette(groups)

    all_vectors = np.concatenate([content, visible, globals_], axis=0)
    coords = PCA(n_components=2).fit_transform(all_vectors)
    n_keys = len(rows)
    content_xy = coords[:n_keys]
    visible_xy = coords[n_keys : 2 * n_keys]
    global_xy = coords[2 * n_keys :]

    fig, axes = plt.subplots(2, 2, figsize=(17, 14))

    # Panel A: depth-aware displacement in PCA space.
    ax = axes[0, 0]
    for i, row in enumerate(rows):
        color = palette[row["group"]]
        ax.annotate(
            "",
            xy=visible_xy[i],
            xytext=content_xy[i],
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.18, "lw": 0.7},
        )
        marker = DEPTH_MARKERS[row["depth"] % len(DEPTH_MARKERS)]
        ax.scatter(
            visible_xy[i, 0],
            visible_xy[i, 1],
            s=28 + 55 * row["salience"],
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
        )
    ax.scatter(
        global_xy[:, 0],
        global_xy[:, 1],
        marker="*",
        s=60,
        color="#222222",
        alpha=0.45,
        label="global tokens",
    )
    top_rows = ranked(rows, "salience", annotate_top)
    for rank, row in enumerate(top_rows, start=1):
        i = row["index"]
        ax.annotate(
            str(rank),
            visible_xy[i],
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color=palette[row["group"]],
            fontweight="bold",
        )
    ax.text(
        0.015,
        0.015,
        "\n".join(
            f"{rank}. {row['label']}"
            for rank, row in enumerate(top_rows, start=1)
        ),
        transform=ax.transAxes,
        va="bottom",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.88,
        },
    )
    for group in groups:
        ax.scatter([], [], color=palette[group], label=group)
    ax.set_title(
        "A. Learned slot map\narrows: content-only -> depth-aware vector",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # Panel B: pairwise cosine similarity in DFS order.
    ax = axes[0, 1]
    image = ax.imshow(
        similarity,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    group_indices: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        group_indices[row["group"]].append(row["index"])
    centers = []
    labels = []
    for group in groups:
        indices = group_indices[group]
        centers.append((indices[0] + indices[-1]) / 2)
        labels.append(group)
        boundary = indices[-1] + 0.5
        ax.axvline(boundary, color="white", lw=1.2)
        ax.axhline(boundary, color="white", lw=1.2)
    ax.set_xticks(centers, labels, rotation=30, ha="right")
    ax.set_yticks(centers, labels)
    ax.set_title(
        "B. Slot cosine similarity in DFS order",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine")

    # Panel C: depth profile.
    ax = axes[1, 0]
    depth_values = sorted({row["depth"] for row in rows})
    norm_means = []
    norm_stds = []
    specialization_means = []
    influence_means = []
    for depth in depth_values:
        depth_rows = [row for row in rows if row["depth"] == depth]
        norms = np.array([row["slot_norm"] for row in depth_rows])
        norm_means.append(norms.mean())
        norm_stds.append(norms.std())
        specialization_means.append(
            np.mean([row["specialization"] for row in depth_rows])
        )
        influence_means.append(
            np.mean([row["depth_influence"] for row in depth_rows])
        )
    bars = ax.bar(
        depth_values,
        norm_means,
        yerr=norm_stds,
        capsize=3,
        color="#4C78A8",
        alpha=0.75,
        label="slot norm",
    )
    ax.set_xlabel("schema depth")
    ax.set_ylabel("depth-aware slot norm", color="#4C78A8")
    ax.tick_params(axis="y", labelcolor="#4C78A8")
    ax.grid(axis="y", alpha=0.15)
    ax2 = ax.twinx()
    line1 = ax2.plot(
        depth_values,
        specialization_means,
        "o-",
        color="#E45756",
        label="distinctiveness",
    )[0]
    line2 = ax2.plot(
        depth_values,
        influence_means,
        "s--",
        color="#54A24B",
        label="depth influence",
    )[0]
    ax2.set_ylabel("cosine-derived score")
    ax2.legend(
        [bars, line1, line2],
        ["slot norm", "distinctiveness", "depth influence"],
        fontsize=8,
        frameon=False,
        loc="best",
    )
    ax.set_title(
        "C. Geometry by schema depth",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )

    # Panel D: top-level group coherence/similarity.
    ax = axes[1, 1]
    group_matrix = np.zeros((len(groups), len(groups)), dtype=np.float32)
    normalized_visible = F.normalize(summary["_visible_vectors"], dim=-1)
    centroids = {}
    for group in groups:
        indices = group_indices[group]
        centroid = normalized_visible[indices].mean(dim=0)
        centroids[group] = F.normalize(centroid, dim=0)
    for i, group_i in enumerate(groups):
        indices_i = group_indices[group_i]
        for j, group_j in enumerate(groups):
            if i == j:
                block = similarity[np.ix_(indices_i, indices_i)]
                group_matrix[i, j] = off_diagonal_mean(torch.from_numpy(block))
            else:
                group_matrix[i, j] = float(
                    torch.dot(centroids[group_i], centroids[group_j]).item()
                )
    image = ax.imshow(
        group_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    for i in range(len(groups)):
        for j in range(len(groups)):
            value = group_matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(value) > 0.55 else "black",
            )
    ax.set_title(
        "D. Top-level schema geometry\n"
        "diagonal = within-group coherence; off-diagonal = centroid cosine",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine")

    fig.suptitle(title, fontsize=18, y=0.995)
    fig.text(
        0.5,
        0.008,
        "Descriptive geometry of learned prompt parameters; not causal attribution.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"prompt_atlas.{extension}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create word clouds and geometry plots for a learned soft prompt."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory containing soft_prompt.pt, or the .pt file itself.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--title")
    parser.add_argument(
        "--key-order",
        "--key_order",
        dest="key_order",
        choices=["dfs", "shuffle"],
        default="dfs",
    )
    parser.add_argument("--order-seed", "--order_seed", dest="order_seed", type=int, default=42)
    parser.add_argument(
        "--no-depth",
        "--no_depth",
        dest="no_depth",
        action="store_true",
        help="Analyze a checkpoint whose forward path disabled depth embeddings.",
    )
    parser.add_argument(
        "--cloud-all-nodes",
        "--cloud_all_nodes",
        dest="cloud_all_nodes",
        action="store_true",
        help="Include intermediate schema nodes in word clouds (leaf fields only by default).",
    )
    parser.add_argument(
        "--annotate-top",
        "--annotate_top",
        dest="annotate_top",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = "uspto_ord"
    title = args.title or "USPTO-ORD learned soft-prompt atlas"

    state = load_prompt_state(checkpoint)
    rows, summary = compute_prompt_geometry(
        state,
        task,
        key_order=args.key_order,
        order_seed=args.order_seed,
        use_depth=not args.no_depth,
    )
    write_metrics(rows, output_dir)
    write_summary(rows, summary, output_dir, checkpoint, title)
    plot_wordclouds(
        rows,
        output_dir,
        title,
        leaves_only=not args.cloud_all_nodes,
    )
    plot_prompt_atlas(
        rows,
        summary,
        output_dir,
        title,
        annotate_top=args.annotate_top,
    )

    print(f"Wrote prompt interpretability artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
