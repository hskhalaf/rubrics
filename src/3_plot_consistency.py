import argparse
import glob
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

RUBRIC_BASE = "/n/home04/aoesterling/rubrics/rubric_items/speceval"
LABEL_WIDTH = 60  # characters before wrapping criterion text


def load_rubric_labels(rubric_base):
    """Returns {alias: short_criterion_label} from all YAML files under rubric_base."""
    labels = {}
    for path in glob.glob(os.path.join(rubric_base, "**", "*.yaml"), recursive=True):
        with open(path) as f:
            data = yaml.safe_load(f)
        alias = data.get("alias", os.path.splitext(os.path.basename(path))[0])
        criterion = data.get("criterion", alias)
        short = criterion.split("\n")[0].split(". ")[0].strip()
        labels[alias] = short
    return labels


def parse_plot_title(data_path):
    """Extract model and provider from filename like speceval_<provider>_<model...>.jsonl"""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    parts = stem.split("_")
    # Expected pattern: speceval_<provider>_<model_parts...>[_cot]
    if len(parts) >= 3:
        provider = parts[1]
        model_parts = parts[2:]
        if model_parts[-1] in ("cot", "nocot"):
            model_parts = model_parts[:-1]
        model = "_".join(model_parts)
        return f"{provider} / {model}"
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to judged .jsonl file")
    args = parser.parse_args()

    rubric_labels = load_rubric_labels(RUBRIC_BASE)

    df = pd.read_json(args.data_path, lines=True)

    # Build per-prompt DataFrame: variance of the 5 scores for the alias rubric item only
    records = []
    for _, row in df.iterrows():
        for alias, judgments in row["judge"].items():
            scores = []
            for j in judgments:
                try:
                    content = j["content"]
                    m = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
                    if m:
                        scores.append(float(json.loads(m.group())["score"]))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            if scores:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                records.append({
                    "prompt_id": row["prompt_id"],
                    "spec": row["spec"],
                    "model": row["model"],
                    "rubric_alias": alias,
                    "avg_score": mean,
                    "variance": variance,
                })
    print(records)
    long_df = pd.DataFrame(records)
    print(long_df.columns)

    # Per-prompt variance is already one row per (prompt, alias); just rename for clarity
    per_prompt = long_df[["prompt_id", "rubric_alias", "variance"]].rename(
        columns={"variance": "mean_variance"}
    )

    # Average per-prompt variance per rubric_alias across prompts
    per_alias_consistency = (
        per_prompt.groupby("rubric_alias")["mean_variance"]
        .mean()
        .reset_index()
        .rename(columns={"mean_variance": "inconsistency"})
        .sort_values("inconsistency", ascending=False)
    )
    per_alias_consistency["label"] = per_alias_consistency["rubric_alias"].map(
        lambda a: "\n".join(textwrap.wrap(rubric_labels.get(a, a), LABEL_WIDTH))
    )

    overall = per_prompt["mean_variance"].mean()
    print(f"Overall mean variance across all prompts: {overall:.4f}")
    print()
    print(per_alias_consistency[["label", "inconsistency"]].to_string(index=False))

    # Plot
    n = len(per_alias_consistency)
    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.6)))
    sns.barplot(data=per_alias_consistency, x="inconsistency", y="label", ax=ax, orient="h")
    ax.set_xlabel("Mean Score Variance across Judge Samples (averaged over prompts)", fontsize=11)
    ax.set_ylabel("")
    plot_label = parse_plot_title(args.data_path)
    ax.set_title(f"Judge Inconsistency per Rubric Alias — {plot_label}\n(variance over samples, averaged over prompts)", fontsize=13)
    ax.set_xlim(0, None)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    out_path = os.path.join("rubric_consistency_{stem}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
