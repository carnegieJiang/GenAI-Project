import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results_from_folders(root_dir):
    root_dir = Path(root_dir)
    rows = []

    for summary_path in root_dir.glob("*/*/*_summary.json"):
        method = summary_path.parent.parent.name
        guidance_folder = summary_path.parent.name

        match = re.search(r"guidance_([0-9.]+)", guidance_folder)
        if match is None:
            print(f"Skip: cannot parse guidance scale from {summary_path}")
            continue

        guidance_scale = float(match.group(1))

        with open(summary_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) == 0:
                continue
            data = data[0]

        row = dict(data)
        row["method"] = method
        row["guidance_scale"] = guidance_scale
        row["summary_path"] = str(summary_path)

        rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "num_samples",
        "avg_seconds_per_image",
        "avg_clip_prompt_alignment",
        "avg_clip_source_output_similarity",
        "avg_dino_content_preservation",
        "avg_lpips_content_preservation",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_metric_dots(
    df,
    metric,
    output_dir="plots",
    model_col="method",
    guidance_col="guidance_scale",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name, group in df.groupby(model_col):
        group = group.sort_values(guidance_col)

        plt.scatter(
            group[guidance_col],
            group[metric],
            label=model_name,
            s=70,
            alpha=0.85,
        )

        plt.plot(
            group[guidance_col],
            group[metric],
            alpha=0.45,
        )

    plt.xlabel("Guidance Scale")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Guidance Scale")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = output_dir / f"{metric}_vs_guidance.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    root_dir = "/home/ec2-user/GenAI-Project/results"  # change this to your actual root folder

    df = load_results_from_folders(root_dir)
    print(df)

    metrics = [
        "avg_seconds_per_image",
        "avg_clip_prompt_alignment",
        "avg_clip_source_output_similarity",
        "avg_dino_content_preservation",
        "avg_lpips_content_preservation",
    ]

    for metric in metrics:
        if metric in df.columns:
            plot_metric_dots(df, metric, output_dir="/home/ec2-user/GenAI-Project/plots")