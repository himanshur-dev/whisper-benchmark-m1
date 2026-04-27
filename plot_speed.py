#!/usr/bin/env python3
"""
Generate speed benchmark graphs from per-model summary CSVs.

Outputs:
    graphs/speed/runtime_by_model_clip.png
    graphs/speed/wer_by_model_clip.png

Usage:
    python plot_speed.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SOURCE_DIR = Path("outputs/speed")
GRAPH_DIR = Path("graphs/speed")

CLIP_ORDER = ["small", "medium", "large", "xlarge"]
MODEL_ORDER = ["openai/small", "mlx/small", "faster/small-int8", "distil/small"]
MODEL_LABELS = {
    "openai/small": "openai/small",
    "mlx/small": "mlx/small",
    "faster/small-int8": "faster/small-int8",
    "distil/small": "distil/small",
}
MARKERS = {
    "openai/small": "o",
    "mlx/small": "s",
    "faster/small-int8": "^",
    "distil/small": "D",
}


def load_data() -> pd.DataFrame:
    summary_paths = [SOURCE_DIR / f"{model.replace('/', '-')}_summary.csv" for model in MODEL_ORDER]
    frames = [pd.read_csv(path) for path in summary_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No per-model summary CSVs found.")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["clip"] = pd.Categorical(df["clip"], categories=CLIP_ORDER, ordered=True)
    df = df.sort_values(["model", "clip"]).reset_index(drop=True)
    return df


def plot_runtime(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))
    x_positions = list(range(len(CLIP_ORDER)))

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("clip")
        if sub.empty:
            continue
        ax.plot(
            x_positions,
            sub["inference_time_s"],
            marker=MARKERS[model],
            linewidth=2,
            markersize=7,
            label=MODEL_LABELS[model],
        )

    ax.set_title("Runtime by Model and Clip", fontsize=13)
    ax.set_xlabel("Audio clip size")
    ax.set_ylabel("Wall-clock runtime (seconds)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CLIP_ORDER, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = GRAPH_DIR / "runtime_by_model_clip.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_wer(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))
    bar_width = 0.18
    model_offsets = {
        "openai/small": -1.5 * bar_width,
        "mlx/small": -0.5 * bar_width,
        "faster/small-int8": 0.5 * bar_width,
        "distil/small": 1.5 * bar_width,
    }
    clip_positions = list(range(len(CLIP_ORDER)))

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].set_index("clip").reindex(CLIP_ORDER)
        heights = sub["wer"].tolist()
        positions = [x + model_offsets[model] for x in clip_positions]
        ax.bar(positions, heights, width=bar_width, label=MODEL_LABELS[model])

    ax.set_title("WER by Model and Clip", fontsize=13)
    ax.set_xlabel("Audio clip size")
    ax.set_ylabel("Word Error Rate")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(clip_positions)
    ax.set_xticklabels(CLIP_ORDER, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = GRAPH_DIR / "wer_by_model_clip.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    try:
        df = load_data()
    except FileNotFoundError:
        print("No per-model summaries found in outputs/speed/. Run the speed benchmark scripts first.")
        return

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} rows for models: {sorted(df['model'].unique())}\n")

    plot_runtime(df)
    plot_wer(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
