#!/usr/bin/env python3
"""
Plot RIR benchmark results from outputs/rir/summary.csv.
Saves graphs/rir/rir_benchmark.png.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

CSV_PATH = "outputs/rir/summary.csv"
OUT_PATH = "graphs/rir/rir_benchmark.png"

PIPELINE_ORDER = ["none", "rms", "vad", "spectral"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-", "--"]


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df[df["pipeline"].isin(PIPELINE_ORDER)]
    rt60_vals = sorted(df["rt60"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Reverb (RIR)", fontsize=13, fontweight="bold")

    for pipe, color, marker, ls in zip(PIPELINE_ORDER, COLORS, MARKERS, LINESTYLES):
        sub = df[df["pipeline"] == pipe].sort_values("rt60")
        ax.plot(sub["rt60"], sub["wer"] * 100,
                color=color, marker=marker, linestyle=ls, linewidth=2, markersize=7, label=pipe)

    ax.set_xlabel("RT60 (seconds)", fontsize=11)
    ax.set_ylabel("WER (%)", fontsize=11)
    ax.set_xticks(rt60_vals)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, title="Pipeline", title_fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
