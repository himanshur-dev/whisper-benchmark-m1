#!/usr/bin/env python3
"""
Plot gaussian benchmark results from outputs/guassian/summary.csv.
Saves outputs/guassian/gaussian_benchmark.png.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_PATH = "outputs/guassian/summary.csv"
OUT_PATH = "graphs/gaussian/gaussian_benchmark.png"

PIPELINE_ORDER = ["none", "rms", "vad", "spectral"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-", "--"]

def main():
    from pathlib import Path
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df[df["pipeline"].isin(PIPELINE_ORDER)]
    df["snr_db"] = df["snr_db"].astype(int)
    snr_vals = sorted(df["snr_db"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Gaussian Noise", fontsize=13, fontweight="bold")

    for pipe, color, marker, ls in zip(PIPELINE_ORDER, COLORS, MARKERS, LINESTYLES):
        sub = df[df["pipeline"] == pipe].sort_values("snr_db")
        ax.plot(sub["snr_db"], sub["wer"] * 100,
                color=color, marker=marker, linestyle=ls, linewidth=2, markersize=7, label=pipe)

    ax.set_xlabel("Gaussian SNR (dB)", fontsize=11)
    ax.set_ylabel("WER (%)", fontsize=11)
    ax.set_xticks(snr_vals)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, title="Pipeline", title_fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
