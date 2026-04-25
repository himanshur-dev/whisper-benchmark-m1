#!/usr/bin/env python3
"""
Load benchmark results and generate analysis plots.

Usage:
    python analyze.py                          # all plots, all results
    python analyze.py --models openai/small    # filter by model
    python analyze.py --output results/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str = "results/raw") -> pd.DataFrame:
    records = []
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(f.read_text())
            # flatten noise_params into top-level columns
            for k, v in data.pop("noise_params", {}).items():
                data[f"noise_{k}"] = v
            data.pop("metrics_samples", None)
            records.append(data)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_timeseries(results_dir: str = "results/raw") -> dict:
    """Returns {file_stem: list_of_sample_dicts}."""
    out = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        data = json.loads(f.read_text())
        samples = data.get("metrics_samples", [])
        if samples:
            out[f.stem] = {"samples": samples, "meta": {
                "model": data.get("model", ""),
                "noise_type": data.get("noise_type", ""),
                "pipeline": data.get("pipeline", ""),
                "clip": data.get("clip", ""),
            }}
    return out


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_wer_vs_snr(df: pd.DataFrame, output_dir: Path):
    for noise_type in ("gaussian", "babble"):
        sub = df[df["noise_type"] == noise_type]
        snr_col = "noise_snr_db"
        if sub.empty or snr_col not in sub.columns:
            continue

        pipelines = sub["pipeline"].unique()
        ncols = min(len(pipelines), 3)
        nrows = int(np.ceil(len(pipelines) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(f"WER vs SNR — {noise_type} noise", fontsize=12)

        for idx, pipeline in enumerate(pipelines):
            ax = axes[idx // ncols][idx % ncols]
            psub = sub[sub["pipeline"] == pipeline]
            for model in sorted(psub["model"].unique()):
                pts = psub[psub["model"] == model].groupby(snr_col)["wer"].mean()
                ax.plot(pts.index, pts.values, marker="o", label=model, linewidth=1.5)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("WER")
            ax.set_title(f"Pipeline: {pipeline}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()

        # hide unused axes
        for idx in range(len(pipelines), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        path = output_dir / f"wer_vs_snr_{noise_type}.png"
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


def plot_wer_vs_rt60(df: pd.DataFrame, output_dir: Path):
    sub = df[df["noise_type"] == "rir"]
    if sub.empty or "noise_rt60" not in sub.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in sorted(sub["model"].unique()):
        pts = sub[sub["model"] == model].groupby("noise_rt60")["wer"].mean()
        ax.plot(pts.index, pts.values, marker="o", label=model, linewidth=1.5)
    ax.set_xlabel("RT60 (s)")
    ax.set_ylabel("WER")
    ax.set_title("WER vs Reverberation (RIR noise)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "wer_vs_rt60.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_rtf_by_model(df: pd.DataFrame, output_dir: Path):
    mean_rtf = df.groupby("model")["rtf"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(mean_rtf.index, mean_rtf.values, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="Real-time (RTF=1)")
    ax.set_xlabel("Real-Time Factor (lower = faster)")
    ax.set_title("Mean RTF by Model")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, mean_rtf.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}×", va="center", fontsize=8)
    plt.tight_layout()
    path = output_dir / "rtf_by_model.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_memory_by_model(df: pd.DataFrame, output_dir: Path):
    if "peak_memory_mb" not in df.columns:
        return
    mem = df.groupby("model")["peak_memory_mb"].max().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(mem.index, mem.values, color="coral")
    ax.set_xlabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage by Model")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, mem.values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} MB", va="center", fontsize=8)
    plt.tight_layout()
    path = output_dir / "memory_by_model.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_wer_by_pipeline(df: pd.DataFrame, output_dir: Path):
    pivot = df.groupby(["pipeline", "model"])["wer"].mean().unstack(fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, alpha=0.85, width=0.7)
    ax.set_xlabel("Preprocessing Pipeline")
    ax.set_ylabel("Mean WER")
    ax.set_title("WER by Preprocessing Pipeline and Model")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = output_dir / "wer_by_pipeline.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_error_breakdown(df: pd.DataFrame, output_dir: Path):
    """Substitution / Deletion / Insertion breakdown per model."""
    needed = {"substitutions", "deletions", "insertions", "ref_word_count"}
    if not needed.issubset(df.columns):
        return

    agg = df.groupby("model")[["substitutions", "deletions", "insertions", "ref_word_count"]].sum()
    agg["sub_pct"] = agg["substitutions"] / agg["ref_word_count"] * 100
    agg["del_pct"] = agg["deletions"] / agg["ref_word_count"] * 100
    agg["ins_pct"] = agg["insertions"] / agg["ref_word_count"] * 100
    agg = agg.sort_values("sub_pct")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(agg))
    width = 0.25
    ax.barh(x - width, agg["sub_pct"], width, label="Substitutions", color="steelblue")
    ax.barh(x,         agg["del_pct"], width, label="Deletions",     color="coral")
    ax.barh(x + width, agg["ins_pct"], width, label="Insertions",    color="seagreen")
    ax.set_yticks(x)
    ax.set_yticklabels(agg.index)
    ax.set_xlabel("% of reference words")
    ax.set_title("WER Error Breakdown by Model")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "error_breakdown.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_gpu_timeseries(results_dir: str, output_dir: Path, n_runs: int = 8):
    ts = load_timeseries(results_dir)
    if not ts:
        return

    keys = list(ts)[:n_runs]
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.8 * len(keys)), squeeze=False)
    fig.suptitle("GPU & CPU Utilization During Inference", fontsize=11)

    for ax, key in zip(axes[:, 0], keys):
        samples = ts[key]["samples"]
        meta = ts[key]["meta"]
        t   = [s["t"] for s in samples]
        gpu = [s["gpu_pct"] for s in samples]
        cpu = [s["cpu_pct"] for s in samples]
        ax.plot(t, gpu, label="GPU %", color="steelblue", linewidth=1.5)
        ax.plot(t, cpu, label="CPU %", color="coral", alpha=0.8, linewidth=1.2)
        ax.set_ylim(0, 108)
        ax.set_ylabel("%")
        title = f"{meta['model']} | {meta['noise_type']} | {meta['pipeline']} | {meta['clip']}"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    path = output_dir / "gpu_timeseries.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs : {len(df)}")
    print(f"Models     : {sorted(df['model'].unique())}")
    print(f"Noise types: {sorted(df['noise_type'].unique())}")
    print(f"Pipelines  : {sorted(df['pipeline'].unique())}")
    print()
    cols = [c for c in ("wer", "cer", "rtf", "peak_memory_mb", "mean_gpu_pct") if c in df.columns]
    print(df.groupby("model")[cols].mean().round(4).to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--output", default="results/plots")
    parser.add_argument("--models", nargs="+", help="Filter to specific models")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)
    if df.empty:
        print(f"No results found in {args.results}. Run run_benchmark.py first.")
        return

    if args.models:
        df = df[df["model"].isin(args.models)]

    print_summary(df)
    print("Generating plots ...")

    plot_wer_vs_snr(df, output_dir)
    plot_wer_vs_rt60(df, output_dir)
    plot_rtf_by_model(df, output_dir)
    plot_memory_by_model(df, output_dir)
    plot_wer_by_pipeline(df, output_dir)
    plot_error_breakdown(df, output_dir)
    plot_gpu_timeseries(args.results, output_dir)

    csv_path = Path("results/summary.csv")
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV : {csv_path}")
    print(f"Plots saved : {output_dir}/")


if __name__ == "__main__":
    main()
