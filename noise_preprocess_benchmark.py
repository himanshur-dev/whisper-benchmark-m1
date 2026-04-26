#!/usr/bin/env python3
"""
Benchmark all combinations of noise condition × preprocessing pipeline
using openai/small. Results are saved incrementally to results/noise_preprocess/
and printed as a summary table at the end.

Usage:
    python noise_preprocess_benchmark.py
    python noise_preprocess_benchmark.py --fresh   # ignore saved results, re-run everything
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.noise import add_gaussian, add_rir
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000
CLIPS = ["small", "medium", "large"]
CLIPS_DIR = Path("audio/clips")
TX_DIR = Path("audio/transcripts")
RESULTS_DIR = Path("results/noise_preprocess")

NOISE_CONDITIONS = [
    {"type": "gaussian", "snr_db": 0,   "label": "gaussian_snr0"},
    {"type": "gaussian", "snr_db": 5,   "label": "gaussian_snr5"},
    {"type": "gaussian", "snr_db": 10,  "label": "gaussian_snr10"},
    {"type": "gaussian", "snr_db": 20,  "label": "gaussian_snr20"},
    {"type": "gaussian", "snr_db": 40,  "label": "gaussian_snr40"},
    {"type": "rir",      "rt60": 0.2,   "label": "rir_rt60_0.2"},
    {"type": "rir",      "rt60": 0.5,   "label": "rir_rt60_0.5"},
    {"type": "rir",      "rt60": 1.0,   "label": "rir_rt60_1.0"},
    {"type": "rir",      "rt60": 2.0,   "label": "rir_rt60_2.0"},
]

PIPELINES = {
    "baseline": [],
    "vad":      ["vad"],
    "spectral": ["spectral_gate"],
    "rms":      ["rms"],
}


def load_clip(name):
    audio, _ = librosa.load(str(CLIPS_DIR / f"{name}.wav"), sr=SR, mono=True)
    return audio.astype(np.float32)


def apply_noise(signal, ncfg, rng):
    if ncfg["type"] == "gaussian":
        return add_gaussian(signal, ncfg["snr_db"], rng=rng)
    if ncfg["type"] == "rir":
        return add_rir(signal, ncfg["rt60"])


def result_path(noise_label, pipeline_name, clip_name):
    return RESULTS_DIR / f"{noise_label}_{pipeline_name}_{clip_name}.json"


def run_one(model, signal, reference, steps):
    duration = len(signal) / SR
    processed = run_pipeline(signal, SR, steps)
    sampler = MetricsSampler(interval=0.2)
    sampler.start()
    result = model.transcribe(processed)
    samples = sampler.stop()
    stats = sampler.summary()
    ev = evaluate(result.text, reference, result.inference_time_s, duration)
    return {
        "wer":        round(ev.wer, 4),
        "cer":        round(ev.cer, 4),
        "subs":       ev.substitutions,
        "dels":       ev.deletions,
        "ins":        ev.insertions,
        "rtf":        round(ev.rtf, 4),
        "infer_s":    round(result.inference_time_s, 3),
        "hypothesis": result.text,
        "reference":  reference,
        **stats,
    }


def print_summary(records):
    df = pd.DataFrame(records)

    # average WER across clips for each noise × pipeline combination
    pivot = df.groupby(["noise_label", "pipeline"])["wer"].mean().unstack("pipeline")

    # order columns
    col_order = [c for c in ["baseline", "vad", "spectral", "rms"] if c in pivot.columns]
    pivot = pivot[col_order]

    # add a delta column showing best improvement over baseline
    if "baseline" in pivot.columns:
        best_other = pivot.drop(columns="baseline").min(axis=1)
        pivot["best_delta"] = (best_other - pivot["baseline"]).round(4)

    W = 80
    print("\n" + "=" * W)
    print("  NOISE × PREPROCESSING — avg WER across all clips  (openai/small)")
    print("  best_delta = lowest WER among [vad, spectral, rms] minus baseline")
    print("=" * W)
    print(pivot.round(4).to_string())
    print("=" * W)

    # also print per-clip breakdown for each noise condition
    print("\n\nPER-CLIP BREAKDOWN")
    for noise_label in df["noise_label"].unique():
        sub = df[df["noise_label"] == noise_label]
        print(f"\n  {noise_label}")
        print(f"  {'PIPELINE':<12} {'CLIP':<8} {'WER':>6} {'CER':>6} {'S/D/I':>9} {'RTF':>6}")
        print(f"  {'-'*55}")
        for pipeline in col_order:
            for clip in CLIPS:
                row = sub[(sub["pipeline"] == pipeline) & (sub["clip"] == clip)]
                if row.empty:
                    continue
                r = row.iloc[0]
                sdi = f"{int(r['subs'])}/{int(r['dels'])}/{int(r['ins'])}"
                print(f"  {pipeline:<12} {clip:<8} {r['wer']:>6.3f} {r['cer']:>6.3f} {sdi:>9} {r['rtf']:>6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Re-run all conditions, ignoring saved results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(NOISE_CONDITIONS) * len(PIPELINES) * len(CLIPS)
    print(f"Conditions: {len(NOISE_CONDITIONS)} noise × {len(PIPELINES)} pipelines × {len(CLIPS)} clips = {total} runs")

    print("\nLoading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    rng = np.random.default_rng(42)
    records = []
    done = 0

    for ncfg in NOISE_CONDITIONS:
        # pre-compute noisy signals for all clips under this noise condition
        noisy = {}
        for clip in CLIPS:
            signal = load_clip(clip)
            noisy[clip] = apply_noise(signal, ncfg, rng)

        for pipeline_name, steps in PIPELINES.items():
            for clip in CLIPS:
                rpath = result_path(ncfg["label"], pipeline_name, clip)

                if not args.fresh and rpath.exists():
                    record = json.loads(rpath.read_text())
                    records.append(record)
                    done += 1
                    print(f"  [skip] {ncfg['label']} | {pipeline_name} | {clip}")
                    continue

                reference = (TX_DIR / f"{clip}.txt").read_text().strip()
                result = run_one(model, noisy[clip], reference, steps)

                record = {
                    "noise_label":  ncfg["label"],
                    "noise_type":   ncfg["type"],
                    "noise_snr_db": ncfg.get("snr_db"),
                    "noise_rt60":   ncfg.get("rt60"),
                    "pipeline":     pipeline_name,
                    "clip":         clip,
                    **result,
                }
                rpath.write_text(json.dumps(record, indent=2))
                records.append(record)
                done += 1

                print(f"  [{done:>3}/{total}] {ncfg['label']:<20} | {pipeline_name:<10} | "
                      f"{clip:<7} WER={result['wer']:.3f}  RTF={result['rtf']:.3f}x")

    print_summary(records)

    csv_path = Path("results/noise_preprocess_summary.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
