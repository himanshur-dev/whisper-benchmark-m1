#!/usr/bin/env python3
"""
Benchmark all models against all room sizes at RT60=2.0s.
Tests baseline vs RMS normalization for each model × room × clip combination.

Usage:
    python model_room_benchmark.py
    python model_room_benchmark.py --fresh   # ignore saved results, re-run everything
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from benchmark.evaluate import evaluate
from benchmark.models import MODEL_REGISTRY, load_model
from benchmark.noise import add_rir
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000
CLIPS = ["small", "medium", "large"]
CLIPS_DIR = Path("audio/clips")
TX_DIR = Path("audio/transcripts")
RESULTS_DIR = Path("results/model_room")
RT60 = 2.0

ROOMS = {
    "small_office": [3.0, 3.0, 2.5],
    "medium_room":  [6.0, 5.0, 3.0],
    "large_office": [10.0, 8.0, 4.0],
    "lecture_hall": [15.0, 12.0, 5.0],
}

PIPELINES = {
    "baseline": [],
    "rms":      ["rms"],
}


def load_clip(name):
    audio, _ = librosa.load(str(CLIPS_DIR / f"{name}.wav"), sr=SR, mono=True)
    return audio.astype(np.float32)


def result_path(model_key, room_name, pipeline_name, clip_name):
    safe = model_key.replace("/", "-")
    return RESULTS_DIR / f"{safe}_{room_name}_{pipeline_name}_{clip_name}.json"


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
        "rtf":        round(ev.rtf, 4),
        "infer_s":    round(result.inference_time_s, 3),
        "hypothesis": result.text,
        **stats,
    }


def print_summary(records):
    df = pd.DataFrame(records)

    # avg WER per model × room × pipeline
    pivot = df.groupby(["model", "room", "pipeline"])["wer"].mean().unstack("pipeline")
    if "baseline" in pivot.columns and "rms" in pivot.columns:
        pivot["delta"] = (pivot["rms"] - pivot["baseline"]).round(4)
        pivot["improvement"] = pivot["delta"].apply(
            lambda d: f"{d:+.3f}" + (" ✓" if d < -0.005 else "")
        )

    W = 82
    print("\n" + "=" * W)
    print(f"  ALL MODELS × ROOM SIZES — RT60={RT60}s  (avg WER across clips)")
    print("=" * W)
    print(f"{'model':<22} {'room':<18} {'baseline':>10} {'rms':>10} {'delta':>10}")
    print("-" * W)
    for (model, room), row in pivot.iterrows():
        b = row.get("baseline", float("nan"))
        r = row.get("rms", float("nan"))
        d = row.get("delta", float("nan"))
        marker = " ✓" if d < -0.005 else ""
        print(f"{model:<22} {room:<18} {b:>10.4f} {r:>10.4f} {d:>+10.4f}{marker}")
    print("=" * W)

    # per-model summary: avg delta across all rooms
    print("\nAverage RMS delta per model (across all rooms and clips):")
    model_summary = df.groupby(["model", "pipeline"])["wer"].mean().unstack("pipeline")
    if "baseline" in model_summary.columns and "rms" in model_summary.columns:
        model_summary["delta"] = (model_summary["rms"] - model_summary["baseline"]).round(4)
        model_summary = model_summary.sort_values("delta")
        print(f"\n{'model':<22} {'baseline':>10} {'rms':>10} {'delta':>10}")
        print("-" * 55)
        for model, row in model_summary.iterrows():
            marker = " ✓" if row["delta"] < -0.005 else ""
            print(f"{model:<22} {row['baseline']:>10.4f} {row['rms']:>10.4f} {row['delta']:>+10.4f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-run all conditions")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODEL_REGISTRY.keys())
    total = len(models) * len(ROOMS) * len(PIPELINES) * len(CLIPS)
    print(f"{len(models)} models × {len(ROOMS)} rooms × {len(PIPELINES)} pipelines × {len(CLIPS)} clips = {total} runs\n")

    # pre-generate all noisy signals (shared across models)
    print("Generating RIRs ...")
    noisy = {}
    for room_name, room_dim in ROOMS.items():
        print(f"  {room_name} {room_dim}")
        noisy[room_name] = {
            clip: add_rir(load_clip(clip), RT60, room_dim=room_dim)
            for clip in CLIPS
        }
    print()

    records = []
    done = 0

    for model_key in models:
        print(f"{'='*60}")
        print(f"Loading: {model_key}")
        t_load = time.perf_counter()
        try:
            model = load_model(model_key)
        except Exception as e:
            print(f"  FAILED: {e}\n")
            continue
        load_time = round(time.perf_counter() - t_load, 2)
        print(f"  Ready in {load_time}s\n")

        for room_name in ROOMS:
            for pipeline_name, steps in PIPELINES.items():
                for clip in CLIPS:
                    rpath = result_path(model_key, room_name, pipeline_name, clip)

                    if not args.fresh and rpath.exists():
                        record = json.loads(rpath.read_text())
                        records.append(record)
                        done += 1
                        print(f"  [skip] {model_key} | {room_name} | {pipeline_name} | {clip}")
                        continue

                    reference = (TX_DIR / f"{clip}.txt").read_text().strip()
                    try:
                        result = run_one(model, noisy[room_name][clip], reference, steps)
                    except Exception as e:
                        print(f"  [error] {e}")
                        continue

                    record = {
                        "model":      model_key,
                        "room":       room_name,
                        "pipeline":   pipeline_name,
                        "clip":       clip,
                        "rt60":       RT60,
                        **result,
                    }
                    rpath.write_text(json.dumps(record, indent=2))
                    records.append(record)
                    done += 1
                    print(f"  [{done:>3}/{total}] {model_key:<22} | {room_name:<18} | "
                          f"{pipeline_name:<10} | {clip:<7}  WER={result['wer']:.3f}  RTF={result['rtf']:.3f}x")

    print_summary(records)

    csv_path = Path("results/model_room_summary.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
