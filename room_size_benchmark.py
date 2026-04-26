#!/usr/bin/env python3
"""
Test whether room size affects the RMS improvement at RT60=2.0s.

Runs baseline vs RMS normalization across four room sizes using openai/small.

Usage:
    python room_size_benchmark.py
"""

import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.noise import add_rir
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000
CLIPS = ["small", "medium", "large"]
CLIPS_DIR = Path("audio/clips")
TX_DIR = Path("audio/transcripts")
RT60 = 2.0

ROOMS = {
    "small_office":  [3.0, 3.0, 2.5],
    "medium_room":   [6.0, 5.0, 3.0],
    "large_office":  [10.0, 8.0, 4.0],
    "lecture_hall":  [15.0, 12.0, 5.0],
}

PIPELINES = {
    "baseline": [],
    "rms":      ["rms"],
}


def load_clip(name):
    audio, _ = librosa.load(str(CLIPS_DIR / f"{name}.wav"), sr=SR, mono=True)
    return audio.astype(np.float32)


def run_one(model, signal, reference, steps):
    duration = len(signal) / SR
    processed = run_pipeline(signal, SR, steps)
    sampler = MetricsSampler(interval=0.2)
    sampler.start()
    result = model.transcribe(processed)
    sampler.stop()
    ev = evaluate(result.text, reference, result.inference_time_s, duration)
    return round(ev.wer, 4)


def main():
    print("Loading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    total = len(ROOMS) * len(PIPELINES) * len(CLIPS)
    print(f"RT60={RT60}s  |  {len(ROOMS)} room sizes × {len(PIPELINES)} pipelines × {len(CLIPS)} clips = {total} runs\n")

    records = []
    done = 0

    for room_name, room_dim in ROOMS.items():
        print(f"  Generating RIR: {room_name} {room_dim} ...")
        noisy = {clip: add_rir(load_clip(clip), RT60, room_dim=room_dim) for clip in CLIPS}

        for pipeline_name, steps in PIPELINES.items():
            wers = []
            for clip in CLIPS:
                reference = (TX_DIR / f"{clip}.txt").read_text().strip()
                wer = run_one(model, noisy[clip], reference, steps)
                wers.append(wer)
                done += 1
                print(f"  [{done:>2}/{total}] {room_name:<18} | {pipeline_name:<10} | {clip:<7}  WER={wer:.3f}")
            records.append({
                "room":     room_name,
                "dims":     str(room_dim),
                "pipeline": pipeline_name,
                "small":    wers[0],
                "medium":   wers[1],
                "large":    wers[2],
                "avg":      round(sum(wers) / len(wers), 4),
            })

    df = pd.DataFrame(records)
    baseline = df[df["pipeline"] == "baseline"].set_index("room")[["small", "medium", "large", "avg"]]
    rms      = df[df["pipeline"] == "rms"].set_index("room")[["small", "medium", "large", "avg"]]
    delta    = (rms - baseline).round(4)

    W = 76
    print("\n" + "=" * W)
    print(f"  ROOM SIZE vs RMS — RT60={RT60}s  (openai/small, avg WER across clips)")
    print("=" * W)
    print(f"{'room':<18}  {'baseline':>10}  {'rms':>10}  {'delta':>10}  {'better?':>8}")
    print("-" * W)
    for room in ROOMS:
        b = baseline.loc[room, "avg"]
        r = rms.loc[room, "avg"]
        d = round(r - b, 4)
        better = "YES" if d < -0.005 else ("~same" if abs(d) <= 0.005 else "WORSE")
        print(f"{room:<18}  {b:>10.4f}  {r:>10.4f}  {d:>+10.4f}  {better:>8}")
    print("=" * W)

    print("\nPer-clip breakdown:")
    print(f"\n{'room':<18}  {'pipeline':<10}  {'small':>7}  {'medium':>7}  {'large':>7}  {'avg':>7}")
    print("-" * 62)
    for room in ROOMS:
        for pipeline in ["baseline", "rms"]:
            row = df[(df["room"] == room) & (df["pipeline"] == pipeline)].iloc[0]
            print(f"{room:<18}  {pipeline:<10}  {row['small']:>7.3f}  {row['medium']:>7.3f}  {row['large']:>7.3f}  {row['avg']:>7.3f}")
        print()

    csv_path = Path("results/room_size_benchmark.csv")
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
