import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.sampler import MetricsSampler

SR = 16000
CLIPS = ["small", "medium", "large", "xlarge"]
CLIPS_DIR = Path("audio/clips")
TX_DIR = Path("audio/transcripts")
RESULTS_DIR = Path("results/speed")
OUTPUTS_DIR = Path("outputs/speed")


def load_clip(name: str) -> np.ndarray:
    audio, _ = librosa.load(str(CLIPS_DIR / f"{name}.wav"), sr=SR, mono=True)
    return audio.astype(np.float32)


def safe_model_name(key: str) -> str:
    return key.replace("/", "-")


def timeseries_path(model_key: str, clip: str) -> Path:
    return OUTPUTS_DIR / f"{safe_model_name(model_key)}_{clip}.csv"


def result_path(model_key: str, clip: str) -> Path:
    return RESULTS_DIR / f"{safe_model_name(model_key)}_{clip}.json"


def summary_path(model_key: str) -> Path:
    return OUTPUTS_DIR / f"{safe_model_name(model_key)}_summary.csv"


def write_timeseries(path: Path, record: dict):
    row = {k: v for k, v in record.items() if k not in ("hypothesis", "reference")}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_summary(model_key: str, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame([
        {k: v for k, v in row.items() if k not in ("hypothesis", "reference")}
        for row in rows
    ])
    df.to_csv(summary_path(model_key), index=False)


def print_summary(rows: list[dict], title: str):
    if not rows:
        return
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"{'model':<22} {'clip':<7} {'dur':>6} {'infer':>7} {'rtf':>7} {'wer':>7} {'cer':>7}")
    print("-" * width)

    for row in rows:
        print(
            f"  {row['model']:<20} {row['clip']:<7} {row['duration_s']:>5.0f}s "
            f"{row['inference_time_s']:>6.1f}s {row['rtf']:>7.3f}x "
            f"{row['wer']:>7.3f} {row['cer']:>7.3f}"
        )

    print("=" * width)


def run_model_benchmark(
    model_key: str,
    *,
    fresh: bool = False,
    sampler_interval: float | None = None,
    title: str | None = None,
) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cached_records: list[dict] = []
    for clip in CLIPS:
        rpath = result_path(model_key, clip)
        if not rpath.exists():
            cached_records = []
            break
        cached_records.append(json.loads(rpath.read_text()))

    if not fresh and len(cached_records) == len(CLIPS):
        print(f"Using cached results for {model_key}\n")
        for record in cached_records:
            rows.append(record)
            print(f"  [skip] {record['clip']}")
        write_summary(model_key, rows)
        print(f"\nSummary updated: {summary_path(model_key)}")
        print_summary(rows, title or f"SPEED BENCHMARK - {model_key}")
        return rows

    print(f"Loading: {model_key}")
    t_load = time.perf_counter()
    model = load_model(model_key)
    load_time_s = round(time.perf_counter() - t_load, 2)
    print(f"  Ready in {load_time_s}s  (includes warmup)\n")

    for i, clip in enumerate(CLIPS, 1):
        rpath = result_path(model_key, clip)
        csv_path = timeseries_path(model_key, clip)

        if not fresh and rpath.exists():
            record = json.loads(rpath.read_text())
            rows.append(record)
            print(f"  [skip] {clip}")
            continue

        signal = load_clip(clip)
        reference = (TX_DIR / f"{clip}.txt").read_text().strip()
        duration = len(signal) / SR

        sampler = MetricsSampler(interval=sampler_interval) if sampler_interval is not None else MetricsSampler()
        sampler.start()
        result = model.transcribe(signal)
        sampler.stop()

        ev = evaluate(result.text, reference, result.inference_time_s, duration)

        record = {
            "model": model_key,
            "clip": clip,
            "duration_s": round(duration, 2),
            "inference_time_s": round(result.inference_time_s, 3),
            "rtf": round(ev.rtf, 4),
            "load_time_s": load_time_s,
            "wer": round(ev.wer, 4),
            "cer": round(ev.cer, 4),
            "substitutions": ev.substitutions,
            "deletions": ev.deletions,
            "insertions": ev.insertions,
            "ref_word_count": ev.ref_word_count,
            "hypothesis": result.text,
            "reference": reference,
        }

        rpath.write_text(json.dumps(record, indent=2))
        write_timeseries(csv_path, record)
        rows.append(record)

        print(
            f"  [{i}/{len(CLIPS)}] {clip:<7}  "
            f"WER={ev.wer:.3f}  RTF={ev.rtf:.3f}x  infer={result.inference_time_s:.1f}s"
        )

    write_summary(model_key, rows)
    print(f"\nSummary updated: {summary_path(model_key)}")
    print_summary(rows, title or f"SPEED BENCHMARK - {model_key}")
    return rows
