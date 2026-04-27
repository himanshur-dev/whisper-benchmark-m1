#!/usr/bin/env python3
"""
Benchmark openai/small on large.wav across all Gaussian SNR levels
and preprocessing pipeline combinations.

Outputs:
  outputs/guassian/raw/<run>.json   — one JSON per run
  outputs/guassian/summary.csv      — all runs in a single CSV for graphing
"""

import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.noise import add_gaussian
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_PATH = Path("audio/clips/large.wav")
TRANSCRIPT_PATH = Path("audio/transcripts/large.txt")
OUTPUT_DIR = Path("outputs/guassian")
SR = 16000

SNR_LEVELS = [0, 5, 10, 20, 40]   # dB

# Each pipeline is (name, list-of-steps).
# Steps available: "vad", "spectral_gate", "rms"
PIPELINES = [
    ("none",     []),
    ("rms",      ["rms"]),
    ("vad",      ["vad"]),
    ("spectral", ["spectral_gate"]),
]

CSV_FIELDS = [
    "model", "clip", "snr_db", "pipeline",
    "wer", "cer", "substitutions", "deletions", "insertions", "hits", "ref_word_count",
    "rtf", "inference_time_s", "audio_duration_s",
    "hypothesis", "reference",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clip(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


def run_name(snr_db: int, pipeline_name: str) -> str:
    return f"openai-small_gaussian_snr{snr_db}_{pipeline_name}_{CLIP_PATH.stem}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not CLIP_PATH.exists():
        raise FileNotFoundError(f"Clip not found: {CLIP_PATH}")
    if not TRANSCRIPT_PATH.exists():
        raise FileNotFoundError(f"Transcript not found: {TRANSCRIPT_PATH}")

    signal = load_clip(CLIP_PATH)
    reference = TRANSCRIPT_PATH.read_text().strip()
    audio_duration = len(signal) / SR
    rng = np.random.default_rng(42)

    total = len(SNR_LEVELS) * len(PIPELINES)
    print(f"openai/small | {CLIP_PATH.name} | {len(SNR_LEVELS)} SNR × {len(PIPELINES)} pipelines = {total} runs\n")

    print("Loading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    load_time = round(time.perf_counter() - t0, 2)
    print(f"  Ready in {load_time}s\n")

    all_records = []

    for snr_db in SNR_LEVELS:
        noisy = add_gaussian(signal, snr_db, rng=rng)

        for pipeline_name, steps in PIPELINES:
            name = run_name(snr_db, pipeline_name)
            rpath = raw_dir / f"{name}.json"

            if rpath.exists():
                print(f"  [skip] {name}")
                record = json.loads(rpath.read_text())
                all_records.append(record)
                continue

            print(f"  SNR={snr_db:>3}dB | pipeline={pipeline_name:<20}", end="  ", flush=True)

            try:
                processed = run_pipeline(noisy, SR, steps) if steps else noisy
            except Exception as e:
                print(f"[error] preprocess: {e}")
                continue

            sampler = MetricsSampler(interval=0.2)
            sampler.start()
            try:
                result = model.transcribe(processed)
            except Exception as e:
                sampler.stop()
                print(f"[error] transcribe: {e}")
                continue
            sampler.stop()

            ev = evaluate(result.text, reference, result.inference_time_s, audio_duration)

            record = {
                "model": "openai/small",
                "model_load_time_s": load_time,
                "clip": CLIP_PATH.stem,
                "snr_db": snr_db,
                "pipeline": pipeline_name,
                "reference": reference,
                "hypothesis": result.text,
                "wer": round(ev.wer, 4),
                "cer": round(ev.cer, 4),
                "substitutions": ev.substitutions,
                "deletions": ev.deletions,
                "insertions": ev.insertions,
                "hits": ev.hits,
                "ref_word_count": ev.ref_word_count,
                "rtf": round(ev.rtf, 4),
                "inference_time_s": round(result.inference_time_s, 3),
                "audio_duration_s": round(audio_duration, 3),
            }

            rpath.write_text(json.dumps(record, indent=2))
            all_records.append(record)
            print(f"WER={ev.wer:.3f}  CER={ev.cer:.3f}  RTF={ev.rtf:.3f}")

    # Write summary CSV
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nWrote {len(all_records)} rows → {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
