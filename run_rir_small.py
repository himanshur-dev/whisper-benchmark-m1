#!/usr/bin/env python3
"""
Benchmark openai/small on large.wav across all RIR RT60 levels
and preprocessing pipeline combinations.

Outputs:
  outputs/rir/raw/<run>.json   — one JSON per run
  outputs/rir/summary.csv      — all runs in a single CSV for graphing
"""

import csv
import json
import time
from pathlib import Path

import librosa
import numpy as np

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.noise import add_rir
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_PATH = Path("audio/clips/small.wav")
TRANSCRIPT_PATH = Path("audio/transcripts/small.txt")
OUTPUT_DIR = Path("outputs/rir")
SR = 16000

RT60_LEVELS = [0.2, 0.5, 1.0, 2.0]   # seconds

PIPELINES = [
    ("none",     []),
    ("rms",      ["rms"]),
    ("vad",      ["vad"]),
    ("spectral", ["spectral_gate"]),
]

CSV_FIELDS = [
    "model", "clip", "rt60", "pipeline",
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


def run_name(rt60: float, pipeline_name: str) -> str:
    return f"openai-small_rir_rt60{rt60}_{pipeline_name}_{CLIP_PATH.stem}"


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

    total = len(RT60_LEVELS) * len(PIPELINES)
    print(f"openai/small | {CLIP_PATH.name} | {len(RT60_LEVELS)} RT60 × {len(PIPELINES)} pipelines = {total} runs\n")

    print("Loading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    load_time = round(time.perf_counter() - t0, 2)
    print(f"  Ready in {load_time}s\n")

    all_records = []

    for rt60 in RT60_LEVELS:
        print(f"  Generating RIR (RT60={rt60}s) ...", end=" ", flush=True)
        reverbed = add_rir(signal, rt60)
        print("done")

        for pipeline_name, steps in PIPELINES:
            name = run_name(rt60, pipeline_name)
            rpath = raw_dir / f"{name}.json"

            if rpath.exists():
                print(f"  [skip] {name}")
                record = json.loads(rpath.read_text())
                all_records.append(record)
                continue

            print(f"  RT60={rt60}s | pipeline={pipeline_name:<12}", end="  ", flush=True)

            try:
                processed = run_pipeline(reverbed, SR, steps) if steps else reverbed
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
                "rt60": rt60,
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
