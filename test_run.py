#!/usr/bin/env python3
"""
Debug/test script. Three modes, each runs all three clips and prints a results table.

Usage:

  # 1. Run any model — baseline pipeline, no noise
  python test_run.py model openai/small
  python test_run.py model mlx/large-v3
  python test_run.py model faster/large-v3-int8

  # 2. Run openai/small with a noise condition
  python test_run.py noise gaussian --snr 10
  python test_run.py noise gaussian --snr 0
  python test_run.py noise rir --rt60 0.5

  # 3. Run openai/small with a preprocessing pipeline
  python test_run.py pipeline baseline
  python test_run.py pipeline vad
  python test_run.py pipeline spectral
  python test_run.py pipeline rms

Available models:
  openai/small  openai/large-v3
  mlx/small     mlx/large-v3
  faster/small-int8  faster/large-v3-int8
  distil/small  distil/large-v3
"""

import argparse
import time
from pathlib import Path

import librosa
import numpy as np

from benchmark.evaluate import evaluate
from benchmark.models import MODEL_REGISTRY, load_model
from benchmark.noise import add_gaussian, add_rir
from benchmark.preprocess import PIPELINE_STEPS, run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000
CLIPS_DIR = Path("audio/clips")
TX_DIR = Path("audio/transcripts")
CLIPS = ["small", "medium", "large"]


def load_clip(name: str):
    audio, _ = librosa.load(str(CLIPS_DIR / f"{name}.wav"), sr=SR, mono=True)
    return audio.astype(np.float32)


def load_ref(name: str) -> str:
    return (TX_DIR / f"{name}.txt").read_text().strip()


def run_one(model, clip_name: str, noise_cfg: dict, steps: list) -> dict:
    signal = load_clip(clip_name)
    reference = load_ref(clip_name)
    duration = len(signal) / SR

    # noise
    rng = np.random.default_rng(42)
    if noise_cfg["type"] == "gaussian":
        signal = add_gaussian(signal, noise_cfg["snr_db"], rng=rng)
    elif noise_cfg["type"] == "rir":
        signal = add_rir(signal, noise_cfg["rt60"])

    # preprocessing
    signal = run_pipeline(signal, SR, steps)

    # transcribe + measure
    sampler = MetricsSampler(interval=0.2)
    sampler.start()
    result = model.transcribe(signal)
    samples = sampler.stop()
    stats = sampler.summary()

    ev = evaluate(result.text, reference, result.inference_time_s, duration)
    rtf = result.inference_time_s / duration

    return {
        "clip":       clip_name,
        "duration":   round(duration, 1),
        "wer":        round(ev.wer, 4),
        "cer":        round(ev.cer, 4),
        "subs":       ev.substitutions,
        "dels":       ev.deletions,
        "ins":        ev.insertions,
        "rtf":        round(rtf, 3),
        "infer_s":    round(result.inference_time_s, 2),
        "peak_rss":   stats.get("peak_rss_mb", 0),
        "peak_mps":   stats.get("peak_mps_mb", 0),
        "mean_cpu":   stats.get("mean_cpu_pct", 0),
        "mean_gpu":   stats.get("mean_gpu_pct", 0),
        "hypothesis": result.text,
        "reference":  reference,
        "ref_words":  ev.ref_word_count,
    }


def print_results(rows: list, title: str):
    W = 78
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)
    print(f"{'CLIP':<8} {'DUR':>5} {'WER':>6} {'CER':>6} {'S/D/I':>9} {'RTF':>6} {'RSS':>7} {'MPS':>7} {'GPU':>5}")
    print("-" * W)
    for r in rows:
        sdi = f"{r['subs']}/{r['dels']}/{r['ins']}"
        print(f"{r['clip']:<8} {r['duration']:>4.1f}s {r['wer']:>6.3f} {r['cer']:>6.3f} "
              f"{sdi:>9} {r['rtf']:>6.3f} {r['peak_rss']:>6.0f}M {r['peak_mps']:>6.0f}M {r['mean_gpu']:>4.0f}%")
    print("-" * W)
    avg_wer = sum(r["wer"] for r in rows) / len(rows)
    avg_rtf = sum(r["rtf"] for r in rows) / len(rows)
    print(f"{'avg':<8} {'':>5} {avg_wer:>6.3f} {'':>6} {'':>9} {avg_rtf:>6.3f}")
    print("=" * W)
    print("\nTranscriptions:")
    for r in rows:
        print(f"\n  [{r['clip']}  {r['duration']}s  {r['ref_words']} words]")
        print(f"  REF: {r['reference']}")
        print(f"  HYP: {r['hypothesis']}")


def cmd_model(args):
    key = args.model
    if key not in MODEL_REGISTRY:
        print(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY)}")
        return

    print(f"\nLoading {key} ...")
    t0 = time.perf_counter()
    model = load_model(key)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    rows = []
    for clip in CLIPS:
        print(f"  transcribing {clip} ...", end=" ", flush=True)
        r = run_one(model, clip, {"type": "none"}, steps=[])
        print(f"WER={r['wer']:.3f}  RTF={r['rtf']:.3f}x")
        rows.append(r)

    print_results(rows, f"model={key}  |  noise=none  |  pipeline=baseline")


def cmd_noise(args):
    if args.noise == "gaussian":
        noise_cfg = {"type": "gaussian", "snr_db": args.snr}
        noise_label = f"gaussian SNR={args.snr}dB"
    elif args.noise == "rir":
        noise_cfg = {"type": "rir", "rt60": args.rt60}
        noise_label = f"RIR RT60={args.rt60}s"

    print(f"\nLoading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    rows = []
    for clip in CLIPS:
        print(f"  transcribing {clip} ...", end=" ", flush=True)
        r = run_one(model, clip, noise_cfg, steps=[])
        print(f"WER={r['wer']:.3f}  RTF={r['rtf']:.3f}x")
        rows.append(r)

    print_results(rows, f"model=openai/small  |  noise={noise_label}  |  pipeline=baseline")


def cmd_pipeline(args):
    pipeline_name = args.pipeline
    pipeline_map = {
        "baseline": [],
        "vad":      ["vad"],
        "spectral": ["spectral_gate"],
        "rms":      ["rms"],
    }
    if pipeline_name not in pipeline_map:
        print(f"Unknown pipeline '{pipeline_name}'. Available: {list(pipeline_map)}")
        return
    steps = pipeline_map[pipeline_name]

    print(f"\nLoading openai/small ...")
    t0 = time.perf_counter()
    model = load_model("openai/small")
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    rows = []
    for clip in CLIPS:
        print(f"  transcribing {clip} ...", end=" ", flush=True)
        r = run_one(model, clip, {"type": "none"}, steps=steps)
        print(f"WER={r['wer']:.3f}  RTF={r['rtf']:.3f}x")
        rows.append(r)

    print_results(rows, f"model=openai/small  |  noise=none  |  pipeline={pipeline_name} {steps}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # model subcommand
    p_model = sub.add_parser("model", help="Run a chosen model, baseline, no noise")
    p_model.add_argument("model", help=f"Model key. Available: {list(MODEL_REGISTRY)}")

    # noise subcommand
    p_noise = sub.add_parser("noise", help="Run openai/small with a chosen noise type")
    p_noise.add_argument("noise", choices=["gaussian", "rir"])
    p_noise.add_argument("--snr", type=float, default=10, help="SNR in dB (gaussian)")
    p_noise.add_argument("--rt60", type=float, default=0.5, help="RT60 in seconds (rir)")

    # pipeline subcommand
    p_pipe = sub.add_parser("pipeline", help="Run openai/small with a chosen preprocessing pipeline")
    p_pipe.add_argument("pipeline", choices=["baseline", "vad", "spectral", "rms"])

    args = parser.parse_args()

    if args.cmd == "model":
        cmd_model(args)
    elif args.cmd == "noise":
        cmd_noise(args)
    elif args.cmd == "pipeline":
        cmd_pipeline(args)


if __name__ == "__main__":
    main()
