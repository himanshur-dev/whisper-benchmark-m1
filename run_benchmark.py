#!/usr/bin/env python3
"""
Main benchmark entrypoint.

Usage:
    python run_benchmark.py                              # run everything in config
    python run_benchmark.py --dry-run                   # one condition per axis
    python run_benchmark.py --models openai/small mlx/small
    python run_benchmark.py --noise-types gaussian rir
    python run_benchmark.py --config config/custom.yaml
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import yaml

from benchmark.evaluate import evaluate
from benchmark.models import load_model
from benchmark.noise import add_gaussian, add_rir
from benchmark.preprocess import run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clip(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


def apply_noise(signal: np.ndarray, ncfg: dict, rng) -> np.ndarray:
    t = ncfg["type"]
    if t == "gaussian":
        return add_gaussian(signal, ncfg["snr_db"], rng=rng)
    if t == "rir":
        return add_rir(signal, ncfg["rt60"], room_dim=ncfg.get("room_dim"))
    raise ValueError(f"Unknown noise type: {t}")


def build_noise_configs(noise_section: dict) -> list:
    configs = []
    for ntype, params in noise_section.items():
        if ntype == "gaussian":
            for snr in params["snr_db"]:
                configs.append({"type": "gaussian", "snr_db": snr,
                                 "label": f"snr{snr}"})
        elif ntype == "rir":
            for rt60 in params["rt60"]:
                configs.append({"type": "rir", "rt60": rt60,
                                 "room_dim": params.get("room_dim"),
                                 "label": f"rt60_{rt60}"})
    return configs


def result_path(results_dir: Path, model_key, noise_type, noise_label, pipeline, clip_stem) -> Path:
    safe_model = model_key.replace("/", "-")
    fname = f"{safe_model}_{noise_type}_{noise_label}_{pipeline}_{clip_stem}.json"
    return results_dir / "raw" / fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--models", nargs="+", help="Limit to these model keys")
    parser.add_argument("--noise-types", nargs="+", help="Limit to these noise types")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one condition per axis (quick sanity check)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dry_run = args.dry_run or cfg.get("dry_run", False)
    rng = np.random.default_rng(cfg.get("seed", 42))

    results_dir = Path(cfg["results_dir"])
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)

    clips_dir = Path(cfg["audio"]["clips_dir"])
    transcripts_dir = Path(cfg["audio"]["transcripts_dir"])
    clips = sorted(clips_dir.glob("*.wav"))

    if not clips:
        print(f"No .wav files found in {clips_dir}. Run scripts/prepare_audio.py first.")
        return

    noise_configs = build_noise_configs(cfg["noise"])
    if args.noise_types:
        noise_configs = [n for n in noise_configs if n["type"] in args.noise_types]

    pipelines = cfg["preprocessing_pipelines"]
    model_keys = cfg["models"]
    if args.models:
        model_keys = [m for m in model_keys if m in args.models]

    if dry_run:
        noise_configs = noise_configs[:1]
        pipelines = pipelines[:1]
        clips = clips[:1]
        print("[dry-run] One condition per axis")

    total = len(model_keys) * len(noise_configs) * len(pipelines) * len(clips)
    print(f"Conditions: {len(model_keys)} models × {len(noise_configs)} noise × "
          f"{len(pipelines)} pipelines × {len(clips)} clips = {total} runs\n")

    for model_key in model_keys:
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

        for clip_path in clips:
            tx_path = transcripts_dir / (clip_path.stem + ".txt")
            if not tx_path.exists():
                print(f"  [skip] No transcript for {clip_path.name}")
                continue

            signal = load_clip(clip_path)
            reference = tx_path.read_text().strip()
            original_duration = len(signal) / SR

            for ncfg in noise_configs:
                try:
                    noisy = apply_noise(signal, ncfg, rng)
                except Exception as e:
                    print(f"  [skip] Noise injection failed ({ncfg['type']}): {e}")
                    continue

                for pipeline in pipelines:
                    rpath = result_path(
                        results_dir, model_key,
                        ncfg["type"], ncfg["label"],
                        pipeline["name"], clip_path.stem,
                    )
                    if rpath.exists():
                        print(f"  [skip] {rpath.name}")
                        continue

                    label = (f"  {model_key} | {ncfg['type']} {ncfg['label']} | "
                             f"{pipeline['name']} | {clip_path.stem}")
                    print(label)

                    try:
                        processed = run_pipeline(noisy, SR, pipeline["steps"])
                    except Exception as e:
                        print(f"    [error] preprocess: {e}")
                        continue

                    sampler = MetricsSampler(interval=0.2)
                    sampler.start()
                    try:
                        result = model.transcribe(processed)
                    except Exception as e:
                        sampler.stop()
                        print(f"    [error] transcribe: {e}")
                        continue
                    samples = sampler.stop()
                    sys_stats = sampler.summary()

                    ev = evaluate(result.text, reference, result.inference_time_s, original_duration)

                    noise_params = {k: v for k, v in ncfg.items()
                                    if k not in ("type", "label", "corpus_dir")}
                    record = {
                        "model": model_key,
                        "model_load_time_s": load_time,
                        "noise_type": ncfg["type"],
                        "noise_params": noise_params,
                        "pipeline": pipeline["name"],
                        "clip": clip_path.stem,
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
                        "audio_duration_s": round(original_duration, 3),
                        **sys_stats,
                        "metrics_samples": samples,
                    }

                    rpath.write_text(json.dumps(record, indent=2))
                    print(f"    WER={ev.wer:.3f}  CER={ev.cer:.3f}  RTF={ev.rtf:.2f}  "
                          f"mem={sys_stats.get('peak_memory_mb', '?'):.0f}MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
