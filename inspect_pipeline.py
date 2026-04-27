#!/usr/bin/env python3
"""
Inspect a single clip through any combination of noise + preprocessing + model.

Saves intermediate WAV files so you can listen and verify each stage:
    <output>/1_original.wav
    <output>/2_noisy.wav        (if noise applied)
    <output>/3_preprocessed.wav
    <output>/result.json

Usage examples:

  # Just preprocessing, no model
  python inspect_pipeline.py --clip audio/clips/small.wav --pipeline spectral

  # Full pipeline
  python inspect_pipeline.py --clip audio/clips/medium.wav --noise gaussian --snr 10 --pipeline spectral --model openai/small

  # Custom steps
  python inspect_pipeline.py --clip audio/clips/large.wav --steps spectral_gate vad --model faster/small-int8

  # RIR noise
  python inspect_pipeline.py --clip audio/clips/medium.wav --noise rir --rt60 0.5 --model openai/small
"""

import argparse
import difflib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml

from benchmark.noise import add_gaussian, add_rir
from benchmark.preprocess import PIPELINE_STEPS, run_pipeline
from benchmark.sampler import MetricsSampler

SR = 16000
W = 62


def sep(char="-"):
    print(char * W)


def save_wav(path, audio, sr=SR):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def word_diff(reference: str, hypothesis: str):
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)

    ref_out, hyp_out = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            ref_out.extend(ref_words[i1:i2])
            hyp_out.extend(hyp_words[j1:j2])
        elif op == "replace":
            ref_out.extend(f"[{w}]" for w in ref_words[i1:i2])
            hyp_out.extend(f"[{w}]" for w in hyp_words[j1:j2])
        elif op == "delete":
            ref_out.extend(f"[-{w}]" for w in ref_words[i1:i2])
        elif op == "insert":
            hyp_out.extend(f"[+{w}]" for w in hyp_words[j1:j2])

    print("REF:", " ".join(ref_out))
    print("HYP:", " ".join(hyp_out))


def resolve_steps(pipeline_name: str, config_path: str = "config/experiment.yaml") -> list:
    defaults = {
        "baseline": [],
        "vad":      ["vad"],
        "spectral": ["spectral_gate"],
        "rms":      ["rms"],
    }
    if pipeline_name in defaults:
        return defaults[pipeline_name]
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for p in cfg.get("preprocessing_pipelines", []):
            if p["name"] == pipeline_name:
                return p["steps"]
    except FileNotFoundError:
        pass
    raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {list(defaults)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect benchmark pipeline on a single clip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clip", required=True, help="Path to .wav audio clip")
    parser.add_argument("--transcript",
                        help="Ground-truth .txt (auto-detected from audio/transcripts/ if omitted)")

    noise_group = parser.add_argument_group("noise (optional)")
    noise_group.add_argument("--noise", choices=["gaussian", "rir"], help="Noise type")
    noise_group.add_argument("--snr", type=float, default=10,
                             help="SNR in dB for gaussian (default: 10)")
    noise_group.add_argument("--rt60", type=float, default=0.5,
                             help="RT60 in seconds for rir (default: 0.5)")

    pre_group = parser.add_argument_group("preprocessing")
    pre_group.add_argument("--pipeline", default="baseline",
                           help="Named pipeline: baseline, vad, spectral, rms")
    pre_group.add_argument("--steps", nargs="+",
                           help=f"Override with explicit steps. Available: {list(PIPELINE_STEPS)}")

    model_group = parser.add_argument_group("model (optional)")
    model_group.add_argument("--model",
                             help="Model key (skip transcription if omitted). "
                                  "e.g. openai/small, faster/small-int8, mlx/large-v3")

    parser.add_argument("--output", default="inspect_output",
                        help="Directory to save audio files and result JSON (default: inspect_output)")
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"Error: clip not found: {clip_path}")
        return

    tx_path = Path(args.transcript) if args.transcript else Path("audio/transcripts") / (clip_path.stem + ".txt")
    has_ref = tx_path.exists()
    reference = tx_path.read_text().strip() if has_ref else None

    steps = args.steps if args.steps else resolve_steps(args.pipeline)

    sep("=")
    print(f"Clip:       {clip_path}")
    print(f"Transcript: {tx_path} {'✓' if has_ref else '(not found — accuracy metrics skipped)'}")
    if args.noise == "gaussian":
        noise_desc = f"gaussian  SNR={args.snr} dB"
    elif args.noise == "rir":
        noise_desc = f"RIR  RT60={args.rt60}s"
    else:
        noise_desc = "none"
    print(f"Noise:      {noise_desc}")
    print(f"Pipeline:   {steps if steps else '(none)'}")
    print(f"Model:      {args.model or '(none)'}")
    sep("=")

    signal, _ = librosa.load(str(clip_path), sr=SR, mono=True)
    signal = signal.astype(np.float32)
    original_duration = len(signal) / SR
    print(f"\nOriginal: {original_duration:.2f}s  |  {len(signal)} samples @ {SR} Hz")

    out_dir = Path(args.output)
    save_wav(out_dir / "1_original.wav", signal)
    print(f"Saved:    {out_dir}/1_original.wav")

    rng = np.random.default_rng(42)
    if args.noise == "gaussian":
        noisy = add_gaussian(signal, args.snr, rng=rng)
    elif args.noise == "rir":
        noisy = add_rir(signal, args.rt60)
    else:
        noisy = signal.copy()

    if args.noise:
        save_wav(out_dir / "2_noisy.wav", noisy)
        print(f"Saved:    {out_dir}/2_noisy.wav")

    sep()
    label = " → ".join(steps) if steps else "(no steps)"
    print(f"Preprocessing: {label}")
    t0 = time.perf_counter()
    processed = run_pipeline(noisy, SR, steps)
    preprocess_time = time.perf_counter() - t0
    processed_duration = len(processed) / SR

    save_wav(out_dir / "3_preprocessed.wav", processed)
    print(f"Saved:    {out_dir}/3_preprocessed.wav")
    print(f"Duration: {original_duration:.2f}s → {processed_duration:.2f}s  "
          f"(preprocessing took {preprocess_time:.2f}s)")

    result_data = {
        "clip": str(clip_path),
        "noise": args.noise,
        "noise_snr_db": args.snr if args.noise == "gaussian" else None,
        "noise_rt60": args.rt60 if args.noise == "rir" else None,
        "pipeline": steps,
        "model": args.model,
        "reference": reference,
        "audio_duration_s": round(original_duration, 3),
        "preprocessed_duration_s": round(processed_duration, 3),
        "preprocess_time_s": round(preprocess_time, 3),
    }

    if args.model:
        from benchmark.evaluate import evaluate
        from benchmark.models import load_model

        sep()
        print(f"Loading model: {args.model} ...")
        t_load = time.perf_counter()
        model = load_model(args.model)
        print(f"Loaded in {time.perf_counter() - t_load:.1f}s")

        sampler = MetricsSampler(interval=0.2)
        print("Transcribing ...")
        sampler.start()
        result = model.transcribe(processed)
        samples = sampler.stop()
        sys_stats = sampler.summary()

        sep()
        print("TRANSCRIPTION OUTPUT")
        sep()
        print(f"  {result.text}")

        if has_ref:
            sep()
            print("WORD DIFF  ( [word] = substitution  [-word] = deletion  [+word] = insertion )")
            sep()
            word_diff(reference, result.text)

            ev = evaluate(result.text, reference, result.inference_time_s, original_duration)

            sep()
            print("ACCURACY")
            sep()
            print(f"  WER:           {ev.wer:.4f}  ({ev.wer*100:.1f}%)")
            print(f"  CER:           {ev.cer:.4f}  ({ev.cer*100:.1f}%)")
            print(f"  Substitutions: {ev.substitutions}")
            print(f"  Deletions:     {ev.deletions}")
            print(f"  Insertions:    {ev.insertions}")
            print(f"  Correct words: {ev.hits} / {ev.ref_word_count}")

            result_data.update({
                "hypothesis": result.text,
                "wer": round(ev.wer, 4),
                "cer": round(ev.cer, 4),
                "substitutions": ev.substitutions,
                "deletions": ev.deletions,
                "insertions": ev.insertions,
                "hits": ev.hits,
                "ref_word_count": ev.ref_word_count,
            })
        else:
            result_data["hypothesis"] = result.text

        sep()
        print("PERFORMANCE")
        sep()
        rtf = result.inference_time_s / original_duration
        print(f"  Audio duration:  {original_duration:.2f}s")
        print(f"  Inference time:  {result.inference_time_s:.2f}s")
        print(f"  RTF:             {rtf:.3f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)")
        print(f"  Mean CPU:        {sys_stats.get('mean_cpu_pct', '?'):.1f}%")
        print(f"  Mean GPU:        {sys_stats.get('mean_gpu_pct', '?'):.1f}%")

        result_data.update({
            "rtf": round(rtf, 4),
            "inference_time_s": round(result.inference_time_s, 3),
            **sys_stats,
            "metrics_samples": samples,
        })

    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result_data, indent=2))

    sep("=")
    print(f"Audio files in:  {out_dir}/")
    print(f"  1_original.wav      — clean input")
    if args.noise:
        print(f"  2_noisy.wav         — after {args.noise} noise")
    print(f"  3_preprocessed.wav  — after preprocessing")
    print(f"Full results:    {json_path}")
    sep("=")


if __name__ == "__main__":
    main()
