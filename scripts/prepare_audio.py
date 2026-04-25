#!/usr/bin/env python3
"""
Download LibriSpeech test-clean and prepare benchmark clips.

Selects utterances within the specified duration range, converts them to
16 kHz mono WAV, and writes matching ground-truth transcript .txt files.

Output:
    audio/clips/          — WAV files
    audio/transcripts/    — matching .txt files (one word per line, uppercase)

Usage:
    python scripts/prepare_audio.py
    python scripts/prepare_audio.py --n-clips 20 --min-duration 8 --max-duration 25
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SR = 16000


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def extract(tar_path: Path, dest_dir: Path):
    marker = dest_dir / "LibriSpeech"
    if marker.exists():
        print(f"  Already extracted.")
        return
    print(f"  Extracting ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path)) as tf:
        tf.extractall(str(dest_dir))
    print(f"  Done.")


def parse_trans(trans_file: Path) -> dict:
    entries = {}
    for line in trans_file.read_text().strip().splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            entries[parts[0]] = parts[1]
    return entries


def convert_clip(src: Path, dst: Path) -> float:
    audio, _ = librosa.load(str(src), sr=SR, mono=True)
    audio = audio.astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, SR)
    return len(audio) / SR


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech test-clean clips for benchmarking"
    )
    parser.add_argument("--n-clips", type=int, default=10,
                        help="Number of clips to prepare (default: 10)")
    parser.add_argument("--min-duration", type=float, default=5.0,
                        help="Minimum clip duration in seconds (default: 5.0)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Maximum clip duration in seconds (default: 30.0)")
    parser.add_argument("--librispeech-dir", default="audio/librispeech",
                        help="Directory to download/extract LibriSpeech")
    parser.add_argument("--clips-dir", default="audio/clips")
    parser.add_argument("--transcripts-dir", default="audio/transcripts")
    args = parser.parse_args()

    ls_dir = Path(args.librispeech_dir)
    clips_dir = Path(args.clips_dir)
    transcripts_dir = Path(args.transcripts_dir)

    tar_path = ls_dir / "test-clean.tar.gz"

    print("=" * 55)
    print("Step 1/3 — Download")
    print("=" * 55)
    download(LIBRISPEECH_URL, tar_path)

    print("\n" + "=" * 55)
    print("Step 2/3 — Extract")
    print("=" * 55)
    extract(tar_path, ls_dir)

    print("\n" + "=" * 55)
    print("Step 3/3 — Convert and select clips")
    print("=" * 55)
    clips_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    ls_root = ls_dir / "LibriSpeech" / "test-clean"
    selected = 0

    for trans_file in sorted(ls_root.rglob("*.trans.txt")):
        if selected >= args.n_clips:
            break

        entries = parse_trans(trans_file)
        chapter_dir = trans_file.parent

        for utt_id, transcript in entries.items():
            if selected >= args.n_clips:
                break

            flac_path = chapter_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                continue

            info = sf.info(str(flac_path))
            duration = info.frames / info.samplerate
            if not (args.min_duration <= duration <= args.max_duration):
                continue

            wav_dst = clips_dir / f"{utt_id}.wav"
            txt_dst = transcripts_dir / f"{utt_id}.txt"

            actual_dur = convert_clip(flac_path, wav_dst)
            # LibriSpeech transcripts are already uppercase; write as-is
            txt_dst.write_text(transcript)

            selected += 1
            print(f"  [{selected:>2}/{args.n_clips}] {utt_id}.wav  ({actual_dur:.1f}s)  \"{transcript[:60]}...\"")

    print(f"\n{selected} clip(s) ready.")
    print(f"  Clips:       {clips_dir}/")
    print(f"  Transcripts: {transcripts_dir}/")
    if selected < args.n_clips:
        print(f"\n  Note: only {selected}/{args.n_clips} clips found in the duration range "
              f"[{args.min_duration}s, {args.max_duration}s]. "
              f"Try widening --min-duration / --max-duration.")
    print("\nNext steps:")
    print("  # Quick sanity check on one clip:")
    print("  python inspect_pipeline.py --clip audio/clips/<id>.wav --model openai/small")
    print()
    print("  # Full benchmark:")
    print("  python run_benchmark.py --config config/experiment.yaml")


if __name__ == "__main__":
    main()
