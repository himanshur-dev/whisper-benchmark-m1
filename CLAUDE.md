# Whisper Benchmark — M1 MacBook Pro

## Project Goal

Benchmark different Whisper model variants on Apple Silicon (M1 MacBook Pro) across noise conditions and preprocessing pipelines. Measure transcription accuracy, runtime performance, and system resource usage to find the best local ASR setup.

## Directory Structure

```
whisper-benchmark/
├── CLAUDE.md
├── requirements.txt
├── config/
│   └── experiment.yaml          # defines models, noise params, preprocessing pipelines
├── audio/
│   ├── clips/                   # .flac or .wav source clips
│   └── transcripts/             # ground-truth .txt files (one per clip, same stem name)
├── benchmark/
│   ├── __init__.py
│   ├── noise.py                 # Gaussian, Babble, RIR injectors
│   ├── preprocess.py            # VAD, SpectralGate, DeepFilterNet, LUFS, RMS
│   ├── models.py                # unified transcribe() interface for all variants
│   ├── sampler.py               # background thread: CPU/GPU/memory time series
│   └── evaluate.py              # WER, CER, S/D/I breakdown, RTF
├── run_benchmark.py             # main entrypoint — reads config, runs cross-product
├── analyze.py                   # loads results/, generates plots
└── results/
    ├── raw/                     # one JSON per run
    └── summary.csv              # aggregated metrics
```

## Audio Source — LibriSpeech test-clean

**Source:** https://www.openslr.org/12 — LibriSpeech ASR corpus, `test-clean` subset (346 MB)

LibriSpeech is the standard ASR benchmark dataset. `test-clean` contains read English speech from LibriVox audiobooks with verified transcripts.

**Download:**
```bash
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

**Format:** Each speaker/chapter directory contains `.flac` audio files and a `.trans.txt` file mapping utterance IDs to transcripts:
```
1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER...
1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM...
```

**Usage in this project:** Select 5–10 representative utterances (10–30 seconds each, varied speakers). Convert to 16 kHz mono WAV. Store in `audio/clips/`, with corresponding ground-truth text in `audio/transcripts/` using matching stem names (`1089-134686-0000.wav` → `1089-134686-0000.txt`).

**Why LibriSpeech:** Clean ground-truth transcripts, well-known in ASR research, enables comparison against published WER benchmarks, diverse speakers.

## Frameworks & Libraries

### Inference
| Library | Purpose | Notes |
|---|---|---|
| `openai-whisper` | Original Whisper (tiny → large-v3) | PyTorch backend, MPS for M1 GPU |
| `mlx-whisper` | MLX-native Whisper | Best M1 perf — uses ANE + GPU natively |
| `faster-whisper` | CTranslate2 Whisper + INT8 quantization | Best CPU-only perf |
| `transformers` (HuggingFace) | distil-whisper-large-v3 | PyTorch + MPS |

### Noise Injection
| Library | Purpose |
|---|---|
| `pyroomacoustics` | Synthesize room impulse responses (RIR) at target RT60 |
| `scipy` | `fftconvolve` for RIR application, bandpass filter |
| `librosa` | Audio loading, resampling |

Babble noise uses speaker clips from LibriSpeech `train-clean-100` (download separately or use any corpus of speech files).

### Preprocessing
| Library | Purpose |
|---|---|
| `silero-vad` | Voice Activity Detection — strips non-speech frames |
| `noisereduce` | Spectral gating noise reduction |
| `deepfilternet` | Neural noise suppression (better than spectral gating on non-stationary noise) |
| `pyloudnorm` | LUFS normalization (EBU R128, target −23 LUFS) |

RMS normalization is implemented manually (no extra library needed).

### Metrics & Evaluation
| Library | Purpose |
|---|---|
| `jiwer` | WER, CER, substitution/deletion/insertion counts |
| `psutil` | CPU utilization %, process memory (RSS) |
| `ioreg` (subprocess) | M1 GPU utilization % — parse `DeviceUtilizationPercent` from `ioreg -rc AGXAccelerator` |

### Analysis & Output
| Library | Purpose |
|---|---|
| `pandas` | Aggregate results, write summary.csv |
| `matplotlib` | Plots: WER vs SNR, RTF bar chart, GPU util time series |

## Model Variants

| Name | Config key | Backend |
|---|---|---|
| whisper-tiny | `openai/tiny` | PyTorch MPS |
| whisper-base | `openai/base` | PyTorch MPS |
| whisper-small | `openai/small` | PyTorch MPS |
| whisper-medium | `openai/medium` | PyTorch MPS |
| whisper-large-v3 | `openai/large-v3` | PyTorch MPS |
| mlx-whisper-small | `mlx/small` | MLX (ANE-native) |
| mlx-whisper-large-v3 | `mlx/large-v3` | MLX (ANE-native) |
| faster-whisper-small INT8 | `faster/small-int8` | CTranslate2 CPU |
| faster-whisper-large-v3 INT8 | `faster/large-v3-int8` | CTranslate2 CPU |
| distil-whisper-large-v3 | `distil/large-v3` | PyTorch MPS |

## Noise Conditions

- **Gaussian** — white noise at SNR: 0, 5, 10, 20, 40 dB
- **Babble** — 5 overlapping speakers at SNR: 0, 5, 10, 20 dB
- **RIR** — synthesized room impulse response at RT60: 0.2, 0.5, 1.0, 2.0 seconds

## Preprocessing Pipelines

Defined in `config/experiment.yaml`. Each pipeline is an ordered list of steps applied before inference:

- `baseline` — LUFS normalization only
- `vad` — VAD → LUFS
- `spectral` — spectral gate → VAD → LUFS
- `deepfilter` — DeepFilterNet → VAD → LUFS
- `rms` — RMS normalization only (for comparison with LUFS)

## Metrics Collected Per Run

### System (time series at 200ms intervals via `sampler.py`)
- CPU utilization %
- Memory usage (MB, RSS)
- GPU utilization % (M1 GPU via `ioreg`)

Summary stats (peak, mean) derived from time series. Raw time series saved in JSON for plotting GPU util over inference duration.

### Accuracy (via `evaluate.py`)
- WER (word error rate)
- CER (character error rate)
- Substitutions, deletions, insertions (raw counts + % of reference)
- RTF (real-time factor = inference_time / audio_duration)

## Results Format

Each run saved as `results/raw/{model}_{noise}_{params}_{pipeline}_{clip}.json`:
```json
{
  "model": "mlx/large-v3",
  "noise": "gaussian",
  "snr_db": 10,
  "pipeline": "deepfilter",
  "clip": "1089-134686-0000",
  "hypothesis": "...",
  "reference": "...",
  "wer": 0.042,
  "cer": 0.018,
  "substitutions": 2,
  "deletions": 1,
  "insertions": 0,
  "rtf": 0.31,
  "inference_time_s": 4.2,
  "audio_duration_s": 13.5,
  "peak_memory_mb": 1820,
  "mean_cpu_pct": 44.1,
  "mean_gpu_pct": 71.3,
  "metrics_samples": [
    {"t": 0.0, "cpu_pct": 12.0, "memory_mb": 1640, "gpu_pct": 0},
    {"t": 0.2, "cpu_pct": 98.0, "memory_mb": 1820, "gpu_pct": 87}
  ]
}
```

## Key Implementation Notes

- All audio must be **16 kHz mono WAV** before any pipeline stage. Convert LibriSpeech `.flac` files on ingest.
- Whisper model weights are loaded once per model, not per run. Reload cost is tracked separately as `model_load_time_s`.
- VAD returns a list of speech segments; re-assemble into a single array before passing to the model.
- `ioreg -rc AGXAccelerator` does not require sudo and returns GPU util on M1. Parse the `DeviceUtilizationPercent` field.
- DeepFilterNet expects 48 kHz input — resample to 48k before DeepFilterNet, resample back to 16k after.
- RIR convolution output is trimmed back to original signal length (`out[:len(signal)]`).
- The cross-product of all conditions can be large. The config supports a `dry_run: true` flag that runs one condition per axis for a quick sanity check.
