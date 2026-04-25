"""
Unified model interface for all Whisper variants.

Each model exposes:
    transcribe(audio: np.ndarray, sr: int = 16000) -> TranscriptionResult

Load a model by key:
    model = load_model("openai/small")
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    duration_s: float
    inference_time_s: float


def _mps_or_cpu() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class OpenAIWhisperModel:
    def __init__(self, size: str):
        import whisper
        device = _mps_or_cpu()
        self.model = whisper.load_model(size, device=device)
        self.device = device

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
        duration = len(audio) / sr
        # fp16 not reliable on MPS; use fp16 only on CUDA
        use_fp16 = self.device == "cuda"
        t0 = time.perf_counter()
        result = self.model.transcribe(audio.astype(np.float32), fp16=use_fp16)
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(result["text"].strip(), duration, elapsed)


class MLXWhisperModel:
    def __init__(self, hf_repo: str):
        import mlx_whisper
        from huggingface_hub import snapshot_download
        self.hf_repo = hf_repo
        # Pre-download weights so the first transcribe() call measures only inference,
        # not a potentially multi-minute HuggingFace download.
        print(f"  Downloading MLX weights: {hf_repo} ...")
        snapshot_download(repo_id=hf_repo)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
        import mlx_whisper
        duration = len(audio) / sr
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(audio.astype(np.float32), path_or_hf_repo=self.hf_repo)
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(result["text"].strip(), duration, elapsed)


class FasterWhisperModel:
    def __init__(self, size: str, compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
        duration = len(audio) / sr
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio.astype(np.float32), beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(text.strip(), duration, elapsed)


class DistilWhisperModel:
    def __init__(self, model_id: str = "distil-whisper/distil-large-v3"):
        import warnings
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = _mps_or_cpu()
        dtype = torch.float32
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True
            ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
        )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
        duration = len(audio) / sr
        t0 = time.perf_counter()
        # return_timestamps=True enables chunked long-form decoding for audio > 30s
        result = self.pipe(
            {"array": audio.astype(np.float32), "sampling_rate": sr},
            return_timestamps=True,
            chunk_length_s=30,
        )
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(result["text"].strip(), duration, elapsed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "openai/small":      lambda: OpenAIWhisperModel("small"),
    "mlx/small":         lambda: MLXWhisperModel("mlx-community/whisper-small-mlx"),
    "faster/small-int8": lambda: FasterWhisperModel("small", "int8"),
    "distil/small":      lambda: DistilWhisperModel("distil-whisper/distil-small.en"),
}


def load_model(key: str):
    if key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{key}'. Available: {available}")
    return MODEL_REGISTRY[key]()
