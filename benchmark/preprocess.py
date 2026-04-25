"""
Preprocessing steps for the Whisper benchmark pipeline.

Each step has the signature:  fn(signal: np.ndarray, sr: int) -> np.ndarray
All steps operate on 16 kHz mono float32 audio.
"""

import numpy as np

_vad_model = None


def _vad(signal: np.ndarray, sr: int) -> np.ndarray:
    global _vad_model
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

    if _vad_model is None:
        _vad_model = load_silero_vad()

    audio = torch.from_numpy(signal).float()
    timestamps = get_speech_timestamps(audio, _vad_model, sampling_rate=sr)
    if not timestamps:
        return signal
    speech = collect_chunks(timestamps, audio)
    return speech.numpy().astype(np.float32)


def _spectral_gate(signal: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr
    denoised = nr.reduce_noise(y=signal, sr=sr, stationary=True)
    return denoised.astype(np.float32)


def _rms_normalize(signal: np.ndarray, sr: int, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-9:
        return signal
    normalized = signal * (target_rms / rms)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


PIPELINE_STEPS = {
    "vad":          _vad,
    "spectral_gate": _spectral_gate,
    "rms":          _rms_normalize,
}


def run_pipeline(signal: np.ndarray, sr: int, steps: list) -> np.ndarray:
    for step in steps:
        if step not in PIPELINE_STEPS:
            raise ValueError(f"Unknown step '{step}'. Available: {list(PIPELINE_STEPS)}")
        signal = PIPELINE_STEPS[step](signal, sr)
    return signal
