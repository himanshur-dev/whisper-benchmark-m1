import numpy as np
from scipy.signal import fftconvolve


def _scale_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = np.sqrt(np.mean(signal ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms < 1e-9:
        return noise
    target_rms = signal_rms / (10 ** (snr_db / 20))
    return noise * (target_rms / noise_rms)


def add_gaussian(signal: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.standard_normal(len(signal)).astype(np.float32)
    noise = _scale_to_snr(signal, noise, snr_db)
    return (signal + noise).astype(np.float32)


def add_rir(
    signal: np.ndarray,
    rt60: float,
    room_dim: list = None,
    sr: int = 16000,
) -> np.ndarray:
    import pyroomacoustics as pra

    room_dim = room_dim or [6.0, 5.0, 3.0]

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    source_pos = [room_dim[0] * 0.3, room_dim[1] * 0.6, 1.5]
    mic_pos = np.array([room_dim[0] * 0.7, room_dim[1] * 0.4, 1.5]).reshape(3, 1)
    room.add_source(source_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()

    rir = room.rir[0][0].astype(np.float32)
    return fftconvolve(signal, rir)[: len(signal)].astype(np.float32)
