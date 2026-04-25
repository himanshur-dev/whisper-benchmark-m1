"""
Background thread that samples CPU, memory, and GPU utilization at a fixed
interval during model inference.

Usage:
    sampler = MetricsSampler(interval=0.2)
    sampler.start()
    result = model.transcribe(audio)
    samples = sampler.stop()       # list[dict] — one entry per interval
    summary = sampler.summary()    # peak/mean aggregates
"""

import re
import subprocess
import threading
import time

import psutil
import torch


def _gpu_memory_mb() -> float:
    """
    Return GPU/accelerator memory in MB.
    Tries MLX first (covers mlx models), then PyTorch MPS (covers openai/distil models).
    """
    try:
        import mlx.core
        return round(mlx.core.get_active_memory() / 1e6, 1)
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return round(torch.mps.current_allocated_memory() / 1e6, 1)
    except Exception:
        pass
    return 0


def _gpu_util_percent() -> int:
    """Read Apple GPU utilization via ioreg (no sudo required)."""
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout
        for line in out.splitlines():
            if "DeviceUtilizationPercent" in line:
                m = re.search(r"=\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return 0


class MetricsSampler:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.samples: list = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()
        self._t0 = 0.0

    def start(self):
        self.samples = []
        self._running = True
        self._t0 = time.perf_counter()
        # warm up cpu_percent (first call always returns 0.0)
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._running = False
        if self._thread:
            self._thread.join()
        return self.samples

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpus = [s["cpu_pct"] for s in self.samples]
        rss  = [s["rss_mb"]  for s in self.samples]
        mps  = [s["mps_mb"]  for s in self.samples]
        gpus = [s["gpu_pct"] for s in self.samples]
        return {
            "peak_rss_mb":    round(max(rss), 1),
            "peak_mps_mb":    round(max(mps), 1),
            "peak_cpu_pct":   round(max(cpus), 1),
            "mean_cpu_pct":   round(sum(cpus) / len(cpus), 1),
            "peak_gpu_pct":   round(max(gpus), 1),
            "mean_gpu_pct":   round(sum(gpus) / len(gpus), 1),
        }

    def _loop(self):
        while self._running:
            t = round(time.perf_counter() - self._t0, 3)
            self.samples.append({
                "t":           t,
                "cpu_pct":     round(self._proc.cpu_percent(interval=None), 1),
                "rss_mb":      round(self._proc.memory_info().rss / 1e6, 1),
                "mps_mb":      _gpu_memory_mb(),
                "gpu_pct":     _gpu_util_percent(),
            })
            time.sleep(self.interval)
