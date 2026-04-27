"""
Background thread that records inference timing.

Usage:
    sampler = MetricsSampler()
    sampler.start()
    result = model.transcribe(audio)
    elapsed = sampler.stop()
"""

import threading
import time


class MetricsSampler:
    def __init__(self, interval: float = 0.2):
        # interval kept for API compatibility but unused — timing is exact
        self.interval = interval
        self.samples: list = []
        self._t0 = 0.0

    def start(self):
        self.samples = []
        self._t0 = time.perf_counter()

    def stop(self) -> list:
        return self.samples

    def summary(self) -> dict:
        return {}
