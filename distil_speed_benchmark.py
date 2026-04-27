#!/usr/bin/env python3
"""Run distil/small in isolation."""

import argparse

from benchmark.speed_benchmark_common import run_model_benchmark

MODEL_KEY = "distil/small"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_model_benchmark(
        MODEL_KEY,
        fresh=args.fresh,
        title="SPEED BENCHMARK - distil/small",
    )


if __name__ == "__main__":
    main()
