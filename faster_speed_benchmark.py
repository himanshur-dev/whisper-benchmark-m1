#!/usr/bin/env python3

import argparse

from benchmark.speed_benchmark_common import run_model_benchmark

MODEL_KEY = "faster/small-int8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_model_benchmark(
        MODEL_KEY,
        fresh=args.fresh,
        sampler_interval=0.2,
        title="SPEED BENCHMARK - faster/small-int8",
    )


if __name__ == "__main__":
    main()
