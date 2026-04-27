#!/usr/bin/env python3
"""
Run the three non-distil speed benchmarks.

Outputs:
  outputs/speed/<model>_<clip>.csv    - per-clip benchmark row
  outputs/speed/<model>_summary.csv   - one row per clip for that model
  results/speed/<model>_<clip>.json   - full raw result including hypothesis text
"""

import argparse

from benchmark.speed_benchmark_common import run_model_benchmark

MODELS = ["openai/small", "mlx/small", "faster/small-int8"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-run all, ignore saved results")
    args = parser.parse_args()
    for model_key in MODELS:
        print("=" * 60)
        run_model_benchmark(
            model_key,
            fresh=args.fresh,
            sampler_interval=0.2,
            title=f"SPEED BENCHMARK - {model_key}",
        )

    print("\nPer-clip CSVs: outputs/speed/<model>_<clip>.csv")
    print("Per-model summaries: outputs/speed/<model>_summary.csv")


if __name__ == "__main__":
    main()
