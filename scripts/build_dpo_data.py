#!/usr/bin/env python3
import argparse
import asyncio

from nel_mdpo.dpo_data import build_dpo_data_from_hipe_tsv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-negative DPO data from HIPE TSV.")
    p.add_argument("--tsv-path", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--language", default="fr")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--base-url", default="http://0.0.0.0:8007/v1")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--max-negatives", type=int, default=3)
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = asyncio.run(build_dpo_data_from_hipe_tsv(**vars(args)))
    print(f"Saved DPO data to {path}")
