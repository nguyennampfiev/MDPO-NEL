#!/usr/bin/env python3
import argparse
import asyncio

from nel_mdpo.selection import SelectionConfig, run_selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run phase 2 NEL candidate selection.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-dir", default="runs/selection")
    p.add_argument("--tsv-path", required=True)
    p.add_argument("--language", default="fr")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://0.0.0.0:8007/v1")
    p.add_argument("--run-tag", default="selection")
    p.add_argument("--agent-concurrency", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = asyncio.run(run_selection(SelectionConfig(**vars(args))))
    print(f"Saved TSV predictions to {path}")
