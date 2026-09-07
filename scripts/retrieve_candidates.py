#!/usr/bin/env python3
import argparse
import asyncio

from nel_mdpo.retrieval import RetrievalConfig, run_retrieval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run phase 1 NEL candidate retrieval.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-dir", default="runs/retrieval")
    p.add_argument("--dataset-name", default="hipe2020")
    p.add_argument("--language", default="fr")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--base-url", default="http://0.0.0.0:8007/v1")
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--search-languages", default=None)
    p.add_argument("--alias-dict", default=None)
    p.add_argument("--hipe-root", default=None)
    p.add_argument("--entity-concurrency", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = asyncio.run(run_retrieval(RetrievalConfig(**vars(args).copy())))
    print(f"Saved retrievals to {path}")
