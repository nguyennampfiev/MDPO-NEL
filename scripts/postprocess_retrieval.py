#!/usr/bin/env python3
import argparse

from nel_mdpo.postprocess import postprocess_retrieval_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-process retrieval JSONL with alias promotion.")
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--alias-dict", default=None)
    p.add_argument("--hipe-root", default=None)
    p.add_argument("--max-candidates", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = postprocess_retrieval_file(
        args.input_file,
        args.output_file,
        alias_dict_path=args.alias_dict,
        hipe_root=args.hipe_root,
        max_candidates=args.max_candidates,
    )
    print(stats)
