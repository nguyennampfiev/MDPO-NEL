from pathlib import Path

from .alias import load_or_build_alias_dictionary, promote_alias_qids
from .data import read_jsonl, write_jsonl


def postprocess_retrieval_file(
    input_file: str | Path,
    output_file: str | Path,
    alias_dict_path: str | Path | None = None,
    hipe_root: str | Path | None = None,
    max_candidates: int = 10,
) -> dict[str, int]:
    """Normalize retrieval records and optionally promote train-set alias hits.

    The output keeps the original record fields and rewrites `retrievals` as a
    deduplicated list of QIDs, ready for the selection pipeline.
    """
    aliases = load_or_build_alias_dictionary(alias_dict_path, hipe_root)
    records = read_jsonl(input_file)
    out = []
    alias_hits = 0
    correct = 0

    for record in records:
        mention = record.get("input_candidate") or record.get("entity") or record.get("mention") or ""
        retrievals, alias_fired = promote_alias_qids(
            mention,
            record.get("retrievals", []),
            aliases,
            max_candidates=max_candidates,
        )
        gold = record.get("gold_qid")
        updated = dict(record)
        updated["input_candidate"] = mention
        updated["retrievals"] = retrievals
        updated["alias_fired"] = alias_fired
        updated["is_correct"] = gold in retrievals if gold and gold != "NIL" else not retrievals
        alias_hits += int(alias_fired)
        correct += int(updated["is_correct"])
        out.append(updated)

    write_jsonl(out, output_file)
    return {"records": len(out), "alias_hits": alias_hits, "correct": correct}
