import asyncio
import json
import random
from pathlib import Path

import aiohttp

from .data import extract_entities, process_hipe_tsv, write_jsonl
from .retrieval import build_retrieval_agent, suggest_candidate_names
from .wikidata import WikidataClient


def format_prompt(mention: str, context: str, candidate_ids: list[str]) -> str:
    return f"Mention: {mention!r}\nContext: {context!r}\nCandidates: {candidate_ids!r}"


def format_answer(qid: str, description: str = "") -> str:
    suffix = f": {description}" if description else ""
    return f"The correct entity is {qid}{suffix}"


def build_multineg_records(
    mention: str,
    context: str,
    gold_qid: str,
    candidates: list[dict],
    max_negatives: int = 3,
) -> list[dict]:
    if gold_qid.upper() == "NIL":
        negatives = candidates[:max_negatives]
        if not negatives:
            return []
        candidate_ids = ["NIL"] + [f"{c['id']}:{c.get('desc') or c.get('label') or c['id']}" for c in negatives]
        random.shuffle(candidate_ids)
        return [
            {
                "scenario": "nil",
                "prompt": format_prompt(mention, context, candidate_ids),
                "chosen": "The correct entity is NIL: no valid entity",
                "rejected": [format_answer(c["id"], c.get("desc") or c.get("label", "")) for c in negatives],
            }
        ]

    chosen = next((c for c in candidates if c["id"] == gold_qid), {"id": gold_qid, "label": mention, "desc": mention})
    negatives = [c for c in candidates if c["id"] != gold_qid][:max_negatives]
    if not negatives:
        return []
    candidate_ids = [f"{chosen['id']}:{chosen.get('desc') or chosen.get('label') or chosen['id']}"]
    candidate_ids += [f"{c['id']}:{c.get('desc') or c.get('label') or c['id']}" for c in negatives]
    random.shuffle(candidate_ids)
    return [
        {
            "scenario": "standard",
            "prompt": format_prompt(mention, context, candidate_ids),
            "chosen": format_answer(chosen["id"], chosen.get("desc") or chosen.get("label", "")),
            "rejected": [format_answer(c["id"], c.get("desc") or c.get("label", "")) for c in negatives],
        }
    ]


async def build_dpo_data_from_hipe_tsv(
    tsv_path: str,
    output_file: str,
    language: str,
    model: str,
    base_url: str,
    chunk_size: int = 256,
    max_negatives: int = 3,
    max_candidates: int = 8,
    concurrency: int = 8,
) -> Path:
    chunks = process_hipe_tsv(tsv_path, chunk_size=chunk_size)
    entities = []
    for chunk in chunks:
        context = " ".join(chunk["tokens"])
        for ent in extract_entities(chunk["tokens"], chunk["el"]):
            entities.append(
                {
                    "mention": ent["entity"],
                    "context": context,
                    "gold_qid": ent["label"],
                }
            )

    agent = build_retrieval_agent(model, base_url, max_candidates)
    wd = WikidataClient([language, "en"], limit=5, max_concurrent=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def process(item: dict, session: aiohttp.ClientSession) -> list[dict]:
        async with sem:
            names = await suggest_candidate_names(agent, item["mention"], item["context"])
            qids_nested = await asyncio.gather(*(wd.search_multilang(session, n) for n in names[:max_candidates]))
        qids = list(dict.fromkeys(q for group in qids_nested for q in group))
        candidates = [{"id": qid, "label": qid, "desc": qid} for qid in qids]
        return build_multineg_records(
            item["mention"],
            item["context"],
            item["gold_qid"],
            candidates,
            max_negatives=max_negatives,
        )

    out = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        for batch_start in range(0, len(entities), concurrency):
            batch = entities[batch_start : batch_start + concurrency]
            nested = await asyncio.gather(*(process(item, session) for item in batch))
            out.extend(record for group in nested for record in group)

    write_jsonl(out, output_file)
    return Path(output_file)


def to_mdpo_candidate_format(record: dict) -> dict:
    """Convert prompt/chosen/rejected rows to candidate-list rows for MDPO loss."""
    answers = [record["chosen"]] + list(record.get("rejected", []))
    return {
        "prompt": record["prompt"],
        "candidate_texts": answers,
        "labels": [1] + [0] * (len(answers) - 1),
        "chosen": record["chosen"],
        "rejected": record.get("rejected", ["NIL"])[0],
        "coverage": "NIL" not in record["chosen"].upper(),
    }


def convert_jsonl_to_mdpo(input_file: str, output_file: str) -> Path:
    rows = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(to_mdpo_candidate_format(json.loads(line)))
    write_jsonl(rows, output_file)
    return Path(output_file)
