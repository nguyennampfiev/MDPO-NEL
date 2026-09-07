import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from agents import Agent, OpenAIResponsesModel, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .data import read_jsonl, write_predictions_to_hipe_tsv
from .prompts import SELECTION_PROMPT
from .wikidata import WikidataClient

QID_RE = re.compile(r"\bQ\d+\b")


class EntityLink(BaseModel):
    entity: str
    kb_id: str | None = Field(default=None)


class NELResult(BaseModel):
    entities: list[EntityLink] = Field(default_factory=list)


@dataclass
class SelectionConfig:
    input_file: str
    output_dir: str
    tsv_path: str
    language: str
    model: str
    base_url: str
    run_tag: str = "selection"
    agent_concurrency: int = 8


def build_selection_agent(model: str, base_url: str) -> Agent:
    return Agent(
        name="NEL Selector",
        instructions=SELECTION_PROMPT,
        model=OpenAIResponsesModel(
            model=model,
            openai_client=AsyncOpenAI(base_url=base_url, api_key="not-needed"),
        ),
        output_type=NELResult,
    )


def extract_qid(text: str | None) -> str:
    if not text:
        return "NIL"
    text = str(text).strip()
    if text.upper() == "NIL":
        return "NIL"
    match = QID_RE.search(text)
    return match.group(0) if match else "NIL"


async def select_entity(
    agent: Agent,
    wikidata: WikidataClient,
    session: aiohttp.ClientSession,
    mention: str,
    context: str,
    candidate_qids: list[str],
    language: str,
    semaphore: asyncio.Semaphore,
) -> str:
    if not candidate_qids:
        return "NIL"

    info_nested = await asyncio.gather(
        *(wikidata.get_entity_info(session, qid, language) for qid in candidate_qids)
    )
    candidate_info = [item for group in info_nested for item in group]
    if not candidate_info:
        return "NIL"

    prompt = f"Mention: {mention!r}\nContext: {context!r}\nCandidates: {candidate_info!r}"
    async with semaphore:
        result = await Runner.run(agent, prompt)
    final = getattr(result, "final_output", None)
    if final and getattr(final, "entities", None):
        return extract_qid(final.entities[0].kb_id)
    return "NIL"


async def run_selection(config: SelectionConfig) -> Path:
    records = read_jsonl(config.input_file)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{config.run_tag}.jsonl"
    flat_path = out_dir / f"{config.run_tag}-flat.json"
    tsv_out = out_dir / f"{config.run_tag}-predictions.tsv"

    agent = build_selection_agent(config.model, config.base_url)
    wikidata = WikidataClient([config.language, "en"], max_concurrent=config.agent_concurrency)
    semaphore = asyncio.Semaphore(config.agent_concurrency)

    max_end = max((r.get("abs_token_end", 0) for r in records), default=0)
    token_preds = ["_"] * max_end
    token_golds = ["_"] * max_end

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        out_rows = []
        for item in records:
            pred = await select_entity(
                agent,
                wikidata,
                session,
                item["input_candidate"],
                item["context"],
                item.get("retrievals", []),
                config.language,
                semaphore,
            )
            row = {
                "entity": item["input_candidate"],
                "token_start": item.get("abs_token_start"),
                "token_end": item.get("abs_token_end"),
                "prediction": pred,
                "ground_truth": item.get("gold_qid"),
                "candidates": item.get("retrievals", []),
            }
            out_rows.append(row)
            for i in range(row["token_start"], row["token_end"]):
                token_preds[i] = pred
                token_golds[i] = row["ground_truth"]

    with open(pred_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    flat_path.write_text(json.dumps({"token_preds": token_preds, "token_golds": token_golds}, ensure_ascii=False) + "\n")
    write_predictions_to_hipe_tsv(config.tsv_path, tsv_out, token_preds)
    return tsv_out
