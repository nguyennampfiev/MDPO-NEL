import asyncio
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from agents import Agent, OpenAIResponsesModel, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .data import read_jsonl, write_jsonl
from .alias import load_or_build_alias_dictionary, promote_alias_qids
from .prompts import RETRIEVAL_PROMPT
from .wikidata import WikidataClient


class WikiSearchResult(BaseModel):
    candidates: list[str] = Field(default_factory=list)


@dataclass
class RetrievalConfig:
    input_file: str
    output_dir: str
    dataset_name: str
    language: str
    model: str
    base_url: str
    split: str = "test"
    max_candidates: int = 8
    search_languages: str | None = None
    alias_dict: str | None = None
    hipe_root: str | None = None
    entity_concurrency: int = 8
    batch_size: int = 32


def build_retrieval_agent(model: str, base_url: str, max_candidates: int) -> Agent:
    instructions = RETRIEVAL_PROMPT + f"\nReturn at most {max_candidates} candidates."
    return Agent(
        name="Wiki Candidate Agent",
        instructions=instructions,
        model=OpenAIResponsesModel(
            model=model,
            openai_client=AsyncOpenAI(base_url=base_url, api_key="not-needed"),
        ),
        output_type=WikiSearchResult,
    )


async def suggest_candidate_names(agent: Agent, mention: str, context: str, period: str | None = None) -> list[str]:
    period_hint = f"\nDocument period: {period[:4]}" if period else ""
    prompt = f"Entity mention: {mention!r}{period_hint}\nContext: {context!r}"
    result = await Runner.run(agent, prompt)
    final = getattr(result, "final_output", None)
    return list(getattr(final, "candidates", []) or [])


async def run_retrieval(config: RetrievalConfig) -> Path:
    records = read_jsonl(config.input_file)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = config.model.replace("openai/", "").replace("/", "-")
    out_path = out_dir / f"{model_slug}-{config.dataset_name}-{config.language}-retrieved.jsonl"
    state_path = out_dir / f"{model_slug}-{config.dataset_name}-{config.language}-retrieved.state.json"

    languages = (
        config.search_languages.split(",")
        if config.search_languages
        else [config.language, "en"]
    )
    agent = build_retrieval_agent(config.model, config.base_url, config.max_candidates)
    wd = WikidataClient(languages=languages, limit=5, max_concurrent=config.entity_concurrency)
    aliases = load_or_build_alias_dictionary(config.alias_dict, config.hipe_root)
    semaphore = asyncio.Semaphore(config.entity_concurrency)

    start_idx = -1
    if state_path.exists():
        import json

        start_idx = json.loads(state_path.read_text()).get("last_idx", -1)

    async def process(idx: int, item: dict, session: aiohttp.ClientSession) -> dict:
        mention = item["input_candidate"]
        context = item.get("summary") or item["context"]
        async with semaphore:
            names = await suggest_candidate_names(agent, mention, context, item.get("period"))
            qid_groups = await asyncio.gather(*(wd.search_multilang(session, name) for name in names[: config.max_candidates]))
        seen = set()
        retrievals = []
        for group in qid_groups:
            for qid in group:
                if qid not in seen:
                    seen.add(qid)
                    retrievals.append(qid)
        retrievals, alias_fired = promote_alias_qids(
            mention,
            retrievals,
            aliases,
            max_candidates=config.max_candidates,
        )
        out = dict(item)
        out["retrievals"] = retrievals
        out["alias_fired"] = alias_fired
        out["is_correct"] = item.get("gold_qid") in out["retrievals"] if item.get("gold_qid") != "NIL" else not out["retrievals"]
        return out

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        pending = [(i, r) for i, r in enumerate(records) if i > start_idx]
        for offset in range(0, len(pending), config.batch_size):
            batch = pending[offset : offset + config.batch_size]
            batch_records = await asyncio.gather(*(process(i, item, session) for i, item in batch))
            write_jsonl(batch_records, out_path, append=True)
            state_path.write_text(__import__("json").dumps({"last_idx": batch[-1][0]}))

    return out_path
