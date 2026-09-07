"""
Suggestion pipeline v2 — FAST edition
======================================
Best of both worlds:
  • Input  : pre-built JSONL with 'summary' field  (from suggestion_pipeline_v2)
  • Speed  : async aiohttp, parallel entities, multi-lang search, Wikidata cache
             (from suggestion_pipeline_fast)
  • Safety : checkpoint / resume support            (from suggestion_pipeline_v2)
  • Output : "label - QID - desc" formatted strings (from suggestion_pipeline_v2)

Speed improvements
------------------
  1. Parallel entity processing     — asyncio.gather + semaphore per batch
  2. Non-blocking Wikidata calls    — aiohttp instead of requests
  3. Multi-language parallel search — concurrent per-language calls
  4. Wikidata string-level cache    — same candidate never fetched twice

Usage examples
--------------
python suggestion_pipeline_v2_fast.py
python suggestion_pipeline_v2_fast.py --dataset_name hipe2020 --language fr
python suggestion_pipeline_v2_fast.py \\
    --dataset_name hipe2020 \\
    --language fr \\
    --vllm_model openai/gpt-oss-20b \\
    --vllm_url http://0.0.0.0:8001/v1 \\
    --max_candidates 8 \\
    --input_file Process/tmp/hipe2020-test-fr.jsonl \\
    --output_dir Suggestion_v2_fast \\
    --entity_concurrency 8
"""

import os
import json
import asyncio
import argparse
import datetime
import aiohttp
from tqdm import tqdm
from typing import List, Optional

# ── Step 0 — CLI args ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HIPE suggestion pipeline v2 — fast edition."
    )
    parser.add_argument("--dataset_name",       default="hipe2020")
    parser.add_argument("--language",           default="fr")
    parser.add_argument("--split",              default="test", choices=["train", "dev", "test"])
    parser.add_argument("--input_file",         default=None,
                        help="Path to pre-built JSONL; auto-derived if omitted")
    parser.add_argument("--output_dir",         default="Retrieval_Simple_Summary_2904")
    parser.add_argument("--vllm_model",         default="openai/gpt-oss-20b")
    parser.add_argument("--vllm_url",           default="http://0.0.0.0:8007/v1")
    parser.add_argument("--max_candidates",     default=8,    type=int)
    parser.add_argument("--max_retries",        default=5,    type=int)
    parser.add_argument("--retry_delay",        default=1.0,  type=float)
    parser.add_argument("--search_languages",   default=None,
                        help="Comma-separated languages for Wikidata search "
                             "(default: <language>,en)")
    parser.add_argument("--wikidata_limit",     default=5,    type=int,
                        help="Max results per language per candidate (default 5)")
    parser.add_argument("--wikidata_retries",   default=5,    type=int)
    parser.add_argument("--wikidata_backoff",   default=1.0,  type=float)
    parser.add_argument("--entity_concurrency", default=8,    type=int,
                        help="Max parallel entity tasks per batch (default 8)")
    parser.add_argument("--batch_size",         default=32,   type=int,
                        help="Entities per parallel batch (default 32)")
    return parser.parse_args()


args = parse_args()

DATASET_NAME       = args.dataset_name
LANGUAGE           = args.language
VLLM_MODEL         = args.vllm_model
VLLM_URL           = args.vllm_url
MAX_CANDIDATES     = args.max_candidates
MAX_RETRIES        = args.max_retries
RETRY_DELAY        = args.retry_delay
WIKIDATA_LIMIT     = args.wikidata_limit
WIKIDATA_RETRIES   = args.wikidata_retries
WIKIDATA_BACKOFF   = args.wikidata_backoff
ENTITY_CONCURRENCY = args.entity_concurrency
BATCH_SIZE         = args.batch_size
SEARCH_LANGUAGES   = (
    args.search_languages.split(",")
    if args.search_languages
    else [LANGUAGE, "en"]
)

# Deduplicate search languages while preserving order
_seen = set()
SEARCH_LANGUAGES = [
    lang for lang in SEARCH_LANGUAGES
    if not (lang in _seen or _seen.add(lang))
]

# Derive paths
if args.input_file:
    input_file = args.input_file
else:
    input_file = f"Process/tmp/{DATASET_NAME}-{args.split}-{LANGUAGE}.jsonl"

os.makedirs(args.output_dir, exist_ok=True)
model_slug  = VLLM_MODEL.replace("/", "-").replace("openai/", "")
output_path = os.path.join(
    args.output_dir,
    f"{model_slug}-{DATASET_NAME}-{LANGUAGE}-retrieved.jsonl",
)
state_path = os.path.join(
    args.output_dir,
    f"{model_slug}-{DATASET_NAME}-{LANGUAGE}-retrieved.state.json",
)

print(f"Dataset         : {DATASET_NAME} / {LANGUAGE} / {args.split}")
print(f"Input JSONL     : {input_file}")
print(f"Output JSONL    : {output_path}")
print(f"State file      : {state_path}")
print(f"Model           : {VLLM_MODEL}  @  {VLLM_URL}")
print(f"Search langs    : {SEARCH_LANGUAGES}")
print(f"Max candidates  : {MAX_CANDIDATES}")
print(f"Wikidata limit  : {WIKIDATA_LIMIT} results per lang per candidate")
print(f"Entity parallel : {ENTITY_CONCURRENCY} concurrent tasks  |  batch size {BATCH_SIZE}")
print()

# ── Step 1 — Pydantic schema + vLLM agent ────────────────────────────────────

from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner
from pydantic import BaseModel, Field


class WikiSearchResult(BaseModel):
    """Candidate entity names suggested by the LLM for Wikidata lookup."""
    candidates: List[str] = Field(
        default_factory=list,
        description=(
            f"List of up to {MAX_CANDIDATES} normalised candidate name strings "
            "that could correspond to valid Wikidata items. "
            "Each entry is a clean human-readable string, no QIDs."
        ),
    )


SEARCH_WIKI_PROMPT = f"""
You are an expert in Wikidata entity linking and historical linguistics.
Your goal is to generate a list of **potential candidate entity names** that
could correspond to valid Wikidata items, given a named entity mention and
a short summary describing it in its historical context.

Guidelines:
- The entity mention may contain typos, OCR errors, extra spaces, or
  unusual capitalisation — normalise it.
- Use the summary (and period if given) to infer the entity's historical
  or cultural background and generate more precise alternate forms.
- Return at most {MAX_CANDIDATES} candidate name strings.
- Each candidate must be a clean, human-readable string — no QIDs, no URLs.
- If no plausible candidates can be found, return just the original mention.
- Do not add markdown, commentary, or natural language outside the candidates.

Mode: No Reasoning
"""

vllm_client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

suggestion_agent = Agent(
    name="Wiki Candidate Agent",
    instructions=SEARCH_WIKI_PROMPT,
    model=OpenAIResponsesModel(
        model=VLLM_MODEL,
        openai_client=vllm_client,
    ),
    output_type=WikiSearchResult,
)

# ── Step 2 — Async + cached Wikidata lookup ───────────────────────────────────

#WIKIDATA_HEADERS = {"User-Agent": "MyWikidataClient/1.0 (myemail@example.com)"}
WIKIDATA_HEADERS  = {"User-Agent": "NEL_Pipeline_Bot/1.0 (mailto:namnamnamnam33@gmail.com)"}
WIKIDATA_RETRIES  = 5
WIKIDATA_BACKOFF  = 2.0
# ✅ Wikidata string-level cache: (query, language, limit) → ["label - QID - desc", ...]
_wikidata_cache: dict = {}
_wikidata_cache_lock = asyncio.Lock()


async def wikidata_search_async(
    query:    str,
    language: str = "en",
    limit:    int = WIKIDATA_LIMIT,
) -> List[str]:
    """
    Non-blocking Wikidata search returning bare QID strings e.g. ['Q220', 'Q142'].
    Results are cached by (query, language, limit) — safe because Wikidata always
    returns the same pool for the same string.
    """
    key = (query.lower().strip(), language, limit)

    async with _wikidata_cache_lock:
        if key in _wikidata_cache:
            return _wikidata_cache[key]

    params = {
        "action":   "wbsearchentities",
        "search":   query,
        "language": language,
        "format":   "json",
        "type":     "item",
        "limit":    limit,
    }

    for attempt in range(1, WIKIDATA_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.wikidata.org/w/api.php",
                    params=params,
                    headers=WIKIDATA_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 429:
                        wait = WIKIDATA_BACKOFF * (2 ** (attempt - 1))
                        print(f"  [WARN] Wikidata rate-limited, waiting {wait:.1f}s ...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        print(f"  [WARN] Wikidata HTTP {resp.status} for '{query}'")
                        return []
                    data = await resp.json()
                    results = [
                        item["id"]
                        for item in data.get("search", [])
                        if item.get("id")
                    ]

                    async with _wikidata_cache_lock:
                        _wikidata_cache[key] = results
                    return results

        except asyncio.TimeoutError:
            wait = WIKIDATA_BACKOFF * attempt
            print(f"  [WARN] Wikidata timeout for '{query}' "
                  f"(attempt {attempt}/{WIKIDATA_RETRIES}), retrying in {wait:.1f}s ...")
            await asyncio.sleep(wait)

        except aiohttp.ClientConnectionError as e:
            wait = WIKIDATA_BACKOFF * attempt
            print(f"  [WARN] Wikidata connection error for '{query}' "
                  f"(attempt {attempt}/{WIKIDATA_RETRIES}): {e}")
            await asyncio.sleep(wait)

        except Exception as e:
            print(f"  [WARN] Wikidata unexpected error for '{query}': {e}")
            return []

    print(f"  [WARN] Wikidata gave up after {WIKIDATA_RETRIES} attempts for '{query}'")
    return []


async def wikidata_search_multilang(
    query:     str,
    languages: List[str] = SEARCH_LANGUAGES,
    limit:     int       = WIKIDATA_LIMIT,
) -> List[str]:
    """Search Wikidata in all languages simultaneously and deduplicate."""
    per_lang = await asyncio.gather(*[
        wikidata_search_async(query, lang, limit) for lang in languages
    ])
    seen:    set       = set()
    results: List[str] = []
    for entries in per_lang:
        for entry in entries:
            if entry not in seen:
                seen.add(entry)
                results.append(entry)
    return results


# ── Step 3 — LLM + Wikidata suggestion runner ────────────────────────────────

def get_kb_id_from_result(completion) -> List[str]:
    """Safely extract candidate list from agent output."""
    if completion and getattr(completion, "final_output", None):
        try:
            return completion.final_output.candidates or []
        except Exception:
            pass
    return []


async def run_suggestion(
    entity:      str,
    context:     str,
    period:      Optional[str] = None,
    max_retries: int   = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> List[str]:
    """
    Ask the LLM to suggest up to MAX_CANDIDATES candidate names for `entity`,
    then look each one up on Wikidata (async, multi-language, cached).

    Returns
    -------
    Deduplicated list of "label - QID - desc" strings, capped at MAX_CANDIDATES.
    Empty list if nothing found.
    """
    period_hint = f"\nDocument period: {str(period)[:4]}" if period else ""
    prompt = (
        f"Entity mention : '{entity}'"
        f"{period_hint}"
        f"\nSummary        : '{context}'"
    )

    for attempt in range(1, max_retries + 1):
        try:
            result     = await Runner.run(suggestion_agent, prompt)
            candidates = get_kb_id_from_result(result)
            candidates = candidates[:MAX_CANDIDATES]

            # ✅ Parallel multi-language Wikidata lookup for all candidates at once
            all_entries = await asyncio.gather(*[
                wikidata_search_multilang(cand) for cand in candidates
            ])

            # Flatten + deduplicate, preserving order
            seen:   set       = set()
            unique: List[str] = []
            for entries in all_entries:
                for entry in entries:
                    if entry not in seen:
                        seen.add(entry)
                        unique.append(entry)

            return unique[:MAX_CANDIDATES]

        except Exception as e:
            print(f"  [attempt {attempt}/{max_retries}] Error for '{entity}': {e}")
            if attempt == max_retries:
                print("  Max retries reached — returning empty retrievals.")
                return []
            await asyncio.sleep(retry_delay)

    return []


# ── Step 4 — SafeEncoder ──────────────────────────────────────────────────────

class SafeEncoder(json.JSONEncoder):
    """Serialise datetime/date objects as ISO strings."""
    def default(self, o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)


# ── Step 5 — Main loop ────────────────────────────────────────────────────────

async def main():
    # Load dataset
    try:
        from datasets import load_dataset as hf_load
        dataset = hf_load("json", data_files=input_file)["train"]
        print(f"Loaded {len(dataset)} entity records (HuggingFace datasets)")
    except Exception:
        records_raw = []
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records_raw.append(json.loads(line))
        dataset = records_raw
        print(f"Loaded {len(dataset)} entity records (manual JSONL read)")

    print()
    print("Sample record:")
    sample = dataset[0]
    for k, v in sample.items():
        if k == "meta":
            print(f"  {k:20s}: {{...{len(v)} keys}}")
        else:
            print(f"  {k:20s}: {str(v)[:100]}")
    print()

    # Load checkpoint
    start_idx = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        start_idx = state.get("last_idx", 0) + 1
        print(f"Resuming from record index {start_idx}")
    else:
        print("Starting fresh run")

    # Semaphore caps concurrent LLM + Wikidata tasks
    semaphore = asyncio.Semaphore(ENTITY_CONCURRENCY)

    outfile = open(output_path, "a", encoding="utf-8")

    # ── per-entity coroutine ──────────────────────────────────────────────────
    async def process_entity(idx: int, item: dict) -> dict:
        entity  = item["input_candidate"]
        gold    = item["gold_qid"]
        period  = str(item.get("period", "") or "") or None
        summary = item.get("summary", entity)
        context = summary if summary and summary != entity else item["context"]

        print(f"\n[{idx}] {entity!r}  (gold: {gold})")
        print(f"  context used: {context[:120]}")

        async with semaphore:
            retrievals = await run_suggestion(
                entity  = entity,
                context = context,
                period  = period,
            )

        is_correct = (
            gold in retrievals          # exact QID match — no substring false-positives
            if gold != "NIL"
            else len(retrievals) == 0
        )

        print(f"  retrievals ({len(retrievals)}): {retrievals[:3]} ...")
        print(f"  correct: {is_correct}")

        return {
            # required by Selection_LLMs.ipynb
            "context":         item["context"],
            "input_candidate": entity,
            "gold_qid":        gold,
            "retrievals":      retrievals,
            "abs_token_start": item["abs_token_start"],
            "abs_token_end":   item["abs_token_end"],
            # bonus fields
            "summary":         summary,
            "period":          period,
            "meta":            item.get("meta", {}),
            "is_correct":      is_correct,
        }

    # ── batched parallel processing ───────────────────────────────────────────
    indices = list(range(start_idx, len(dataset)))

    for batch_start in tqdm(range(0, len(indices), BATCH_SIZE),
                            desc="Suggestion", unit="batch"):
        batch_indices = indices[batch_start: batch_start + BATCH_SIZE]
        batch_items   = [(i, dataset[i]) for i in batch_indices]

        # ✅ All entities in the batch run in parallel (semaphore limits concurrency)
        tasks   = [process_entity(i, item) for i, item in batch_items]
        records = await asyncio.gather(*tasks)

        # Write + checkpoint after each batch
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False, cls=SafeEncoder) + "\n")
        outfile.flush()

        last_idx = batch_indices[-1]
        with open(state_path, "w") as f:
            json.dump({"last_idx": last_idx}, f)

    outfile.close()
    print(f"\n✅ Done — saved to: {output_path}")
    print(f"   Wikidata cache entries: {len(_wikidata_cache)}")


# ── Step 6 — Accuracy report ──────────────────────────────────────────────────

def print_report():
    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total     = len(records)
    correct   = sum(1 for r in records if r.get("is_correct"))
    nil_total = sum(1 for r in records if r["gold_qid"] == "NIL")
    nil_ok    = sum(1 for r in records if r["gold_qid"] == "NIL" and r.get("is_correct"))
    empty_ret = sum(1 for r in records if len(r["retrievals"]) == 0)

    print(f"\n{'='*55}")
    print(f"Total entities      : {total}")
    if total:
        print(f"Recall@{MAX_CANDIDATES:<2}           : {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"NIL recall          : {nil_ok}/{nil_total}")
    print(f"Empty retrievals    : {empty_ret}  (LLM/Wikidata found nothing)")
    print()
    print("Sample enriched record:")
    if records:
        r = records[0]
        for k, v in r.items():
            if k in ("context", "meta"):
                print(f"  {k:20s}: {str(v)[:80]}")
            else:
                print(f"  {k:20s}: {v}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
    print_report()