"""
Suggestion pipeline — multi-language Wikidata search + alias dictionary lookup.

Alias dict (from newpipeline) is checked first as a fast, high-precision step.
Matching QIDs are promoted to the front of the retrievals list, then the LLM
+ multi-lang Wikidata search fills in the rest.

Usage examples
--------------
# defaults (fr, newseye)
python suggestion_pipeline.py

# German run with custom alias dict
python suggestion_pipeline.py --language de --dataset_name newseye \
    --alias_dict alias_dictionary_multilingual.json

# custom model / output dir
python suggestion_pipeline.py --language fr --output_dir runs/exp1 \
    --vllm_model openai/my-model --max_candidates 15

# skip alias dict entirely
python suggestion_pipeline.py --no_alias
"""

import os
import json
import asyncio
import datetime
import time
import argparse
import requests
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner
from pydantic import BaseModel, Field
from typing import List, Dict, Set, Optional

# ── Step 1 — CLI args + Config ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Wikidata suggestion pipeline with alias dict + multi-language search.'
    )
    parser.add_argument('--dataset_name',     default='newseye',
                        help='Dataset name used in file paths (default: newseye)')
    parser.add_argument('--language',         default='fr',
                        help='Dataset language code, e.g. fr / de / nl (default: fr)')
    parser.add_argument('--input_dir',        default='Process/tmp_2303',
                        help='Directory containing the input JSONL (default: Process/tmp)')
    parser.add_argument('--output_dir',       default='Suggestion_v2',
                        help='Directory for output JSONL + state file (default: Suggestion_v2)')
    parser.add_argument('--vllm_model',       default='openai/gpt-oss-20b',
                        help='vLLM model name (default: openai/gpt-oss-20b)')
    parser.add_argument('--vllm_url',         default='http://0.0.0.0:8000/v1',
                        help='vLLM base URL (default: http://0.0.0.0:8000/v1)')
    parser.add_argument('--max_candidates',   default=10, type=int,
                        help='Max candidates in final retrievals list (default: 10)')
    parser.add_argument('--max_retries',      default=3,  type=int,
                        help='Max LLM call retries per entity (default: 3)')
    parser.add_argument('--retry_delay',      default=1.0, type=float,
                        help='Seconds between LLM retries (default: 1.0)')
    parser.add_argument('--search_languages', default='fr,en,de,la',
                        help='Comma-separated Wikidata search languages in priority order '
                             '(default: fr,en,de,la)')
    parser.add_argument('--alias_dict',       default='alias_dictionary_multilingual.json',
                        help='Path to alias dictionary JSON built from training TSVs '
                             '(default: alias_dictionary_multilingual.json)')
    parser.add_argument('--no_alias',         action='store_true',
                        help='Skip alias dictionary lookup entirely')
    parser.add_argument('--hipe_data_path',   default='../../../HIPE-2022-data/data/v2.1',
                        help='Base path to HIPE-2022 TSV data — used to build alias dict '
                             'if the JSON file does not exist yet '
                             '(default: ../../../HIPE-2022-data/data/v2.1)')
    parser.add_argument("--no_nil_classifier", action="store_true",
                        help="Skip LLM NIL classifier — fall back to empty-retrieval heuristic")
    parser.add_argument("--mock", action="store_true",
                        help=(
                            "Dry-run / mock mode: skip all LLM calls and Wikidata HTTP requests. "
                            "Uses synthetic candidate strings so you can verify the full "
                            "suggestion→selection data flow without a running model or network."
                        ))
    return parser.parse_args()


args = parse_args()

dataset_name     = args.dataset_name
language         = args.language
VLLM_MODEL       = args.vllm_model
VLLM_URL         = args.vllm_url
MAX_CANDIDATES   = args.max_candidates
MAX_RETRIES      = args.max_retries
RETRY_DELAY      = args.retry_delay
SEARCH_LANGUAGES = [l.strip() for l in args.search_languages.split(',')]
USE_ALIAS          = not args.no_alias
USE_NIL_CLASSIFIER = not args.no_nil_classifier
MOCK_MODE          = args.mock

input_file  = os.path.join(args.input_dir, f'{dataset_name}-test-{language}.jsonl')
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, f'gpt-oss-20b-{dataset_name}-{language}-retrieved.jsonl')
state_path  = os.path.join(args.output_dir, f'gpt-oss-20b-{dataset_name}-{language}-retrieved.state.json')

print(f'Dataset        : {dataset_name} / {language}')
print(f'Input          : {input_file}')
print(f'Output         : {output_path}')
print(f'Model          : {VLLM_MODEL}  @  {VLLM_URL}')
print(f'Max candidates : {MAX_CANDIDATES}')
print(f'Search langs   : {SEARCH_LANGUAGES}')
print(f'Alias dict     : {"disabled" if not USE_ALIAS else args.alias_dict}')
print(f'NIL classifier: {"enabled" if USE_NIL_CLASSIFIER else "disabled (heuristic)"}')
if MOCK_MODE:
    print()
    print('⚠️  MOCK MODE ENABLED — no LLM calls, no Wikidata requests.')
    print('   Synthetic candidates will be written so you can test the full pipeline flow.')
print()

# ── Step 2 — Alias dictionary (from newpipeline) ──────────────────────────────

HEADERS = {'User-Agent': 'MyWikidataClient/1.0 (myemail@example.com)'}


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_alias_from_tsv(tsv_file_path: str, alias_dict: Dict[str, Set[str]]):
    """Extract mention→QID mappings from one HIPE TSV file."""
    from hipe_commons.helpers.tsv import tsv_to_dict
    tsv_data = tsv_to_dict(tsv_file_path)
    tokens = tsv_data['TOKEN']
    labels = tsv_data['NE-COARSE-LIT']
    nel    = tsv_data['NEL-LIT']
    i, total = 0, len(tokens)
    while i < total:
        if labels[i] != '_':
            entity_tokens, qids, label = [], [], labels[i]
            while i < total and labels[i] == label:
                entity_tokens.append(tokens[i])
                qids.append(nel[i])
                i += 1
            qid = next((q for q in qids if q not in ('_', 'NIL')), None)
            if qid:
                alias_dict[normalize(' '.join(entity_tokens))].add(qid)
        else:
            i += 1


def build_alias_dict(base_path: str) -> Dict[str, List[str]]:
    """Build alias dict from all available training TSVs."""
    alias: Dict[str, Set[str]] = defaultdict(set)
    datasets  = ['newseye', 'hipe2020', 'hipe']
    languages = ['fr', 'de', 'sv', 'fi', 'nl']
    for ds in datasets:
        for lang in languages:
            path = f'{base_path}/{ds}/{lang}/HIPE-2022-v2.1-{ds}-train-{lang}.tsv'
            if os.path.exists(path):
                print(f'  [alias] Processing: {path}')
                extract_alias_from_tsv(path, alias)
            else:
                print(f'  [alias] Skipping (not found): {path}')
    return {k: list(v) for k, v in alias.items()}


def load_alias_dict(path: str, hipe_data_path: str) -> Dict[str, List[str]]:
    """Load alias dict from JSON, or build+save it if the file is missing."""
    if os.path.exists(path):
        print(f'[alias] Loading alias dict from {path}')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        print(f'[alias] Loaded {len(data):,} aliases')
        return data
    print(f'[alias] {path} not found — building from TSV data at {hipe_data_path}')
    data = build_alias_dict(hipe_data_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'[alias] Saved {len(data):,} aliases to {path}')
    return data


_qid_label_cache: Dict[str, str] = {}


def _fetch_wikidata_label(qid: str) -> str:
    """Fetch the best available label for a QID (result cached by caller)."""
    try:
        resp = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={
                'action':    'wbgetentities',
                'ids':       qid,
                'props':     'labels',
                'languages': '|'.join(SEARCH_LANGUAGES),
                'format':    'json',
            },
            headers=HEADERS, timeout=8,
        )
        entity = resp.json().get('entities', {}).get(qid, {})
        labels = entity.get('labels', {})
        for lang in SEARCH_LANGUAGES:
            if lang in labels:
                return labels[lang]['value']
        return qid
    except Exception:
        return qid


def alias_to_retrieval_strings(qids: List[str]) -> List[str]:
    """
    Convert raw QIDs from alias dict into the unified
    'label - QID - description' format used by the rest of the pipeline.
    Labels are fetched from Wikidata on demand and cached.
    """
    results = []
    for qid in qids:
        if qid not in _qid_label_cache:
            _qid_label_cache[qid] = _fetch_wikidata_label(qid)
        label = _qid_label_cache[qid]
        results.append(f'{label} - {qid} - [alias match]')
    return results


# Load alias dict (or skip if --no_alias)
ALIAS_DICT: Dict[str, List[str]] = {}
if USE_ALIAS:
    ALIAS_DICT = load_alias_dict(args.alias_dict, args.hipe_data_path)

# ── Step 3 — Load pre-built JSONL ─────────────────────────────────────────────

dataset = load_dataset('json', data_files=input_file)['train']
print(f'Loaded {len(dataset)} entity records')
print()
print('Sample record:')
r = dataset[0]
for k, v in r.items():
    if k == 'meta':
        print(f'  {k:20s}: {{...{len(v)} keys}}')
    else:
        print(f'  {k:20s}: {str(v)[:100]}')

# ── Step 4 — Pydantic schema + vLLM agent ─────────────────────────────────────

class WikiSearchResult(BaseModel):
    """Candidate entity names suggested by the LLM for Wikidata lookup."""
    candidates: List[str] = Field(
        default_factory=list,
        description=(
            f'List of up to {MAX_CANDIDATES} normalised candidate name strings '
            'that could correspond to valid Wikidata items. '
            'Each entry is a clean human-readable string, no QIDs.'
        )
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

vllm_client = AsyncOpenAI(base_url=VLLM_URL, api_key='not-needed')

suggestion_agent = Agent(
    name='Wiki Candidate Agent',
    instructions=SEARCH_WIKI_PROMPT,
    model=OpenAIResponsesModel(
        model=VLLM_MODEL,
        openai_client=vllm_client,
    ),
    output_type=WikiSearchResult,
)

# ── NIL Classifier agent ───────────────────────────────────────────────────────

class NILDecision(BaseModel):
    """Binary NIL/LINK decision for an entity mention."""
    is_nil: bool = Field(
        description=(
            'True if the entity is a NIL entity (not linkable to any Wikidata item). '
            'False if at least one candidate is a plausible match.'
        )
    )

NIL_CLASSIFIER_PROMPT = """
You are an expert Wikidata entity linker for historical newspaper text (approx. 1800–1940).
Given a named entity mention, its context, and a list of Wikidata candidates,
decide whether the entity is NIL (cannot be linked to any Wikidata item).

Rules:
- Set is_nil=true  if NO candidate plausibly matches the mention in this context
- Set is_nil=true  if the mention is a generic common noun, partial name, or
  clearly refers to a very local/obscure person or place not in Wikidata
- Set is_nil=false if at least one candidate is a plausible match given the context
- Use the document period to judge whether a historical form matches a candidate
- Treat OCR noise charitably — a garbled mention can still be linkable

Mode: No Reasoning
"""

nil_agent = Agent(
    name='NIL Classifier Agent',
    instructions=NIL_CLASSIFIER_PROMPT,
    model=OpenAIResponsesModel(
        model=VLLM_MODEL,
        openai_client=vllm_client,
    ),
    output_type=NILDecision,
)


async def classify_nil(
    entity:     str,
    context:    str,
    retrievals: List[str],
    period:     Optional[str] = None,
) -> bool:
    """
    Ask the LLM whether `entity` is a NIL entity given the retrieved candidates.

    Falls back to the heuristic (empty retrievals → NIL) if:
    - USE_NIL_CLASSIFIER is False
    - retrievals list is empty (nothing to judge against)
    - LLM call fails after retries
    """
    if not USE_NIL_CLASSIFIER or not retrievals:
        # heuristic: no candidates retrieved → treat as NIL
        return len(retrievals) == 0

    cand_block = '\n'.join(f'  - {r}' for r in retrievals[:5])
    period_str = str(period)[:4] if period else 'unknown'
    prompt = (
        f"Entity mention : '{entity}'\n"
        f"Document period: {period_str}\n"
        f"Context        : '{context[:250]}'\n"
        f"Candidates:\n{cand_block}"
    )
    try:
        result = await Runner.run(nil_agent, prompt)
        if result and getattr(result, 'final_output', None):
            return result.final_output.is_nil
    except Exception as e:
        print(f'  [NIL classifier] Error for "{entity}": {e} — falling back to heuristic')
    # fallback
    return len(retrievals) == 0


# ── Step 5 — Wikidata lookup helpers ──────────────────────────────────────────

WIKIDATA_RETRIES = 3
WIKIDATA_BACKOFF = 2.0


def wikidata_search(
    query:    str,
    language: str = 'en',
    limit:    int = 5,
    retries:  int = WIKIDATA_RETRIES,
    backoff:  float = WIKIDATA_BACKOFF,
) -> List[str]:
    """
    Search Wikidata for query and return formatted strings:
        "<label> - <QID> - <description>"
    Retries on timeout / rate-limit with exponential backoff.
    Returns [] if all attempts fail.
    """
    params = {
        'action':   'wbsearchentities',
        'search':   query,
        'language': language,
        'format':   'json',
        'type':     'item',
        'limit':    limit,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                'https://www.wikidata.org/w/api.php',
                params=params, headers=HEADERS, timeout=10,
            )
            if resp.status_code == 429:
                wait = backoff * attempt
                print(f'  [WARN] Wikidata rate-limited, waiting {wait:.1f}s ...')
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f'  [WARN] Wikidata HTTP {resp.status_code} for "{query}"')
                return []
            results = []
            for item in resp.json().get('search', []):
                label = item.get('label', '').strip()
                qid   = item.get('id',    '').strip()
                desc  = item.get('description', '').strip()
                results.append(f'{label} - {qid} - {desc}')
            return results

        except requests.exceptions.Timeout:
            wait = backoff * attempt
            print(f'  [WARN] Wikidata timeout for "{query}" '
                  f'(attempt {attempt}/{retries}), retrying in {wait:.1f}s ...')
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = backoff * attempt
            print(f'  [WARN] Wikidata connection error for "{query}" '
                  f'(attempt {attempt}/{retries}): {e}, retrying in {wait:.1f}s ...')
            time.sleep(wait)
        except Exception as e:
            print(f'  [WARN] Wikidata unexpected error for "{query}": {e}')
            return []

    print(f'  [WARN] Wikidata gave up after {retries} attempts for "{query}"')
    return []


def wikidata_search_multilang(
    query:     str,
    languages: List[str] = SEARCH_LANGUAGES,
    limit:     int = 5,
) -> List[str]:
    """
    Search Wikidata across multiple languages, deduplicated by QID.
    Returns merged list ordered by first occurrence across languages.
    """
    seen_qids: Set[str] = set()
    results:   List[str] = []
    for lang in languages:
        for entry in wikidata_search(query, language=lang, limit=limit):
            parts = entry.split(' - ')
            qid = parts[1].strip() if len(parts) >= 2 else ''
            if qid and qid not in seen_qids:
                seen_qids.add(qid)
                results.append(entry)
    return results


def get_kb_id_from_result(completion) -> List[str]:
    """Safely extract candidate list from agent output."""
    if completion and getattr(completion, 'final_output', None):
        try:
            return completion.final_output.candidates or []
        except Exception:
            pass
    return []

# ── Mock helpers (--mock mode) ────────────────────────────────────────────────

def _mock_wikidata_candidates(entity: str, n: int = 3) -> List[str]:
    """
    Return deterministic fake Wikidata candidates for *entity*.
    Format matches the real pipeline: "label - QID - description".
    QIDs are stable across runs (derived from the entity string) so
    downstream selection_pipeline.py can pattern-match them.
    """
    base = abs(hash(entity)) % 900_000 + 100_000  # reproducible 6-digit offset
    return [
        f'{entity} (mock #{i + 1}) - Q{base + i} - mock description for {entity}'
        for i in range(n)
    ]


async def _mock_classify_nil(entity: str, retrievals: List[str]) -> bool:
    """Always return False (not NIL) in mock mode unless retrievals is empty."""
    return len(retrievals) == 0


# ── Step 6 — Suggestion runner (alias first, then LLM + Wikidata) ─────────────

async def run_suggestion(
    entity:      str,
    context:     str,
    period:      Optional[str] = None,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> List[str]:
    """
    Retrieve Wikidata candidates for `entity` using two complementary sources:

    1. Alias dictionary  — exact-match on normalised mention → high-precision QIDs
                           promoted to the front of the list (fast, no LLM call)
    2. LLM + Wikidata    — LLM generates name variants, looked up on Wikidata
                           across multiple languages (covers OCR noise + unseen entities)

    Returns a deduplicated list of "<label> - <QID> - <desc>" strings,
    alias hits first, capped at MAX_CANDIDATES.
    """
    # ── 1. Alias dict lookup ──────────────────────────────────────────────────
    alias_hits: List[str] = []
    alias_qids: Set[str]  = set()
    if USE_ALIAS:
        qids = ALIAS_DICT.get(normalize(entity), [])
        if qids:
            alias_hits = alias_to_retrieval_strings(qids)
            alias_qids = {e.split(' - ')[1].strip() for e in alias_hits}
            print(f'  [alias] {len(alias_hits)} hit(s) for "{entity}"')

    # ── 2. LLM candidate generation + multi-lang Wikidata search ─────────────
    period_hint = f'\nDocument period: {str(period)[:4]}' if period else ''
    prompt = (
        f"Entity mention : '{entity}'"
        f"{period_hint}"
        f"\nSummary        : '{context}'"
    )

    wiki_hits: List[str] = []
    if MOCK_MODE:
        # Skip LLM + Wikidata entirely; use deterministic synthetic candidates
        wiki_hits = _mock_wikidata_candidates(entity, n=MAX_CANDIDATES)
        print(f'  [mock] Generated {len(wiki_hits)} synthetic candidate(s) for "{entity}"')
    else:
        for attempt in range(1, max_retries + 1):
            try:
                result     = await Runner.run(suggestion_agent, prompt)
                candidates = get_kb_id_from_result(result)[:MAX_CANDIDATES]
                for cand in candidates:
                    wiki_hits.extend(wikidata_search_multilang(cand))
                break
            except Exception as e:
                print(f'  [attempt {attempt}/{max_retries}] Error for "{entity}": {e}')
                if attempt == max_retries:
                    print('  Max retries reached — using alias hits only.')
                else:
                    await asyncio.sleep(retry_delay)

    # ── 3. Merge: alias hits first, Wikidata hits deduplicated after ──────────
    seen_qids: Set[str] = set(alias_qids)
    merged: List[str]   = list(alias_hits)

    for entry in wiki_hits:
        parts = entry.split(' - ')
        qid = parts[1].strip() if len(parts) >= 2 else ''
        if qid and qid not in seen_qids:
            seen_qids.add(qid)
            merged.append(entry)

    return merged[:MAX_CANDIDATES]

# ── Step 7 — Load checkpoint ──────────────────────────────────────────────────

start_idx = 0
if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    start_idx = state.get('last_idx', 0) + 1
    print(f'Resuming from record index {start_idx}')
else:
    print('Starting fresh run')

outfile = open(output_path, 'a', encoding='utf-8')


class SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)

# ── Step 8 — Main loop ────────────────────────────────────────────────────────

async def main():
    alias_hit_count = 0

    for idx in tqdm(range(start_idx, len(dataset)), desc='Suggestion', initial=start_idx):
        item   = dataset[idx]
        entity = item['input_candidate']
        gold   = item['gold_qid']
        period = str(item.get('period', '') or '') or None

        summary = item.get('summary', entity)
        context = summary if summary and summary != entity else item['context']

        print(f'\n[{idx}] {entity!r}  (gold: {gold})')
        print(f'  context used: {context[:120]}')

        retrievals   = await run_suggestion(entity=entity, context=context, period=period)
        alias_fired  = USE_ALIAS and normalize(entity) in ALIAS_DICT
        if alias_fired:
            alias_hit_count += 1

        # ── NIL classification: use LLM classifier for NIL entities ────────
        if gold == 'NIL':
            if MOCK_MODE:
                predicted_nil = await _mock_classify_nil(entity, retrievals)
            else:
                predicted_nil = await classify_nil(
                    entity     = entity,
                    context    = context,
                    retrievals = retrievals,
                    period     = period,
                )
            is_correct = predicted_nil
        else:
            is_correct = any(gold in r for r in retrievals)

        nil_label = f'  predicted_nil={predicted_nil}' if gold == 'NIL' else ''
        print(f'  retrievals ({len(retrievals)}): {retrievals[:3]} ...')
        print(f'  correct: {is_correct}  |  alias fired: {alias_fired}{nil_label}')

        record = {
            'context':         item['context'],
            'input_candidate': entity,
            'gold_qid':        gold,
            'retrievals':      retrievals,
            'abs_token_start': item['abs_token_start'],
            'abs_token_end':   item['abs_token_end'],
            'summary':         summary,
            'period':          period,
            'meta':            item.get('meta', {}),
            'is_correct':      is_correct,
            'alias_fired':     alias_fired,
            'predicted_nil':   (predicted_nil if gold == 'NIL' else None),
        }
        outfile.write(json.dumps(record, ensure_ascii=False, cls=SafeEncoder) + '\n')
        outfile.flush()

        with open(state_path, 'w') as f:
            json.dump({'last_idx': idx}, f)

    outfile.close()
    print(f'\nDone — saved to: {output_path}')
    if USE_ALIAS:
       print(f'Alias dict fired on {alias_hit_count}/{len(dataset)} entities '
              f'({100*alias_hit_count/len(dataset):.1f}%)')

# ── Step 9 — Accuracy report ──────────────────────────────────────────────────

def print_report():
    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total      = len(records)
    correct    = sum(1 for r in records if r.get('is_correct'))
    nil_total  = sum(1 for r in records if r['gold_qid'] == 'NIL')
    nil_ok     = sum(1 for r in records if r['gold_qid'] == 'NIL' and r.get('is_correct'))
    empty_ret  = sum(1 for r in records if len(r['retrievals']) == 0)
    alias_ok   = sum(1 for r in records if r.get('alias_fired') and r.get('is_correct'))

    nil_clf_used = sum(1 for r in records
                       if r['gold_qid'] == 'NIL' and r.get('predicted_nil') is not None)

    print(f'Total entities      : {total}')
    print(f'Recall@{MAX_CANDIDATES:<2}           : {correct}/{total}  ({100*correct/total:.1f}%)')
    print(f'NIL recall          : {nil_ok}/{nil_total}  ({100*nil_ok/nil_total:.1f}% of NIL)' if nil_total else 'NIL recall          : n/a')
    print(f'NIL classifier used : {nil_clf_used}/{nil_total} entities  (rest used heuristic)')
    print(f'Empty retrievals    : {empty_ret}  (LLM/Wikidata found nothing)')
    if USE_ALIAS:
        print(f'Alias-assisted hits : {alias_ok}  (correct AND alias fired)')
    print()
    print('Sample enriched record:')
    r = records[0]
    for k, v in r.items():
        if k in ('context', 'meta'):
            print(f'  {k:20s}: {str(v)[:80]}')
        else:
            print(f'  {k:20s}: {v}')


if __name__ == '__main__':
    asyncio.run(main())
    print_report()