import json, os, re, unicodedata
from typing import Dict, List

DATASETS = [
    ('newseye',  'fr'),
    ('hipe2020', 'fr'),
    ('newseye',  'de'),
    ('hipe2020', 'de'),
]
output_dir = 'Suggestion_2103'
alias_dict_path = 'alias_dictionary_multilingual.json'

# ── load alias dict ────────────────────────────────────────────────────────────
print(f'Loading alias dict from {alias_dict_path}')
with open(alias_dict_path, encoding='utf-8') as f:
    ALIAS_DICT: Dict[str, List[str]] = json.load(f)
print(f'Loaded {len(ALIAS_DICT):,} entries\n')

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', text.lower().strip())

# ── evaluate ───────────────────────────────────────────────────────────────────
for ds, lang in DATASETS:
    path = os.path.join(output_dir, f'gpt-oss-20b-{ds}-{lang}-retrieved.jsonl')
    if not os.path.exists(path):
        print(f'{ds}/{lang}: file not found'); continue

    records = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    total     = len(records)
    nil_total = sum(1 for r in records if r['gold_qid'] == 'NIL')
    non_nil   = total - nil_total

    # ── baseline (from file) ───────────────────────────────────────────────────
    correct_base    = sum(1 for r in records if r.get('is_correct'))
    nil_ok_base     = sum(1 for r in records if r['gold_qid'] == 'NIL' and r.get('is_correct'))
    non_nil_ok_base = sum(1 for r in records if r['gold_qid'] != 'NIL' and r.get('is_correct'))

    # ── alias second-chance: only incorrect non-NIL records ───────────────────
    alias_rescues = 0
    for r in records:
        if r['gold_qid'] == 'NIL' or r.get('is_correct'):
            continue  # skip NIL and already-correct
        mention = r.get('entity', r.get('mention', r.get('surface', '')))
        alias_qids = ALIAS_DICT.get(normalize(mention), [])
        if r['gold_qid'] in alias_qids:
            alias_rescues += 1

    correct_alias    = correct_base    + alias_rescues
    non_nil_ok_alias = non_nil_ok_base + alias_rescues

    print(f'── {ds}/{lang} ──────────────────────────')
    print(f'  Total / NIL / Non-NIL : {total} / {nil_total} / {non_nil}')
    print()
    print(f'  {"":20s} {"Baseline":>15}   {"+ Alias":>15}   {"Δ":>6}')
    print(f'  {"─"*60}')
    print(f'  {"Recall@10":<20} {correct_base:>6}/{total} ({100*correct_base/total:.1f}%)   '
          f'{correct_alias:>6}/{total} ({100*correct_alias/total:.1f}%)   '
          f'+{alias_rescues}')
    if nil_total:
        pct = 100*nil_ok_base/nil_total
        print(f'  {"NIL recall":<20} {nil_ok_base:>6}/{nil_total} ({pct:.1f}%)   '
              f'{"(unchanged)":>20}')
    if non_nil:
        print(f'  {"Non-NIL recall":<20} {non_nil_ok_base:>6}/{non_nil} ({100*non_nil_ok_base/non_nil:.1f}%)   '
              f'{non_nil_ok_alias:>6}/{non_nil} ({100*non_nil_ok_alias/non_nil:.1f}%)   '
              f'+{alias_rescues}')
    print(f'\n  Alias rescues (non-NIL): {alias_rescues}\n')