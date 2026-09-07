# NEL Retrieval, Selection, and MDPO Fine-Tuning

This repository contains the main code for a two-phase Named Entity Linking
(NEL) pipeline for historical newspaper text:

1. **Retrieval**: generate normalized entity-name candidates with an LLM, then
   retrieve matching Wikidata QIDs.
2. **Selection**: choose the best Wikidata QID from the retrieved candidates.
3. **Fine-tuning**: build preference datasets and train the selector with
   multi-negative DPO / MDPO-style objectives.

The code was refactored from exploratory notebooks into reusable Python modules
and command-line entry points.

## Layout

```text
nel_mdpo/
  data.py                 HIPE TSV loading, entity extraction, TSV writing
  prompts.py              Shared retrieval and selection prompts
  wikidata.py             Async Wikidata search and entity lookup helpers
  retrieval.py            Phase 1 candidate retrieval pipeline
  selection.py            Phase 2 candidate selection pipeline
  dpo_data.py             Preference dataset builder
  trainers.py             Multi-negative DPO and MDPO/CADPO trainers
  train_mdpo.py           Fine-tuning CLI
scripts/
  retrieve_candidates.py  Run retrieval
  postprocess_retrieval.py Normalize/deduplicate retrievals and promote aliases
  select_entities.py      Run selection
  build_dpo_data.py       Build preference data from HIPE TSV
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

The retrieval and selection CLIs expect an OpenAI-compatible local model server
such as vLLM.

```bash
vllm serve openai/gpt-oss-120b --port 8007
```

## Phase 1: Retrieval

Input JSONL records should contain at least:

```json
{"input_candidate": "Paris", "context": "...", "gold_qid": "Q90", "abs_token_start": 10, "abs_token_end": 11}
```

Run:

```bash
python scripts/retrieve_candidates.py \
  --input-file data/hipe2020-test-fr.jsonl \
  --output-dir runs/retrieval \
  --dataset-name hipe2020 \
  --language fr \
  --model openai/gpt-oss-120b \
  --base-url http://0.0.0.0:8007/v1 \
  --split test \
  --max-candidates 8 \
  --alias-dict data/alias_dictionary_multilingual.json \
  --hipe-root ../HIPE-2022-data/data/v2.1
```

Output records include `retrievals`, a list of candidate QIDs.

Retrieval post-processing can also be run separately:

```bash
python scripts/postprocess_retrieval.py \
  --input-file runs/retrieval/gpt-oss-120b-hipe2020-fr-retrieved.jsonl \
  --output-file runs/retrieval/gpt-oss-120b-hipe2020-fr-256_alias.jsonl \
  --alias-dict data/alias_dictionary_multilingual.json \
  --hipe-root ../HIPE-2022-data/data/v2.1
```

## Phase 2: Selection

Run selection over the retrieval output and write token-level predictions:

```bash
python scripts/select_entities.py \
  --input-file runs/retrieval/gpt-oss-120b-hipe2020-fr-retrieved.jsonl \
  --output-dir runs/selection \
  --tsv-path ../HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --language fr \
  --model openai/gpt-oss-120b \
  --base-url http://0.0.0.0:8007/v1
```

## Build Preference Data

```bash
python scripts/build_dpo_data.py \
  --tsv-path ../HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
  --output-file data/dpo_hipe2020_fr_train.jsonl \
  --language fr \
  --model openai/gpt-oss-120b \
  --base-url http://0.0.0.0:8007/v1 \
  --chunk-size 256 \
  --max-negatives 3
```

## Legacy NewPaper Sources

The `legacy_newpaper/` folder keeps the original scripts used during the
experiments, including the segment-256 retrieval variants and alias workflow:

```text
legacy_newpaper/retrieval_simple_with_summary_context.py
legacy_newpaper/retrieval_with_alias.py
legacy_newpaper/add_alias_report.py
legacy_newpaper/alias_dictionary_multilingual.json
```

## GPT-OSS-120B Segment-256 Retrieval Files

`data/retrieval_gpt_oss_120b_256/` contains both retrieval versions for each
subset/language:

- original retrieval: `gpt-oss-120b-<dataset>-<lang>-256.jsonl`
- retrieval after adding aliases: `gpt-oss-120b-<dataset>-<lang>-256_alias.jsonl`

Included subsets/languages:

```text
hipe2020: de, en, fr
newseye: de, fi, fr, sv
```

## Train MDPO / Multi-Negative DPO

For standard multi-negative DPO records with `prompt`, `chosen`, and
`rejected` fields:

```bash
python -m nel_mdpo.train_mdpo \
  --train-files data/*train*.jsonl \
  --eval-files data/*dev*.jsonl \
  --model-name openai/gpt-oss-120b \
  --output-dir outputs/mdpo \
  --objective mdpo
```

Use `--objective multidpo` for the simpler chosen-vs-many-negatives objective.

## Data And Results

Training data used for `GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4` is hosted with
the model on Hugging Face:

```text
https://huggingface.co/Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4/tree/main/training_data/dpo_dataset_2003_chunk256
```

It corresponds to the `NewPaper/DPO_dataset_2003` chunk-256 DPO files, excluding
the extra `ajmc`, `letemps`, and `topres19th` subsets.

Prediction TSVs for that model/run are committed in:

```text
results/gpt_oss_20b_mdpo_3003_1600_chunk256/
```

Included outputs:

```text
DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-1600-256_HIPE2020_DE_256.tsv
DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-1600-256_HIPE2020_FR_256.tsv
DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-1600-256_NEWSEYE_DE_256.tsv
DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-1600-256_NEWSEYE_FR_256.tsv
```

The broader alias/base result set referenced in
`HIPE-2022-baseline/evaluate_notebook.ipynb` cell 87 is committed separately:

```text
results/gpt_oss_20b_mdpo_3003_2000_alias_base_gpt_oss_120b_1600_chunk256/
```

This set includes:

```text
HIPE2020_DE
HIPE2020_EN
HIPE2020_FR
NEWSEYE_DE
NEWSEYE_FI
NEWSEYE_FR
```

No `NEWSEYE_SV` TSV was found for that exact `GPT-OSS-120B-1600` filename
pattern.

The closest `NEWSEYE_SV` result referenced by `NewPaper/compare_tsvs.py` is
committed separately:

```text
results/gpt_oss_20b_mdpo_3003_2000_alias_base_gpt_oss_20b_chunk256/
```

Included file:

```text
DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-2000-ALIAS-BaseGPTOSS20B-256_NEWSEYE_SV_256.tsv
```

It was validated with `HIPE-scorer/clef_evaluation.py --task nel
--original_nel`. The scorer output is included as
`DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-2000-ALIAS-BaseGPTOSS20B-256_NEWSEYE_SV_256_nel.tsv`.
For this file, the scorer reports `NEL-LIT-micro-strict` F1 = `59.3` and
`NEL-LIT-macro_doc-strict` F1 = `63.6`.

Large datasets, generated predictions, checkpoints, and model weights are
ignored by `.gitignore` so this folder can be uploaded to GitHub cleanly.
