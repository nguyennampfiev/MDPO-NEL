# Reproducibility Notes

This document records the environment, data, model, and scoring setup used for
the refactored NEL retrieval, selection, and MDPO artifacts in this repository.

## What Can Be Reproduced

The repository supports three levels of reproduction:

1. Reuse the released artifacts: download the model/data from Hugging Face and
   inspect the committed retrieval/results files.
2. Re-run selection/evaluation against the committed retrieval/result artifacts.
3. Re-run the full pipeline, including retrieval, DPO data construction,
   training, MXFP4 export, and HIPE scoring.

The full pipeline requires external datasets, GPUs, Wikidata access, and local
OpenAI-compatible model servers.

## Environments

Two local environments were used.

### Training, Export, And Evaluation

Use an environment equivalent to local `IE-2025`.

Pinned package versions observed in that environment are in:

```text
requirements.txt
```

Important packages:

```text
python 3.10
torch 2.7.1+cu126
transformers 4.55.2
trl 0.15.2
peft 0.15.2
bitsandbytes 0.45.5
unsloth 2025.11.2
openai 1.108.1
datasets 3.5.1
```

### vLLM Serving

Use an environment equivalent to local `GPT-OSS`.

Pinned package versions observed in that environment are in:

```text
requirements-serve.txt
```

Important packages:

```text
python 3.12
vllm 0.19.0
torch 2.10.0
transformers 5.5.0
openai 2.8.1
```

Serve the released checkpoint-2000 model with:

```bash
/Utilisateurs/tnguye28/.conda/envs/GPT-OSS/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4 \
  --served-model-name GPT-OSS-20B-MDPO-NEL-3003-2000-mxfp4 \
  --host 0.0.0.0 \
  --port 8007
```

The Hugging Face repository name contains `1600`, but the root weights were
replaced with checkpoint `2000`.

## Hugging Face Artifacts

Model weights:

```text
https://huggingface.co/Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4
```

Training data:

```text
https://huggingface.co/Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4/tree/main/training_data/dpo_dataset_2003_chunk256
```

The training data folder contains chunk-256 DPO JSONL train/dev files for:

```text
hipe2020: de, fr
newseye: de, fi, fr, sv
```

It excludes the extra `ajmc`, `letemps`, and `topres19th` subsets.

## External Data

The evaluation commands expect HIPE-2022 data in the original local layout:

```text
../HIPE-2022-data/data/v2.1/<dataset>/<lang>/HIPE-2022-v2.1-<dataset>-test-<lang>.tsv
```

For example:

```text
../HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv
../HIPE-2022-data/data/v2.1/newseye/sv/HIPE-2022-v2.1-newseye-test-sv.tsv
```

The official HIPE scorer was used from:

```text
../HIPE-scorer/clef_evaluation.py
```

The local exploratory code also uses `hipe_commons.helpers.tsv`, which comes
from the local HIPE tooling under the notebook workspace. It is not packaged in
this repository.

## Retrieval Artifacts

GPT-OSS-120B segment-256 retrieval files are committed in:

```text
data/retrieval_gpt_oss_120b_256/
```

They include original and alias-augmented retrieval files:

```text
gpt-oss-120b-<dataset>-<lang>-256.jsonl
gpt-oss-120b-<dataset>-<lang>-256_alias.jsonl
```

Languages:

```text
hipe2020: de, en, fr
newseye: de, fi, fr, sv
```

## Training

The refactored training CLI trains LoRA/PEFT checkpoints. It does not directly
write the final MXFP4 merged model folder.

Example training command shape:

```bash
python -m nel_mdpo.train_mdpo \
  --train-files data/dpo_*_train_chunk256.jsonl \
  --eval-files data/dpo_*_dev_chunk256.jsonl \
  --model-name unsloth/gpt-oss-20b \
  --output-dir dpo_multineg_3003_randomly_shuffle \
  --objective multidpo \
  --max-seq-length 2048 \
  --max-prompt-length 1024 \
  --max-answer-length 128 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6
```

The released checkpoint-2000 MXFP4 folder was exported from:

```text
dpo_multineg_3003_randomly_shuffle/checkpoint-2000
```

## MXFP4 Export

The MXFP4 export is a separate Unsloth merge step:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="dpo_multineg_3003_randomly_shuffle/checkpoint-2000",
    dtype=None,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

model.save_pretrained_merged(
    "dpo_multineg_3003_randomly_shuffle_2000_mxfp4",
    tokenizer,
    save_method="mxfp4",
)
tokenizer.save_pretrained("dpo_multineg_3003_randomly_shuffle_2000_mxfp4")
```

For GPT-OSS 20B, copy the base/pretrained `config.json` from the
`unsloth/gpt-oss-20b` snapshot into the exported MXFP4 folder before serving.

## Paper Score Artifacts

Validated paper-score artifacts are committed in:

```text
results/gpt_oss_20b_mdpo_3003_2000_alias_chunk256_paper_scores/
```

The reported metric is:

```text
NEL-LIT-micro-strict-TIME-ALL-LED-ALL-@1 F1
```

from:

```bash
python ../HIPE-scorer/clef_evaluation.py \
  --ref <HIPE_TEST_TSV> \
  --pred <PREDICTION_TSV> \
  --task nel \
  --outdir . \
  --original_nel \
  --hipe_edition hipe-2020 \
  --log hipe-2022.log \
  --n_best 1 \
  --skip-check
```

Expected micro-strict F1 values from the committed scorer `_nel.tsv` files:

```text
hipe2020 FR  72.4
hipe2020 DE  65.6
hipe2020 EN  73.8
newseye  FR  68.7
newseye  DE  59.2
newseye  FI  62.7
newseye  SV  65.0
```

The files show `NEWSEYE_FI = 62.7` and `NEWSEYE_SV = 65.0`. If a paper table
lists `FI = 65.0` and `SV = 62.7`, those two columns are swapped relative to
the scorer outputs.

## Known Limits

- The repository includes refactored code and released artifacts, but the
  exploratory notebooks remain the source of some exact historical runs.
- Some seven-language paper-score values are represented by scorer output TSVs
  only; the matching prediction TSVs were found for the four exact
  `DPO-Sample...2000-ALIAS-256` runs, while EN/FI/SV are represented by
  `DPO-MultiNegative...ALIAS-GPT-OSS-120B` scorer outputs.
- Wikidata search results can drift over time, so exact retrieval reproduction
  may require using the committed retrieval JSONL files rather than live
  retrieval.
- Large model weights are hosted on Hugging Face, not committed to GitHub.
