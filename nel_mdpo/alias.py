import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize_alias(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_aliases_from_hipe_tsv(tsv_path: str | Path) -> dict[str, set[str]]:
    from hipe_commons.helpers.tsv import tsv_to_dict

    tsv = tsv_to_dict(str(tsv_path))
    tokens = tsv["TOKEN"]
    ne_labels = tsv["NE-COARSE-LIT"]
    nel_labels = tsv["NEL-LIT"]
    aliases: dict[str, set[str]] = defaultdict(set)

    i = 0
    while i < len(tokens):
        if ne_labels[i] in {"_", "O"}:
            i += 1
            continue

        label = ne_labels[i]
        mention_tokens = []
        qids = []
        while i < len(tokens) and ne_labels[i] == label:
            mention_tokens.append(tokens[i])
            qids.append(nel_labels[i])
            i += 1

        qid = next((q for q in qids if q not in {"_", "NIL"}), None)
        if qid:
            aliases[normalize_alias(" ".join(mention_tokens))].add(qid)

    return aliases


def build_alias_dictionary(
    hipe_root: str | Path,
    datasets: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, list[str]]:
    hipe_root = Path(hipe_root)
    datasets = datasets or ["newseye", "hipe2020", "hipe"]
    languages = languages or ["fr", "de", "sv", "fi", "nl", "en"]
    merged: dict[str, set[str]] = defaultdict(set)

    for dataset in datasets:
        for language in languages:
            path = hipe_root / dataset / language / f"HIPE-2022-v2.1-{dataset}-train-{language}.tsv"
            if not path.exists():
                continue
            for mention, qids in extract_aliases_from_hipe_tsv(path).items():
                merged[mention].update(qids)

    return {mention: sorted(qids) for mention, qids in merged.items()}


def load_or_build_alias_dictionary(
    alias_path: str | Path | None,
    hipe_root: str | Path | None = None,
) -> dict[str, list[str]]:
    if not alias_path:
        return {}
    alias_path = Path(alias_path)
    if alias_path.exists():
        return json.loads(alias_path.read_text(encoding="utf-8"))
    if not hipe_root:
        raise FileNotFoundError(f"{alias_path} does not exist and no HIPE root was provided")
    aliases = build_alias_dictionary(hipe_root)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
    return aliases


def promote_alias_qids(
    mention: str,
    retrievals: list[str],
    alias_dict: dict[str, list[str]],
    max_candidates: int,
) -> tuple[list[str], bool]:
    alias_qids = alias_dict.get(normalize_alias(mention), [])
    seen = set()
    merged = []

    for qid in alias_qids:
        if qid not in seen:
            seen.add(qid)
            merged.append(qid)

    for item in retrievals:
        qid = extract_qid(item)
        if qid and qid not in seen:
            seen.add(qid)
            merged.append(qid)

    return merged[:max_candidates], bool(alias_qids)


def extract_qid(text: str) -> str:
    match = re.search(r"\bQ\d+\b", str(text))
    return match.group(0) if match else ""
