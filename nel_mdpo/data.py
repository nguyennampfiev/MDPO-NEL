import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: str | Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_hipe_tsv(tsv_path: str | Path, chunk_size: int = 128) -> list[dict]:
    from hipe_commons.helpers.tsv import tsv_to_dict

    tsv = tsv_to_dict(str(tsv_path))
    tokens = tsv["TOKEN"]
    ne_labels = tsv["NE-COARSE-LIT"]
    nel_labels = tsv["NEL-LIT"]

    chunks = []
    total = len(tokens)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        while end < total and nel_labels[end] != "_":
            end += 1
        chunks.append(
            {
                "tokens": tokens[start:end],
                "labels": ne_labels[start:end],
                "el": nel_labels[start:end],
                "abs_start": start,
            }
        )
    return chunks


def extract_entities(tokens: list[str], labels: list[str], blank_label: str = "_") -> list[dict]:
    entities = []
    i = 0
    while i < len(tokens):
        if labels[i] == blank_label:
            i += 1
            continue

        start = i
        label = labels[i]
        mention_tokens = []
        while i < len(tokens) and labels[i] == label:
            mention_tokens.append(tokens[i])
            i += 1
        entities.append(
            {
                "entity": " ".join(mention_tokens),
                "label": label,
                "start": start,
                "end": i,
            }
        )
    return entities


def write_predictions_to_hipe_tsv(
    tsv_path: str | Path,
    output_path: str | Path,
    predictions: list[str],
    labels_column: str = "NEL-LIT",
) -> None:
    from hipe_commons.helpers.tsv import get_tsv_data

    lines = [line.split("\t") for line in get_tsv_data(str(tsv_path), "").split("\n")]
    label_col = lines[0].index(labels_column)

    for i, pred in enumerate(predictions, start=1):
        if i >= len(lines):
            break
        lines[i][label_col] = pred if pred and pred.upper() != "NIL" else "_"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join("\t".join(line) for line in lines))
