"""Write the final question dataset to disk in JSON, JSONL, or CSV format."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def write_dataset(
    questions: list[dict],
    output_dir: str,
    output_format: str,
    topic: str,
) -> str:
    """Persist questions to disk and return the absolute output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    safe_topic = _slugify(topic)
    fmt = output_format.lower()

    if fmt == "json":
        return _write_json(questions, output_dir, safe_topic)
    if fmt == "jsonl":
        return _write_jsonl(questions, output_dir, safe_topic)
    if fmt == "csv":
        return _write_csv(questions, output_dir, safe_topic)

    raise ValueError(
        f"Unknown output format: {output_format!r}. Supported: 'json', 'jsonl', 'csv'."
    )


# ── Format writers ─────────────────────────────────────────────────────────────

def _write_json(questions: list[dict], output_dir: str, slug: str) -> str:
    path = os.path.join(output_dir, f"{slug}_dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
    return path


def _write_jsonl(questions: list[dict], output_dir: str, slug: str) -> str:
    path = os.path.join(output_dir, f"{slug}_dataset.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return path


def _write_csv(questions: list[dict], output_dir: str, slug: str) -> str:
    path = os.path.join(output_dir, f"{slug}_dataset.csv")
    if not questions:
        Path(path).touch()
        return path

    flat_rows = [_flatten_for_csv(q) for q in questions]
    fieldnames = list(flat_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return path


def _flatten_for_csv(q: dict) -> dict:
    """Expand the nested 'options' dict into flat option_a … option_d columns."""
    row = {k: v for k, v in q.items() if k != "options"}
    options = q.get("options") or {}
    for key in ("A", "B", "C", "D"):
        row[f"option_{key.lower()}"] = options.get(key, "")
    return row


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert a topic name to a safe filename component."""
    import re
    return re.sub(r"[^\w]+", "_", text.lower()).strip("_")[:60]
