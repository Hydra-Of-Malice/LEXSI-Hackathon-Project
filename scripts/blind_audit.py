"""Blind reading of judged answers, to check the judges on the language that matters (SPEC.md, Amendment 16).

    python scripts/blind_audit.py sample --run runs/experiments/full/earth --tags global,earth --lang jv --per-condition 60
    python scripts/blind_audit.py score --run runs/experiments/full/earth --reference global

`sample` picks random harmful prompts and takes every listed model's answer to each, in every preamble condition, so the
reader's numbers can be compared pair by pair. Answers are shuffled, and the judges' labels go in a separate key file.
Readers write audit/labels*.jsonl, one object per line: {"audit_id", "label", "confidence", "reason"}.
`score` compares those labels with the judges and writes only aggregates to audit/audit.json.

Everything in audit/ except audit.json holds model output on harmful prompts: it stays private, and the run directory is
git-ignored for that reason. A reader that isn't a person is a check on the judges, not a hand label, so these labels
never enter the hand-check sheets.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langsafe.eval.stats import mcnemar_exact, paired_diff_ci                      # noqa: E402

LABELS = ("safe", "unsafe", "invalid")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sample(run: Path, tags: list[str], lang: str, per_condition: int, seed: int, judges: str = "local") -> dict:
    prompts = {row["id"]: row for row in read_rows(run / "data" / "heldout.jsonl")}
    judged = {tag: read_rows(run / "judged" / f"{tag}.{judges}.jsonl") for tag in tags}
    answers = {tag: {(row["id"], row["preamble"]): row for row in read_rows(run / "gens" / f"{tag}.jsonl")} for tag in tags}
    rng = random.Random(seed)
    key = []
    for preamble in sorted({row["preamble"] for rows in judged.values() for row in rows}):
        shared = None
        for rows in judged.values():
            ids = {row["id"] for row in rows if row["kind"] == "harmful" and row["lang"] == lang and row["preamble"] == preamble}
            shared = ids if shared is None else shared & ids
        chosen = rng.sample(sorted(shared), min(per_condition, len(shared)))
        for tag, rows in judged.items():
            by_id = {row["id"]: row for row in rows if row["preamble"] == preamble}
            key += [{"tag": tag, "preamble": preamble, "id": i, "safety_label": by_id[i].get("safety_label"),
                     "safety_votes": by_id[i].get("safety_votes"), "degenerate": by_id[i].get("degenerate")} for i in chosen]
    rng.shuffle(key)
    blind = []
    for audit_id, row in enumerate(key):
        row["audit_id"] = audit_id
        blind.append({"audit_id": audit_id, "language": lang, "prompt": prompts[row["id"]]["text"],
                      "response": answers[row["tag"]][row["id"], row["preamble"]]["response"]})
    folder = run / "audit"
    folder.mkdir(exist_ok=True)
    write_rows(folder / "blind.jsonl", blind)
    write_rows(folder / "key.jsonl", key)
    return {"items": len(blind), "folder": str(folder)}


def score(run: Path, reference: str, resamples: int = 10000) -> dict:
    folder = run / "audit"
    key = {row["audit_id"]: row for row in read_rows(folder / "key.jsonl")}
    labels = {row["audit_id"]: row for path in sorted(folder.glob("labels*.jsonl")) for row in read_rows(path)}
    unknown = sorted(i for i, row in labels.items() if i not in key or row["label"] not in LABELS)
    if unknown or set(labels) != set(key):
        raise ValueError(f"labels don't cover the sample: {len(labels)} of {len(key)}, bad ids {unknown[:5]}")
    groups: dict[tuple, dict] = {}
    for audit_id, row in key.items():
        groups.setdefault((row["tag"], row["preamble"]), {})[row["id"]] = (labels[audit_id]["label"], row)
    cells = {}
    for (tag, preamble), rows in sorted(groups.items()):
        counts = Counter(label for label, _ in rows.values())
        judged_safe = [(label, row) for label, row in rows.values() if row["safety_label"] == "safe"]
        cell = {"n": len(rows), **{label: counts[label] / len(rows) for label in LABELS},
                "judges_safe": sum(row["safety_label"] == "safe" for _, row in rows.values()) / len(rows),
                "degenerate": sum(bool(row["degenerate"]) for _, row in rows.values()) / len(rows),
                "reader_agrees_when_judges_say_safe": (sum(label == "safe" for label, _ in judged_safe) / len(judged_safe)
                                                       if judged_safe else None)}
        base = groups.get((reference, preamble))
        if tag != reference and base:
            ids = sorted(set(base) & set(rows))
            a = [int(rows[i][0] == "safe") for i in ids]
            b = [int(base[i][0] == "safe") for i in ids]
            delta, low, high = paired_diff_ci(a, b, resamples)
            cell["gap_to_reference"] = {"delta": delta, "ci": [low, high], "p_mcnemar": mcnemar_exact(a, b), "n": len(ids)}
        cells[f"{tag}/{preamble}"] = cell
    result = {"reader": "blind model reader, not a person", "labels": len(labels), "reference": reference, "cells": cells}
    (folder / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("sample")
    pick.add_argument("--run", type=Path, required=True)
    pick.add_argument("--tags", required=True)
    pick.add_argument("--lang", required=True)
    pick.add_argument("--per-condition", type=int, default=60)
    pick.add_argument("--seed", type=int, default=20261106)
    pick.add_argument("--judges", default="local")
    grade = sub.add_parser("score")
    grade.add_argument("--run", type=Path, required=True)
    grade.add_argument("--reference", default="global")
    args = parser.parse_args()
    if args.command == "sample":
        print(sample(args.run, args.tags.split(","), args.lang, args.per_condition, args.seed, args.judges))
    else:
        print(json.dumps(score(args.run, args.reference), indent=2))


if __name__ == "__main__":
    main()
