"""Judge a set of generations with both LLM judges plus the rule detector, and queue disagreements for hand-checking."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from ..config import ROOT
from ..data.records import Prompt
from ..judge.agreement import cohen_kappa
from ..judge.llm_judge import LLMJudge, consensus
from ..judge.rules import refusal_signal


def judge_generations(gens: list[dict], prompts: dict[str, Prompt], judges_cfg: list[dict], out_path: Path,
                      workers: int = 8) -> dict:
    judges = [LLMJudge(j["name"], j["base_url"], j["model"]) for j in judges_cfg]
    items = [(prompts[g["id"]].text, g["response"], g["lang"]) for g in gens]
    verdicts = {j.name: j.judge_many(items, workers) for j in judges}

    rows, queue = [], []
    for k, g in enumerate(gens):
        labels = [verdicts[j.name][k].label for j in judges]
        final, flag = consensus(labels)
        row = {**{x: g[x] for x in ("id", "lang", "kind", "preamble", "model_tag", "fingerprint")},
               "label": final, "judges": dict(zip([j.name for j in judges], labels)),
               "rule_refusal": refusal_signal(g["response"], g["lang"]), "needs_hand_check": flag}
        rows.append(row)
        if flag:
            queue.append({**row, "prompt": prompts[g["id"]].text, "response": g["response"]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kappa = {}
    if len(judges) == 2:
        for lang in sorted({r["lang"] for r in rows}):
            a = [r["judges"][judges[0].name] for r in rows if r["lang"] == lang]
            b = [r["judges"][judges[1].name] for r in rows if r["lang"] == lang]
            kappa[lang] = cohen_kappa(a, b)
    return {"rows": len(rows), "hand_check_queue": len(queue), "kappa_by_lang": kappa, "queue": queue}


def write_hand_check_sheet(queue: list[dict], sample: list[dict], path: Path) -> None:
    """CSV for a human: every disagreement plus a random audit sample. Contains model outputs: keep it private."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "lang", "kind", "model_tag", "preamble", "judges", "prompt", "response", "hand_label"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in queue + sample:
            w.writerow({**r, "judges": json.dumps(r.get("judges", {})), "hand_label": ""})


def audit_sample(rows: list[dict], gens_by_key: dict, prompts: dict[str, Prompt], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pick = rng.sample(rows, min(n, len(rows)))
    return [{**r, "prompt": prompts[r["id"]].text, "response": gens_by_key[(r["id"], r["preamble"])]["response"]}
            for r in pick]
