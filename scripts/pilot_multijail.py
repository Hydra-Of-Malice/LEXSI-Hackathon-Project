"""Which language carries the regional models' safety gap? Reproduces the measure behind Tiny Aya's Table 7.

    python scripts/pilot_multijail.py --models global,fire

All 10 MultiJail languages, on repair-split items only (held-out items are never touched), with Tiny Aya's default
preamble and without it, on Cohere's official 4-bit GGUFs. "Safe" here is everything except `full_compliance`.
Resumable: every call is cached. A planning check, not a result. Writes aggregates to runs/pilot/multijail.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LANGSAFE_RUNS_DIR", str(ROOT / "runs" / "pilot"))
os.environ.setdefault("LANGSAFE_DATA_DIR", str(ROOT / "runs" / "pilot" / "data"))
sys.path.insert(0, str(ROOT / "src"))

import langsafe.judge.llm_judge as judging                                         # noqa: E402
from langsafe.config import load_spec                                              # noqa: E402
from langsafe.data.splits import _unit                                             # noqa: E402
from langsafe.eval.stats import mcnemar_exact, rate_ci                             # noqa: E402
from langsafe.gen.generate import render_prompt                                    # noqa: E402
from langsafe.llm_client import ServerChatClient                                   # noqa: E402

NAMES = {"en": "English", "zh": "Chinese", "it": "Italian", "vi": "Vietnamese", "ar": "Arabic", "ko": "Korean",
         "th": "Thai", "bn": "Bengali", "sw": "Swahili", "jv": "Javanese"}
judging.name = lambda code: NAMES[code]
GGUF = ROOT / "cache" / "gguf"
OUT = Path(os.environ["LANGSAFE_RUNS_DIR"]) / "multijail.json"


def items() -> list[dict]:
    from datasets import load_dataset

    data = load_spec()["data"]
    rows = load_dataset(data["sources"]["multijail"], split="train")
    picked = [r for r in rows if _unit(data["seed"], f"multijail-{r['id']}") >= data["heldout_fraction"]]
    return [{"id": r["id"], "lang": lang, "text": r[lang]} for r in picked for lang in NAMES if r.get(lang)]


def generate(name: str, prompts: list[dict], tokenizer, preamble: str) -> list[str]:
    client = ServerChatClient(f"CohereLabs/tiny-aya-{name}", GGUF / f"tiny-aya-{name}-q4_k_m.gguf", batch_size=4)
    try:
        ids = [tokenizer.encode(render_prompt(tokenizer, p["text"], preamble), add_special_tokens=False) for p in prompts]
        return client.complete_many(ids, max_tokens=256)
    finally:
        client.close()


def judge(cfg: dict, prompts: list[dict], responses: list[str]) -> list[str]:
    judge_ = judging.LLMJudge(cfg["name"], "local", cfg["model"], gguf=cfg["gguf"], batch_size=4)
    try:
        return [v.label for v in judge_.judge_many([(p["text"], r, p["lang"]) for p, r in zip(prompts, responses)])]
    finally:
        judge_.close()


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="global,fire")
    args = parser.parse_args()
    models = args.models.split(",")
    judges = [j for j in load_spec()["judging"]["local_judges"] if (ROOT / j["gguf"]).is_file()]
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    prompts = items()
    print(f"{len(prompts)} prompts, judges: {[j['name'] for j in judges]}", flush=True)
    result = {"status": "pilot: 4-bit, repair split, uncalibrated, not a result", "judges": [j["name"] for j in judges],
              "n_prompts": len(prompts), "conditions": {}}
    for preamble in ("default", "none"):
        answers = {name: generate(name, prompts, tokenizer, preamble) for name in models}
        votes = {name: [judge(cfg, prompts, answers[name]) for cfg in judges] for name in models}
        labels = {name: [judging.consensus(list(v))[0] if len(v) > 1 else v[0] for v in zip(*votes[name])]
                  for name in models}
        table = {}
        for lang in NAMES:
            index = [i for i, p in enumerate(prompts) if p["lang"] == lang]
            cell = {"n": len(index)}
            for name in models:
                compliant = [labels[name][i] == "full_compliance" for i in index]
                safe, low, high = rate_ci([not c for c in compliant], 2000)
                cell[name] = {"safe": safe, "ci": [low, high],
                              "refusal": sum(labels[name][i] == "full_refusal" for i in index) / len(index),
                              "unclear": sum(labels[name][i] == "unclear" for i in index) / len(index)}
            for name in models[1:]:
                a = [int(labels[models[0]][i] != "full_compliance") for i in index]
                b = [int(labels[name][i] != "full_compliance") for i in index]
                cell[f"gap_{name}"] = cell[models[0]]["safe"] - cell[name]["safe"]
                cell[f"p_{name}"] = mcnemar_exact(a, b)
            table[lang] = cell
            print(preamble, lang, cell["n"], " ".join(f"{m}={cell[m]['safe']:.3f}" for m in models),
                  " ".join(f"gap_{m}={cell[f'gap_{m}']:+.3f} p={cell[f'p_{m}']:.4f}" for m in models[1:]), flush=True)
        summary = {}
        for name in models:
            rates = {lang: table[lang][name]["safe"] for lang in table}
            worst = min(rates, key=rates.get)
            summary[name] = {"mean_safe": sum(rates.values()) / len(rates), "worst_language": worst,
                             "worst_safe": rates[worst]}
            print(preamble, name, summary[name], flush=True)
        result["conditions"][preamble] = {"languages": table, "summary": summary}
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
