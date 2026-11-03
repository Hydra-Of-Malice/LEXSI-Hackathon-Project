"""langsafe command line.

    langsafe build-data [--translate] [--no-curatorkit]
    langsafe generate   --model CohereLabs/tiny-aya-fire --tag fire-base --family fire [--quant nf4] [--adapter DIR] [--limit N]
    langsafe judge      --tag fire-base
    langsafe capability --model ... --tag fire-base [--limit N]
    langsafe report     --tags global-base,fire-base,fire-resta --family fire
    langsafe ledger-verify
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .config import ROOT, load_spec, spec_digest
from .provenance import Ledger

RUNS = ROOT / "runs"


def _heldout(family: str, limit: int | None):
    from .data.build import load_split

    langs = set(load_spec()["languages"][family])
    ps = [p for p in load_split("heldout") if p.lang in langs]
    if limit:                                   # smoke tests: first N per (lang, kind)
        seen: dict = {}
        out = []
        for p in ps:
            k = (p.lang, p.kind)
            if seen.get(k, 0) < limit:
                seen[k] = seen.get(k, 0) + 1
                out.append(p)
        ps = out
    return ps


def cmd_build_data(a):
    m = __import__("langsafe.data.build", fromlist=["build"]).build(translate=a.translate,
                                                                    use_curatorkit=not a.no_curatorkit)
    print(json.dumps({k: m[k] for k in ("files", "counts", "translation_failures")}, indent=2, ensure_ascii=False))


def cmd_generate(a):
    from .gen.generate import Generator
    from .gen.model import load

    spec = load_spec()
    dec = spec["decoding"]
    model, tok = load(a.model, dtype=dec["dtype"], quant=a.quant, adapter=a.adapter)
    gen = Generator(model, tok, dec, a.tag)
    prompts = _heldout(a.family, a.limit)
    out = []
    for pre in spec["preamble_conditions"]:
        out += gen.run(prompts, pre)
    path = RUNS / "gens" / f"{a.tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Ledger.default().append("generate", {"tag": a.tag, "model": a.model, "quant": a.quant, "adapter": a.adapter,
                                         "fingerprint": gen.fp, "n": len(out), "spec": spec_digest(spec)})
    print(f"{len(out)} generations -> {path.relative_to(ROOT)} (fingerprint {gen.fp})")


def cmd_judge(a):
    from .data.build import load_split
    from .eval.judge_run import judge_generations, write_hand_check_sheet

    spec = load_spec()
    prompts = {p.id: p for p in load_split("heldout")}
    gens = [json.loads(l) for l in open(RUNS / "gens" / f"{a.tag}.jsonl", encoding="utf-8")]
    res = judge_generations(gens, prompts, spec["judging"]["judges"], RUNS / "judged" / f"{a.tag}.jsonl")
    write_hand_check_sheet(res.pop("queue"), [], RUNS / "hand_check" / f"{a.tag}.csv")
    Ledger.default().append("judge", {"tag": a.tag, **res})
    print(json.dumps(res, indent=2))


def cmd_capability(a):
    from .eval.capability import accuracy, load_items
    from .gen.model import load

    spec = load_spec()
    model, tok = load(a.model, dtype=spec["decoding"]["dtype"], quant=a.quant, adapter=a.adapter)
    res = {}
    for lang in spec["capability_languages"]:
        r = accuracy(model, tok, load_items(lang, a.limit))
        res[lang] = {"accuracy": r["accuracy"], "n": r["n"]}
        print(lang, res[lang])
    path = RUNS / "capability" / f"{a.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    Ledger.default().append("capability", {"tag": a.tag, "model": a.model, "quant": a.quant, "result": res})


def cmd_report(a):
    from .eval.axes import safety_table
    from .eval.report import markdown_table

    langs = load_spec()["languages"][a.family]
    tables = {}
    for tag in a.tags.split(","):
        rows = [json.loads(l) for l in open(RUNS / "judged" / f"{tag}.jsonl", encoding="utf-8")]
        for pre in sorted({r["preamble"] for r in rows}):
            tables[f"{tag} [{pre}]"] = safety_table([r for r in rows if r["preamble"] == pre])
    print("## Harmful refusal (higher is safer)\n")
    print(markdown_table(tables, langs, "harmful_refusal"))
    print("## Benign over-refusal (lower is better)\n")
    print(markdown_table(tables, langs, "over_refusal"))


def cmd_ledger_verify(a):
    ok, bad = Ledger.default().verify()
    print("ledger OK" if ok else f"ledger BROKEN at entry {bad}")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="langsafe")
    sub = ap.add_subparsers(required=True)

    s = sub.add_parser("build-data"); s.set_defaults(fn=cmd_build_data)
    s.add_argument("--translate", action="store_true"); s.add_argument("--no-curatorkit", action="store_true")

    for name, fn in (("generate", cmd_generate), ("capability", cmd_capability)):
        s = sub.add_parser(name); s.set_defaults(fn=fn)
        s.add_argument("--model", required=True); s.add_argument("--tag", required=True)
        s.add_argument("--quant", choices=["nf4"]); s.add_argument("--adapter"); s.add_argument("--limit", type=int)
        if name == "generate":
            s.add_argument("--family", choices=["fire", "earth"], required=True)

    s = sub.add_parser("judge"); s.set_defaults(fn=cmd_judge); s.add_argument("--tag", required=True)
    s = sub.add_parser("report"); s.set_defaults(fn=cmd_report)
    s.add_argument("--tags", required=True); s.add_argument("--family", choices=["fire", "earth"], required=True)
    s = sub.add_parser("ledger-verify"); s.set_defaults(fn=cmd_ledger_verify)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
