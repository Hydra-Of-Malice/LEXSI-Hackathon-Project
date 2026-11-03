"""Build the evaluation (held-out) and repair sets from the public sources.

    langsafe build-data                 # native sources only, no API key needed
    langsafe build-data --translate     # also translate into ta / hi-Latn / hi and benign look-alikes (needs HF_TOKEN)
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, load_spec, spec_digest
from ..provenance import Ledger
from .records import Prompt, read_prompts, write_jsonl
from .sources import load_aya_redteaming, load_multijail, load_xstest
from .splits import assign_splits, drop_leaks, leakage_report

OUT = ROOT / "data" / "processed"


def _cap(prompts: list[Prompt], per_lang: int, seed: int) -> list[Prompt]:
    """Deterministic per-language cap, sampled by group so parallel items stay together where possible.

    Uses a different salt from the split hash: sharing it would put every capped item in the same split.
    """
    by_lang = defaultdict(list)
    for p in prompts:
        by_lang[p.lang].append(p)
    out = []
    for lang, ps in sorted(by_lang.items()):
        ps.sort(key=lambda p: hashlib.sha256(f"cap:{seed}:{p.group}".encode()).hexdigest())
        out.extend(ps[:per_lang])
    return out


def build(translate: bool = False, use_curatorkit: bool = True) -> dict:
    spec = load_spec()
    d = spec["data"]
    langs = sorted(set(spec["languages"]["fire"]) | set(spec["languages"]["earth"]))
    seed = d["seed"]

    harmful: list[Prompt] = []
    for lang in ("en", "hi", "ar"):
        harmful += list(load_aya_redteaming(lang))
    mj = list(load_multijail(["en", "bn", "sw", "ar"]))
    harmful += mj
    xs = list(load_xstest())
    harmful += [p for p in xs if p.kind == "harmful"]
    benign = [p for p in xs if p.kind == "benign"]

    failed: list[str] = []
    if translate:
        from ..llm_client import ChatClient
        from .translate import translate_all

        j = spec["judging"]["judges"][0]          # Aya Expanse 32B translates; it is a Cohere model by design
        client = ChatClient(j["base_url"], j["model"])
        mj_en = [p for p in mj if p.lang == "en"]
        missing = [l for l in langs if l not in {"en", "bn", "sw", "ar"}]          # hi, ta, hi-Latn
        new, f1 = translate_all(client, mj_en, missing)
        harmful += new
        new_b, f2 = translate_all(client, benign, [l for l in langs if l != "en"])
        benign += new_b
        failed = f1 + f2

    pool = [p for p in harmful + benign if p.lang in langs]
    curation = None
    if use_curatorkit:
        from .curate import curate

        pool, curation = curate(pool)
    harmful = _cap([p for p in pool if p.kind == "harmful"], d["max_harmful_per_lang"], seed)
    benign = _cap([p for p in pool if p.kind == "benign"], d["max_benign_per_lang"], seed)
    allp = assign_splits(harmful + benign, d["heldout_fraction"], seed)
    report = leakage_report(allp)
    if not report["ok"]:
        allp = drop_leaks(allp, report)
        report = leakage_report(allp)
    assert report["ok"], f"split leakage remains: {report}"

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("heldout", "repair"):
        rows = [p for p in allp if p.split == split]
        path = OUT / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {"path": str(path.relative_to(ROOT)), "rows": len(rows),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    counts = Counter((p.split, p.kind, p.lang) for p in allp)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(), "spec_digest": spec_digest(spec), "seed": seed,
        "translated": translate, "translation_failures": failed, "curatorkit": curation, "files": files,
        "counts": {f"{s}/{k}/{l}": n for (s, k, l), n in sorted(counts.items())},
        "leakage_check": {"ok": True, "near_dup_jaccard": 0.8},
        "sources": d["sources"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Ledger.default().append("build-data", {"manifest": manifest})
    return manifest


def load_split(split: str) -> list[Prompt]:
    return list(read_prompts(OUT / f"{split}.jsonl"))
