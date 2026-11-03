"""Loaders for the public datasets, normalised to `Prompt`.

Only prompts and labels are loaded. No model completions are ever stored from these sources.
"""
from __future__ import annotations

import ast
import json
from typing import Iterator

from ..languages import AYA_FILES, MULTIJAIL_COLUMNS
from .records import Prompt, make_id

# XSTest types: safe prompts have plain type names, unsafe look-alikes are prefixed "contrast_".
XSTEST_UNSAFE_PREFIX = "contrast_"


def _first_category(raw: str | list | None) -> str:
    if not raw:
        return ""
    if isinstance(raw, list):
        return raw[0] if raw else ""
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return str(raw)
    return val[0] if isinstance(val, list) and val else str(val)


def aya_redteaming_rows(lang: str, rows: Iterator[dict]) -> Iterator[Prompt]:
    """Rows from one aya_<lang>.jsonl file. Items are written natively per language, so groups are per language."""
    for i, r in enumerate(rows):
        group = f"aya-{lang}-{i:05d}"
        yield Prompt(
            id=make_id(group, lang), group=group, lang=lang, kind="harmful", text=r["prompt"].strip(),
            source="CohereLabs/aya_redteaming", source_id=f"{AYA_FILES[lang]}#{i}",
            category=_first_category(r.get("harm_category")), origin="native",
            meta={"global_or_local": r.get("global_or_local", "")},
        )


def multijail_rows(rows: Iterator[dict], langs: list[str]) -> Iterator[Prompt]:
    """MultiJail rows are parallel: one group per row, one Prompt per requested language."""
    for r in rows:
        group = f"multijail-{r['id']}"
        for lang in langs:
            text = (r.get(MULTIJAIL_COLUMNS[lang]) or "").strip()
            if not text:
                continue
            yield Prompt(
                id=make_id(group, lang), group=group, lang=lang, kind="harmful", text=text,
                source="DAMO-NLP-SG/MultiJail", source_id=str(r["id"]),
                category=_first_category(r.get("tags")), origin="native" if lang == "en" else "translated",
                meta={"upstream_source": r.get("source", "")},
            )


def xstest_rows(rows: Iterator[dict]) -> Iterator[Prompt]:
    """XSTest v2 prompts (English). Safe prompts become `benign`, contrast prompts become `harmful`."""
    for r in rows:
        ptype = r.get("type", "")
        kind = "harmful" if ptype.startswith(XSTEST_UNSAFE_PREFIX) else "benign"
        group = f"xstest-{r['id']}"
        yield Prompt(
            id=make_id(group, "en"), group=group, lang="en", kind=kind, text=r["prompt"].strip(),
            source="natolambert/xstest-v2-copy", source_id=str(r["id"]), category=ptype, origin="native",
        )


# ---- network loaders (thin wrappers so the normalisers above stay testable offline) ----------------------------

def load_aya_redteaming(lang: str, repo: str = "CohereLabs/aya_redteaming") -> Iterator[Prompt]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, AYA_FILES[lang], repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        yield from aya_redteaming_rows(lang, (json.loads(line) for line in f if line.strip()))


def load_multijail(langs: list[str], repo: str = "DAMO-NLP-SG/MultiJail") -> Iterator[Prompt]:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    yield from multijail_rows(iter(ds), [l for l in langs if l in MULTIJAIL_COLUMNS])


def load_xstest(repo: str = "natolambert/xstest-v2-copy") -> Iterator[Prompt]:
    from datasets import load_dataset

    ds = load_dataset(repo, split="prompts")
    yield from xstest_rows(iter(ds))
