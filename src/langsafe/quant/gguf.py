"""4-bit GGUF (Q4_K_M) verification through llama.cpp. NF4 runs in-process (gen/model.py --quant nf4).

    python llama.cpp/convert_hf_to_gguf.py runs/checkpoints/fire-circuit-... --outtype f16 --outfile x-f16.gguf
    llama.cpp/build/bin/llama-quantize x-f16.gguf x-q4_k_m.gguf Q4_K_M
    llama.cpp/build/bin/llama-server -m x-q4_k_m.gguf --port 8080 -c 4096

Prompts are rendered with the same Hugging Face chat template (and preamble condition) as the full-precision runs and
sent raw to /completion, so the only thing that changes is the weights' precision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import urllib.request

from ..config import ROOT
from ..data.records import Prompt
from ..gen.generate import render_prompt


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


class GGUFGenerator:
    """Same interface and cache format as gen.generate.Generator, backed by a running llama-server."""

    def __init__(self, tokenizer, decoding: dict, tag: str, gguf_path: str, server: str = "http://127.0.0.1:8080",
                 cache_dir: Path = ROOT / "runs" / "generations"):
        self.tok, self.decoding, self.tag, self.server = tokenizer, decoding, tag, server.rstrip("/")
        self.fp = hashlib.sha256(Path(gguf_path).read_bytes()[: 1 << 24]).hexdigest()[:16]   # first 16 MB
        self.cache_path = cache_dir / f"{tag}-{self.fp}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                r = json.loads(line)
                self._cache[r["key"]] = r

    def _key(self, p: Prompt, preamble: str) -> str:
        d = json.dumps({"fp": self.fp, "pre": preamble, "dec": self.decoding, "text": p.text}, sort_keys=True)
        return hashlib.sha256(d.encode()).hexdigest()[:24]

    def run(self, prompts: Iterable[Prompt], preamble: str) -> list[dict]:
        prompts = list(prompts)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for p in prompts:
                k = self._key(p, preamble)
                if k in self._cache:
                    continue
                res = _post(f"{self.server}/completion", {
                    "prompt": render_prompt(self.tok, p.text, preamble), "n_predict": self.decoding["max_new_tokens"],
                    "temperature": 0, "top_k": 1, "repeat_penalty": self.decoding.get("repetition_penalty", 1.0),
                    "cache_prompt": True})
                r = {"key": k, "id": p.id, "lang": p.lang, "kind": p.kind, "preamble": preamble,
                     "model_tag": self.tag, "fingerprint": self.fp, "response": res["content"].strip()}
                self._cache[k] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        return [self._cache[self._key(p, preamble)] for p in prompts]


def gguf_mmlu_accuracy(tokenizer, items: list[dict], server: str = "http://127.0.0.1:8080") -> dict:
    """Global-MMLU-Lite letter scoring via llama-server top-token probabilities."""
    from ..eval.capability import TEMPLATE

    correct = []
    for r in items:
        prompt = render_prompt(tokenizer, TEMPLATE.format(q=r["question"], a=r["option_a"], b=r["option_b"],
                                                          c=r["option_c"], d=r["option_d"]))
        res = _post(f"{server.rstrip('/')}/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 40,
                                                         "temperature": 0})
        probs = {}
        for cand in res.get("completion_probabilities", [{}])[0].get("top_probs", []) or \
                res.get("completion_probabilities", [{}])[0].get("probs", []):
            t = (cand.get("token") or cand.get("tok_str") or "").strip()
            if t in "ABCD" and t:
                probs[t] = max(probs.get(t, 0.0), cand.get("prob", 0.0))
        pick = max(probs, key=probs.get) if probs else ""
        correct.append(int(pick == r["answer"].strip().upper()))
    return {"accuracy": sum(correct) / len(correct), "n": len(correct)}
