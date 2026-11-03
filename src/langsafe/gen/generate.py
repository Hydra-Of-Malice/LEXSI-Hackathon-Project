"""Batched, deterministic generation with Tiny Aya, with the default system preamble on or off.

Tiny Aya's chat template always injects a ~366-token system preamble. Real users often replace or strip it, so we
measure both conditions. `none` removes the template's system turn and keeps everything else byte-identical.
Generations are cached per (model fingerprint, preamble, decoding, prompt).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from ..config import ROOT
from ..data.records import Prompt

SYSTEM_TURN = re.compile(r"<\|START_OF_TURN_TOKEN\|><\|SYSTEM_TOKEN\|>.*?<\|END_OF_TURN_TOKEN\|>", re.S)


def render_prompt(tokenizer, user_text: str, preamble: str = "default") -> str:
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True)
    if preamble == "default":
        return text
    if preamble == "none":
        stripped, n = SYSTEM_TURN.subn("", text, count=1)
        if n != 1:
            raise ValueError("could not find the system turn in the rendered chat template; check the template")
        return stripped
    raise ValueError(f"unknown preamble condition {preamble!r}")


def model_fingerprint(model) -> str:
    """Cheap fingerprint of the weights actually loaded (catches 'evaluated the wrong checkpoint')."""
    import torch

    h = hashlib.sha256()
    for i, (n, p) in enumerate(model.named_parameters()):
        if i % 16 == 0:                                   # sample every 16th tensor to keep it fast
            flat = p.detach().float().flatten()
            h.update(n.encode())
            h.update(flat[:: max(1, flat.numel() // 4096)].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class Generator:
    def __init__(self, model, tokenizer, decoding: dict, tag: str, cache_dir: Path = ROOT / "runs" / "generations"):
        self.model, self.tok, self.decoding, self.tag = model, tokenizer, decoding, tag
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.fp = model_fingerprint(model)
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
        import torch

        prompts = list(prompts)
        todo = [p for p in prompts if self._key(p, preamble) not in self._cache]
        bs = self.decoding.get("batch_size", 16)
        gen_kwargs = {k: self.decoding[k] for k in ("do_sample", "max_new_tokens", "repetition_penalty")
                      if k in self.decoding}
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), bs):
                batch = todo[i:i + bs]
                texts = [render_prompt(self.tok, p.text, preamble) for p in batch]
                enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(**enc, **gen_kwargs, pad_token_id=self.tok.pad_token_id)
                for p, seq in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                    r = {"key": self._key(p, preamble), "id": p.id, "lang": p.lang, "kind": p.kind,
                         "preamble": preamble, "model_tag": self.tag, "fingerprint": self.fp,
                         "response": self.tok.decode(seq, skip_special_tokens=True).strip()}
                    self._cache[r["key"]] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
        return [self._cache[self._key(p, preamble)] for p in prompts]
