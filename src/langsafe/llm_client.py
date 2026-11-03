"""Chat clients for judging and translation, with an on-disk cache.

Two backends share one interface (`chat`, `chat_many`, `close`):
- `ChatClient`: any OpenAI-compatible endpoint (the Hugging Face router by default; billed to inference credits).
- `LocalChatClient`: the same model family run on the local GPU in NF4 (Kaggle T4 or a laptop; no credits).

Every call is cached by a hash of (model, messages, params), so re-running an evaluation never re-pays for, or
silently changes, a judgement that was already made. `make_client(base_url="local", ...)` picks the local backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT

CACHE_PATH = ROOT / "runs" / "llm_cache.sqlite"


class _Cache:
    def __init__(self, path: Path, model_key: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.model_key = model_key

    def key(self, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"m": self.model_key, "msgs": messages, "p": params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, v))
            self._db.commit()


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "HF_TOKEN", cache_path: Path = CACHE_PATH,
                 max_retries: int = 5, workers: int = 8):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Set {api_key_env} to call {model} at {base_url}")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.max_retries = max_retries
        self.workers = workers
        self._cache = _Cache(cache_path, model)

    def chat(self, messages: list[dict], **params) -> str:
        params = {"temperature": 0, "max_tokens": 512, **params}
        k = self._cache.key(messages, params)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, **params)
                text = resp.choices[0].message.content or ""
                break
            except Exception:  # rate limits and transient provider errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        self._cache.put(k, text)
        return text

    def chat_many(self, batch: list[list[dict]], **params) -> list[str]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.chat(m, **params), batch))

    def close(self) -> None:
        pass


class LocalChatClient:
    """Greedy, batched generation on the local GPU. The model loads on the first cache miss, so a fully cached
    re-run needs no GPU. Cache keys carry a `local:` prefix and the quantisation, so they never collide with the
    hosted model of the same name."""

    def __init__(self, model: str, quant: str | None = "auto", batch_size: int = 8, cache_path: Path = CACHE_PATH):
        if quant == "auto":            # NF4 on CUDA; fp16 elsewhere (an 8B judge in fp16 is ~16 GB, fine on a 64 GB Mac)
            from .gen.model import best_device

            quant = "nf4" if best_device() == "cuda" else None
        self.model = model
        self.quant = quant
        self.batch_size = batch_size
        self._cache = _Cache(cache_path, f"local:{model}:{quant}")
        self._m = self._tok = None

    def _load(self):
        if self._m is None:
            from .gen.model import load

            self._m, self._tok = load(self.model, dtype="float16", quant=self.quant)
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token

    def _render(self, messages: list[dict]) -> str:
        # enable_thinking is read by Qwen3's template and ignored by others.
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def chat(self, messages: list[dict], **params) -> str:
        return self.chat_many([messages], **params)[0]

    def chat_many(self, batch: list[list[dict]], **params) -> list[str]:
        params = {"temperature": 0, "max_tokens": 512, **params}
        keys = [self._cache.key(m, params) for m in batch]
        out = [self._cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if not todo:
            return out
        import torch
        from tqdm import tqdm

        self._load()
        # Longest first keeps padding low within a batch.
        todo.sort(key=lambda i: -sum(len(x["content"]) for x in batch[i]))
        for s in tqdm(range(0, len(todo), self.batch_size), desc=f"local {self.model}", leave=False):
            idx = todo[s:s + self.batch_size]
            enc = self._tok([self._render(batch[i]) for i in idx], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(self._m.device)
            with torch.no_grad():
                gen = self._m.generate(**enc, max_new_tokens=params["max_tokens"], do_sample=False,
                                       pad_token_id=self._tok.pad_token_id)
            texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t
                self._cache.put(keys[i], t)
        return out

    def close(self) -> None:
        if self._m is not None:
            import gc

            import torch

            self._m = self._tok = None
            gc.collect()
            torch.cuda.empty_cache()


def make_client(base_url: str, model: str, **kwargs) -> ChatClient | LocalChatClient:
    if base_url == "local":
        return LocalChatClient(model, **kwargs)
    return ChatClient(base_url, model, **kwargs)
