"""OpenAI-compatible chat client (Hugging Face router by default) with an on-disk cache.

Every call is cached by a hash of (model, messages, params), so re-running an evaluation never re-pays for, or
silently changes, a judgement that was already made.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from .config import ROOT

CACHE_PATH = ROOT / "runs" / "llm_cache.sqlite"


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "HF_TOKEN", cache_path: Path = CACHE_PATH,
                 max_retries: int = 5):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Set {api_key_env} to call {model} at {base_url}")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.max_retries = max_retries
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def _key(self, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"m": self.model, "msgs": messages, "p": params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], **params) -> str:
        params = {"temperature": 0, "max_tokens": 512, **params}
        k = self._key(messages, params)
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            return row[0]
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
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, text))
            self._db.commit()
        return text
