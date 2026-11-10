"""Offline checks for the router-judge sample: no network or billed call is made."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'router_sample.py'


def _module():
    loader = importlib.util.spec_from_file_location('router_sample', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def _rows():
    return [{'id': f'p{i}::{lang}', 'lang': lang, 'kind': kind, 'preamble': pre, 'response': f'answer {i}'}
            for lang in ('th', 'jv') for kind in ('harmful', 'benign') for pre in ('default', 'none') for i in range(30)]


def test_sample_fills_every_cell_and_is_repeatable():
    module = _module()
    key = lambda row: f"{row['id']}|{row['preamble']}"
    first = module.stratified_sample(_rows(), 5, 7, key)
    assert len(first) == 2 * 2 * 2 * 5
    cells = {(r['lang'], r['kind'], r['preamble']) for r in first}
    assert len(cells) == 8
    assert first == module.stratified_sample(list(reversed(_rows())), 5, 7, key)
    assert first != module.stratified_sample(_rows(), 5, 8, key)


def test_small_cells_give_what_they_have():
    module = _module()
    rows = [r for r in _rows() if not (r['lang'] == 'jv' and r['kind'] == 'harmful')][:-25]
    sample = module.stratified_sample(rows, 10, 1, lambda row: row['id'] + row['preamble'])
    assert sum(1 for r in sample if (r['lang'], r['kind'], r['preamble']) == ('jv', 'benign', 'none')) == 5


def test_usage_counts_billed_calls_but_not_cache_hits(monkeypatch, tmp_path):
    module = _module()
    from langsafe import llm_client

    class Cache:
        def __init__(self): self.store = {}
        def key(self, messages, params): return str(messages)
        def get(self, key): return self.store.get(key)
        def put(self, key, value): self.store[key] = value

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label": "full_refusal"}'))],
                               usage=SimpleNamespace(prompt_tokens=600, completion_tokens=12))

    client = object.__new__(llm_client.ChatClient)
    client.model, client.max_retries, client._cache = 'judge-32b', 1, Cache()
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_client.ChatClient, 'chat', llm_client.ChatClient.chat)
    module.record_usage()
    module.record_usage()  # once per tag in main(): must not count each call twice
    client.chat([{'role': 'user', 'content': 'a'}])
    client.chat([{'role': 'user', 'content': 'a'}])  # cached: not billed
    client.chat([{'role': 'user', 'content': 'b'}])
    assert dict(module.USAGE['judge-32b']) == {'calls': 2, 'prompt_tokens': 1200, 'completion_tokens': 24}


def test_content_agreement_skips_disputed_rows():
    module = _module()
    base = {'prompt': 'q', 'kind': 'harmful', 'preamble': 'default'}
    local = [dict(base, id='1', lang='th', response='r1', safety_label='safe'),
             dict(base, id='2', lang='th', response='r2', safety_label='unsafe'),
             dict(base, id='3', lang='th', response='r3', safety_label='disputed')]
    router = [dict(row, safety_label=label) for row, label in zip(local, ('safe', 'safe', 'safe'))]
    result = module.content_agreement(local, router)['by_lang']['th']
    assert result['shared'] == 3 and result['decided_by_both'] == 2 and result['agreement'] == 0.5
    assert result['local_safe'] == 1 and result['router_safe'] == 2
