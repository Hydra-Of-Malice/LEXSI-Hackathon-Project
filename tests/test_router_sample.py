"""Offline checks for the router-judge sample: no network or billed call is made."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_usage_counts_each_billed_call_once_even_from_many_threads(monkeypatch, tmp_path):
    """The judge sends requests from 8 threads; the first full sample's counter stacked wrappers and counted calls
    about five times over. Every billed call must count exactly once, and cache hits not at all."""
    import time

    module = _module()
    from langsafe import llm_client
    import openai

    billed = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def create(**request):
                time.sleep(0.002)  # let threads overlap
                billed.append(request['messages'][0]['content'])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label": "safe"}'))],
                                       usage=SimpleNamespace(prompt_tokens=600, completion_tokens=12))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)
    monkeypatch.setenv('HF_TOKEN', 'not-a-real-token')
    monkeypatch.setattr(llm_client.ChatClient, '__init__', llm_client.ChatClient.__init__)
    module.record_usage()
    module.record_usage()  # main() calls it once per tag; it must not wrap twice
    client = llm_client.ChatClient('https://router.example/v1', 'judge-32b', cache_path=tmp_path / 'cache.sqlite')
    batch = [[{'role': 'user', 'content': f'item {i % 40}'}] for i in range(120)]  # 40 distinct, 80 repeats
    client.chat_many(batch)
    client.chat_many(batch)  # all cached now
    client.close()
    assert dict(module.USAGE['judge-32b']) == {'calls': len(billed), 'prompt_tokens': 600 * len(billed),
                                               'completion_tokens': 12 * len(billed)}
    assert 40 <= len(billed) < 120


def _judged(judge_set, n, label=lambda i: 'full_refusal' if i % 3 else 'full_compliance'):
    return [{'id': f'p{i}', 'lang': 'th' if i % 2 else 'jv', 'kind': 'harmful', 'preamble': 'default',
             'prompt': f'prompt {i}', 'response': f'answer {i}', 'judge_set': judge_set,
             'judges': {'a': {}, 'b': {}}, 'rubric_hash': 'rubric', 'label': label(i),
             'safety_label': 'safe' if i % 4 else 'unsafe'} for i in range(n)]


def test_calibration_file_covers_the_whole_local_evaluation(tmp_path):
    import json as _json
    from langsafe.eval.judge_run import evaluation_digest
    from langsafe.provenance import Ledger

    module = _module()
    run, out = tmp_path / 'earth', tmp_path / 'earth-router-sample'
    local = _judged('local', 40)
    (run / 'judged').mkdir(parents=True)
    (out / 'judged').mkdir(parents=True)
    (run / 'judged' / 'earth.local.jsonl').write_text(''.join(_json.dumps(r) + '\n' for r in local))
    summary = run / 'judged' / 'earth.local.summary.json'
    summary.write_text(_json.dumps({'evaluation_digest': evaluation_digest(local)}))
    router = _judged('router', 40)[::4]  # the sample: every fourth answer
    (out / 'judged' / 'earth.router-sample.jsonl').write_text(''.join(_json.dumps(r) + '\n' for r in router))
    path = Path(module.write_calibration(run, out, 'earth', 20, replace=False))
    result = _json.loads(path.read_text())
    assert path == run / 'calibration' / 'earth.router-comparison.json'
    assert result['verified'] and result['shared_rows'] == 10 and result['sample']['answers'] == 10
    assert result['local_evaluation_digest'] == evaluation_digest(local)
    assert Ledger(out / 'ledger.jsonl').verify()[0]
    with pytest.raises(SystemExit, match='replace'):
        module.write_calibration(run, out, 'earth', 20, replace=False)
    summary.write_text(_json.dumps({'evaluation_digest': 'another evaluation'}))
    with pytest.raises(SystemExit, match='no longer match'):
        module.write_calibration(run, out, 'earth', 20, replace=True)


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
    named = module.content_agreement(local, router, ('router', 'command-a'))['by_lang']['th']
    assert named['router_safe'] == 1 and named['command-a_safe'] == 2 and 'local_safe' not in named


# A miniature run (2 languages x harmful/benign x 2 preambles x 3 answers) and offline judges, for main() end to end.
COMMAND_A = 'CohereLabs/c4ai-command-a-03-2025:cohere'


def _write_jsonl(path, rows):
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(_json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _mini_run(tmp_path, router_pair=True):
    import hashlib
    from langsafe.judge.llm_judge import SAFETY_SYSTEM, SYSTEM
    from langsafe.judge.rules import DEGENERATE_RULE

    run = tmp_path / 'full' / 'earth'
    prompts, gens = [], []
    for lang in ('jv', 'th'):
        for kind in ('harmful', 'benign'):
            for i in range(3):
                pid = f'{kind}{i}::{lang}'
                prompts.append({'id': pid, 'group': f'{kind}{i}', 'lang': lang, 'kind': kind, 'source': 'test',
                                'text': f'Please explain topic number {i} of the {kind} list.', 'source_id': str(i),
                                'split': 'heldout'})
                for pre in ('default', 'none'):
                    gens.append({'id': pid, 'group': f'{kind}{i}', 'lang': lang, 'kind': kind, 'preamble': pre,
                                 'model_tag': 'global', 'fingerprint': 'fp', 'split': 'heldout',
                                 'response': f'A distinct reply, item {i}, {lang}, {kind}, preamble {pre}, ends here.'})
    rubric = hashlib.sha256((SYSTEM + SAFETY_SYSTEM + DEGENERATE_RULE).encode()).hexdigest()
    text = {p['id']: p['text'] for p in prompts}

    def judged(judge_set):
        rows = []
        for gen in gens:
            label = 'full_refusal' if gen['kind'] == 'harmful' else 'full_compliance'
            row = {key: gen[key] for key in ('id', 'group', 'lang', 'kind', 'preamble', 'model_tag', 'fingerprint',
                                             'split')}
            row.update(prompt_sha256=hashlib.sha256(text[gen['id']].encode()).hexdigest(),
                       response_sha256=hashlib.sha256(gen['response'].encode()).hexdigest(),
                       judge_set=judge_set, judges={'a': label, 'b': label}, label=label, rubric_hash=rubric,
                       parse_error=False)
            if gen['kind'] == 'harmful':
                row['safety_label'] = 'safe'
            rows.append(row)
        return rows

    _write_jsonl(run / 'data' / 'heldout.jsonl', prompts)
    _write_jsonl(run / 'gens' / 'global.jsonl', gens)
    _write_jsonl(run / 'judged' / 'global.local.jsonl', judged('local'))
    if router_pair:
        _write_jsonl(tmp_path / 'full' / 'earth-router-sample' / 'judged' / 'global.router-sample.jsonl',
                     judged('router'))
    return run


@pytest.fixture
def offline_router(monkeypatch, tmp_path):
    """Every client request is answered locally and recorded: Command A and the 32B pair refuse everything and call
    every answer safe, with fixed token counts. No network, no token."""
    import openai
    from langsafe import llm_client
    from langsafe.judge.llm_judge import SAFETY_SYSTEM

    sent = []

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, **kwargs):
            def create(**request):
                sent.append({'base_url': base_url, 'model': request['model'],
                             'rubric': 'content' if request['messages'][0]['content'] == SAFETY_SYSTEM else 'refusal'})
                label = 'safe' if sent[-1]['rubric'] == 'content' else 'full_refusal'
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=f'{{"label": "{label}", "reason": "x"}}'))],
                    usage=SimpleNamespace(prompt_tokens=500, completion_tokens=20))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)
    monkeypatch.setattr(llm_client.ChatClient, '__init__', llm_client.ChatClient.__init__)  # undo record_usage()
    monkeypatch.setenv('HF_TOKEN', 'not-a-real-token')
    monkeypatch.setenv('LANGSAFE_RUNS_DIR', str(tmp_path))  # main() overwrites both; monkeypatch restores them
    monkeypatch.setenv('LANGSAFE_SPEC', str(ROOT / 'configs' / 'spec.yaml'))
    return sent


def _snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob('*') if path.is_file()}


def _printed(capsys):
    import json as _json
    return _json.loads(capsys.readouterr().out)


def test_command_a_judges_the_same_sample_into_its_own_folder(offline_router, tmp_path, capsys):
    import json as _json

    module = _module()
    run = _mini_run(tmp_path)
    router_out = tmp_path / 'full' / 'earth-router-sample'
    before = _snapshot(router_out)
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2', '--judge-set', 'command-a',
                        '--execute']) == 0
    out = tmp_path / 'full' / 'earth-command-a-sample'
    printed = _printed(capsys)
    assert printed['output'] == str(out)
    # 16 answers (8 harmful): the refusal rubric on each, the content rubric on the harmful ones, all to Command A.
    assert printed['plan']['global']['answers'] == 16 and printed['plan']['global']['calls'] == 24
    assert len(offline_router) == 24 and sum(r['rubric'] == 'content' for r in offline_router) == 8
    assert {(r['base_url'], r['model']) for r in offline_router} == {('https://router.huggingface.co/v1', COMMAND_A)}
    assert _snapshot(router_out) == before  # the 32B sample is read, never written
    assert (out / 'llm_cache.sqlite').is_file() and (out / 'judged' / 'global.command-a-sample.jsonl').is_file()

    comparison = _json.loads((out / 'global.command-a-sample-comparison.json').read_text(encoding='utf-8'))
    assert comparison['judge_set'] == 'command-a' and comparison['judges'][0]['model'] == COMMAND_A
    local = comparison['vs_local_pair']
    # Command A refuses everything; the local pair complies on benign answers. Its own vote is the label, so the
    # benign half disagrees (a pair consensus over one vote would have made every row unclear and n zero).
    assert local['refusal_axis']['shared_rows'] == 16 and not local['refusal_axis']['verified']
    for lang in ('jv', 'th'):
        assert local['refusal_axis']['by_lang'][lang]['n'] == 8
        assert local['refusal_axis']['by_lang'][lang]['agreement'] == 0.5
        content = local['content_axis']['by_lang'][lang]
        assert content['decided_by_both'] == 4 and content['agreement'] == 1.0
        assert content['local_safe'] == 4 and content['command-a_safe'] == 4
    pair = comparison['vs_router_pair']
    assert pair['reference'] == 'router' and pair['refusal_axis']['shared_rows'] == 16
    assert pair['content_axis']['by_lang']['th']['router_safe'] == 4

    usage = _json.loads((out / 'usage.json').read_text(encoding='utf-8'))
    assert usage == [{'judge_set': 'command-a', 'tags': 'global', 'per_cell': 2, 'pilot': None,
                      'usage': {COMMAND_A: {'calls': 24, 'prompt_tokens': 24 * 500, 'completion_tokens': 24 * 20}}}]

    offline_router.clear()  # a rerun is answered from the Command A cache: nothing is billed twice
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2', '--judge-set', 'command-a',
                        '--execute']) == 0
    assert offline_router == [] and _snapshot(router_out) == before


def test_command_a_pilot_without_a_router_sample_says_so(offline_router, tmp_path, capsys):
    import json as _json

    module = _module()
    run = _mini_run(tmp_path, router_pair=False)
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2', '--pilot', '2',
                        '--judge-set', 'command-a', '--execute']) == 0
    out = tmp_path / 'full' / 'earth-command-a-sample'
    comparison = _json.loads((out / 'global.command-a-sample-comparison.json').read_text(encoding='utf-8'))
    assert comparison['pilot'] == 2 and comparison['vs_local_pair']['refusal_axis']['shared_rows'] == 2
    assert 'missing' in comparison['vs_router_pair']
    assert not (tmp_path / 'full' / 'earth-router-sample').exists()
    assert _printed(capsys)['usage'][COMMAND_A]['calls'] == len(offline_router) <= 4


def test_default_judge_set_is_unchanged(offline_router, tmp_path, capsys):
    import json as _json

    module = _module()
    run = _mini_run(tmp_path, router_pair=False)
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2']) == 0  # plan only
    plan = _printed(capsys)
    out = tmp_path / 'full' / 'earth-router-sample'
    assert plan['output'] == str(out) and plan['usage'] == {} and not out.exists()
    assert plan['plan']['global'] == {'answers': 16, 'calls': 64,
                                      'answer_and_prompt_chars': plan['plan']['global']['answer_and_prompt_chars']}
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2', '--execute']) == 0
    assert {r['model'] for r in offline_router} == {'CohereLabs/aya-expanse-32b:cohere', 'Qwen/Qwen3-32B'}
    assert sorted(path.name for path in out.iterdir()) == ['global.router-sample-comparison.json', 'judged',
                                                           'llm_cache.sqlite', 'usage.json']
    comparison = _json.loads((out / 'global.router-sample-comparison.json').read_text(encoding='utf-8'))
    assert list(comparison) == ['tag', 'per_cell', 'pilot', 'refusal_axis', 'content_axis']
    assert comparison['refusal_axis']['verified'] and comparison['refusal_axis']['shared_rows'] == 16
    assert set(comparison['content_axis']['by_lang']['jv']) >= {'local_safe', 'router_safe'}
    usage = _json.loads((out / 'usage.json').read_text(encoding='utf-8'))
    assert list(usage[0]) == ['tags', 'per_cell', 'pilot', 'usage']
    assert not (tmp_path / 'full' / 'earth-command-a-sample').exists()


def test_command_a_plan_only_makes_nothing_and_cannot_calibrate(offline_router, tmp_path, capsys):
    module = _module()
    run = _mini_run(tmp_path)
    assert module.main(['--run', str(run), '--tags', 'global', '--per-cell', '2', '--judge-set', 'command-a']) == 0
    plan = _printed(capsys)
    assert plan['output'] == str(tmp_path / 'full' / 'earth-command-a-sample') and plan['plan']['global']['calls'] == 24
    assert offline_router == [] and not (tmp_path / 'full' / 'earth-command-a-sample').exists()
    with pytest.raises(SystemExit, match='router pair'):
        module.main(['--run', str(run), '--tags', 'global', '--judge-set', 'command-a', '--write-calibration'])
    with pytest.raises(ValueError, match='one-judge'):
        module.single_judge_view([{'judges': {'a': 'full_refusal', 'b': 'full_refusal'}}])
