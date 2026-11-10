#!/usr/bin/env python3
"""Re-judge a stratified sample of held-out answers with the pre-registered router judges (the 32B pair), and measure
how far the local 8B judges agree with them on the same answers.

The spec makes the router pair the main judges and the local pair a fallback, "agreement with the router pair is
measured on a shared sample before either is used for a headline number". This is that sample.

    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20             # plan only
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --pilot 4 --execute
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --execute

The sample takes `--per-cell` answers from every language x harmful/benign x preamble cell, in a seeded order, so the
same command always picks the same answers. `--pilot N` judges only the first N per tag and reports the tokens the
providers billed, to price the rest. Router calls use this machine's Hugging Face login, which is never printed; they
are billed to that account. Everything goes to <run>-router-sample/ next to the run, which git ignores: the judged
sample (raw verdicts on harmful answers stay private), the comparison per tag, and usage.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
USAGE = defaultdict(lambda: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0})


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stratified_sample(rows: list[dict], per_cell: int, seed: int, key) -> list[dict]:
    """`per_cell` rows from every (language, kind, preamble) cell, in an order fixed by the seed and each row's key."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row['lang'], row['kind'], row['preamble'])].append(row)
    chosen = []
    for cell in sorted(cells):
        ordered = sorted(cells[cell], key=lambda row: hashlib.sha256(f'{seed}|router-sample|{key(row)}'.encode()).hexdigest())
        chosen.extend(ordered[:per_cell])
    return chosen


def record_usage() -> None:
    """Wrap the router client so each billed call's reported tokens are counted. Cache hits cost nothing."""
    from langsafe import llm_client

    original = llm_client.ChatClient.chat

    def chat(self, messages, **params):
        params = {'temperature': 0, 'max_tokens': 512, **params}
        if self._cache.get(self._cache.key(messages, params)) is not None:
            return original(self, messages, **params)
        create = self.client.chat.completions.create

        def counted(**kwargs):
            response = create(**kwargs)
            usage = getattr(response, 'usage', None)
            entry = USAGE[self.model]
            entry['calls'] += 1
            entry['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0) or 0
            entry['completion_tokens'] += getattr(usage, 'completion_tokens', 0) or 0
            return response

        self.client.chat.completions.create = counted
        try:
            return original(self, messages, **params)
        finally:
            self.client.chat.completions.create = create

    llm_client.ChatClient.chat = chat


def content_agreement(local: list[dict], router: list[dict]) -> dict:
    """Agreement on the content-safety consensus label, the primary axis, per language (Gwet's AC1, Amendment 16)."""
    from langsafe.eval.judge_run import _evidence_key
    from langsafe.judge.agreement import gwet_ac1

    local_by, router_by = {_evidence_key(r): r for r in local}, {_evidence_key(r): r for r in router}
    shared = sorted(set(local_by) & set(router_by))
    result = {}
    for lang in sorted({local_by[k]['lang'] for k in shared}):
        keys = [k for k in shared if local_by[k]['lang'] == lang]
        decided = [k for k in keys if local_by[k].get('safety_label') not in (None, 'disputed')
                   and router_by[k].get('safety_label') not in (None, 'disputed')]
        a = [local_by[k]['safety_label'] for k in decided]
        b = [router_by[k]['safety_label'] for k in decided]
        categories = tuple(sorted(set(a) | set(b))) or ('safe',)
        result[lang] = {'shared': len(keys), 'decided_by_both': len(decided),
                        'agreement': sum(x == y for x, y in zip(a, b)) / len(decided) if decided else None,
                        'ac1': gwet_ac1(a, b, categories) if len(categories) > 1 and decided else None,
                        'local_safe': sum(x == 'safe' for x in a), 'router_safe': sum(y == 'safe' for y in b)}
    return {'shared_rows': len(shared), 'by_lang': result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'experiments' / 'full' / 'earth')
    parser.add_argument('--tags', required=True)
    parser.add_argument('--per-cell', type=int, default=20)
    parser.add_argument('--pilot', type=int, default=0, help='judge only the first N sampled answers per tag')
    parser.add_argument('--execute', action='store_true', help='make the billed router calls')
    args = parser.parse_args(argv)
    run = args.run.resolve()
    out = run.parent / (run.name + '-router-sample')
    os.environ['LANGSAFE_RUNS_DIR'] = str(out)  # the router cache lives with the sample, not in the run
    os.environ.setdefault('LANGSAFE_SPEC', str(ROOT / 'configs' / 'spec.yaml'))
    from langsafe.config import load_spec
    from langsafe.eval.judge_run import compare_judge_sets, generation_key, judge_generations

    spec = load_spec()
    prompts = {row['id']: SimpleNamespace(text=row['text']) for row in read_rows(run / 'data' / 'heldout.jsonl')}
    plan = {}
    for tag in args.tags.split(','):
        sample = stratified_sample(read_rows(run / 'gens' / f'{tag}.jsonl'), args.per_cell, spec['data']['seed'],
                                   generation_key)
        if args.pilot:
            sample = sample[:args.pilot]
        chars = sum(len(prompts[row['id']].text) + min(len(row['response'] or ''), 3000) for row in sample)
        plan[tag] = {'answers': len(sample), 'calls': 4 * len(sample), 'answer_and_prompt_chars': chars}
        if not args.execute:
            continue
        if not os.environ.get('HF_TOKEN'):
            from huggingface_hub import get_token
            token = get_token()
            if not token:
                raise SystemExit('No Hugging Face login on this machine; run `hf auth login` yourself first')
            os.environ['HF_TOKEN'] = token
        record_usage()
        judged = out / 'judged' / f'{tag}.router-sample.jsonl'
        judged.parent.mkdir(parents=True, exist_ok=True)
        judge_generations(sample, prompts, spec['judging']['judges'], judged, audit_size=0,
                          min_kappa=spec['judging']['min_judge_kappa'], min_ac1=spec['judging']['min_judge_ac1'],
                          primary_safety_axis=spec.get('experiment', {}).get('primary_safety_axis', 'refusal'))
        router = read_rows(judged)
        # A tag has one answer per prompt and preamble; judged rows carry other fields than generations, so match on these.
        keys = {(row['id'], row['preamble']) for row in sample}
        local = [row for row in read_rows(run / 'judged' / f'{tag}.local.jsonl') if (row['id'], row['preamble']) in keys]
        if len(local) != len(sample) or len(router) != len(sample):
            raise SystemExit(f'{tag}: {len(local)} local and {len(router)} router verdicts for {len(sample)} answers')
        comparison = {'tag': tag, 'per_cell': args.per_cell, 'pilot': args.pilot or None,
                      'refusal_axis': compare_judge_sets(local, router), 'content_axis': content_agreement(local, router)}
        (out / f'{tag}.router-sample-comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        plan[tag]['comparison'] = str(out / f'{tag}.router-sample-comparison.json')
    if args.execute:
        usage_path = out / 'usage.json'
        earlier = json.loads(usage_path.read_text(encoding='utf-8')) if usage_path.is_file() else []
        earlier.append({'tags': args.tags, 'per_cell': args.per_cell, 'pilot': args.pilot or None, 'usage': dict(USAGE)})
        usage_path.write_text(json.dumps(earlier, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'plan': plan, 'usage': dict(USAGE)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
