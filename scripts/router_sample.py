#!/usr/bin/env python3
"""Re-judge a stratified sample of held-out answers with the pre-registered router judges (the 32B pair), and measure
how far the local 8B judges agree with them on the same answers.

The spec makes the router pair the main judges and the local pair a fallback, "agreement with the router pair is
measured on a shared sample before either is used for a headline number". This is that sample.

    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20             # plan only
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --pilot 4 --execute
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --execute
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --write-calibration
    python scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --judge-set command-a --execute

The sample takes `--per-cell` answers from every language x harmful/benign x preamble cell, in a seeded order, so the
same command always picks the same answers. `--pilot N` judges only the first N per tag and reports the tokens the
providers billed, to price the rest. Router calls use this machine's Hugging Face login, which is never printed; they
are billed to that account. Everything goes to <run>-router-sample/ next to the run, which git ignores: the judged
sample (raw verdicts on harmful answers stay private), the comparison per tag, and usage.json.

`--judge-set command-a` judges the same answers with Command A, the Tiny Aya report's own judge (decision D9), as a
single judge on the same two rubrics. It writes to <run>-command-a-sample/ and compares Command A with the local pair
and with the 32B router pair (read from <run>-router-sample/, never written). One judge has no pair consensus, so on
the refusal axis its own vote is the label; these comparisons are descriptive, not a spec calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
USAGE = defaultdict(lambda: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
LOCK = threading.Lock()
# Judge sets beside the spec's router pair. Kept here, not in the frozen spec: they are an extra comparison, not judges
# a headline number rests on.
JUDGE_SETS = {
    'command-a': [{'name': 'command-a', 'base_url': 'https://router.huggingface.co/v1',
                   'model': 'CohereLabs/c4ai-command-a-03-2025:cohere'}],
}


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
    """Count each billed router call's reported tokens. Cache hits never reach the provider, so they are not counted.

    Each client's request function is wrapped once, when the client is made, and counts under a lock: the judge sends
    requests from several threads at once, and swapping the function per request let wrappers stack and count one call
    several times (the first full sample reported about five times its real calls). Calling this twice does nothing."""
    from langsafe import llm_client

    original = llm_client.ChatClient.__init__
    if getattr(original, 'counts_usage', False):
        return

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        create = self.client.chat.completions.create

        def counted(**request):
            response = create(**request)
            usage = getattr(response, 'usage', None)
            with LOCK:
                entry = USAGE[self.model]
                entry['calls'] += 1
                entry['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0) or 0
                entry['completion_tokens'] += getattr(usage, 'completion_tokens', 0) or 0
            return response

        self.client.chat.completions.create = counted

    __init__.counts_usage = True
    llm_client.ChatClient.__init__ = __init__


def content_agreement(local: list[dict], router: list[dict], names: tuple[str, str] = ('local', 'router')) -> dict:
    """Agreement on the content-safety consensus label, the primary axis, per language (Gwet's AC1, Amendment 16).

    `names` label the two sides' safe counts; the defaults keep the router sample's own files unchanged."""
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
                        f'{names[0]}_safe': sum(x == 'safe' for x in a), f'{names[1]}_safe': sum(y == 'safe' for y in b)}
    return {'shared_rows': len(shared), 'by_lang': result}


def single_judge_view(rows: list[dict]) -> list[dict]:
    """A one-judge set's rows with that judge's own refusal vote as the label.

    `consensus` needs two votes, so judge_generations labels every one-judge row `unclear`. The judge's vote is what a
    single-judge method reports (a parse error is already `unclear`). The content label needs no change: one vote is
    unanimous."""
    view = []
    for row in rows:
        if len(row.get('judges', {})) != 1:
            raise ValueError('single_judge_view takes rows from a one-judge set')
        view.append({**row, 'label': next(iter(row['judges'].values()))})
    return view


def compare_single_judge(reference: list[dict], judged: list[dict], names: tuple[str, str]) -> dict:
    """Both axes for a one-judge set against a reference pair on the same answers. `verified` stays false, because the
    spec's calibration is between two pairs; the numbers are descriptive."""
    from langsafe.eval.judge_run import compare_judge_sets

    return {'reference': names[0], 'judge': names[1],
            'refusal_axis': compare_judge_sets(reference, single_judge_view(judged)),
            'content_axis': content_agreement(reference, judged, names)}


def write_calibration(run: Path, out: Path, tag: str, per_cell: int, replace: bool) -> str:
    """Write the run's calibration/<tag>.router-comparison.json from the judged sample, without new router calls.

    The report reads this file to decide whether a locally judged language has its shared-sample router calibration.
    It must describe the tag's whole local evaluation, so the comparison takes every local verdict and the sampled
    router verdicts (they overlap on the sample), and its local digest must equal the one in the tag's judge summary.
    Provenance goes in the sample folder's own ledger: the run's ledger is a hash chain the full run is still extending,
    and a second writer could fork it."""
    from langsafe.eval.judge_run import compare_judge_sets
    from langsafe.provenance import Ledger

    router_path = out / 'judged' / f'{tag}.router-sample.jsonl'
    router = read_rows(router_path)
    local = read_rows(run / 'judged' / f'{tag}.local.jsonl')
    summary = json.loads((run / 'judged' / f'{tag}.local.summary.json').read_text(encoding='utf-8'))
    result = compare_judge_sets(local, router)
    if result['local_evaluation_digest'] != summary.get('evaluation_digest'):
        raise SystemExit(f'{tag}: the local verdicts no longer match their judge summary')
    if not result['verified'] or result['shared_rows'] != len(router):
        raise SystemExit(f'{tag}: comparison is not verified, or not every sampled answer is shared')
    result['sample'] = {'design': 'stratified by language x harmful/benign x preamble, seeded order',
                        'per_cell': per_cell, 'answers': len(router), 'script': 'scripts/router_sample.py',
                        'router_verdicts_sha256': hashlib.sha256(router_path.read_bytes()).hexdigest()}
    result['content_axis'] = content_agreement(local, router)
    target = run / 'calibration' / f'{tag}.router-comparison.json'
    if target.exists() and not replace:
        raise SystemExit(f'{target} exists; pass --replace to overwrite it')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)
    Ledger(out / 'ledger.jsonl').append('calibrate', {
        'tag': tag, 'artifact': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'compare_router': True, 'sample': result['sample']})
    return str(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'experiments' / 'full' / 'earth')
    parser.add_argument('--tags', required=True)
    parser.add_argument('--per-cell', type=int, default=20)
    parser.add_argument('--pilot', type=int, default=0, help='judge only the first N sampled answers per tag')
    parser.add_argument('--execute', action='store_true', help='make the billed router calls')
    parser.add_argument('--write-calibration', action='store_true',
                        help="write the run's calibration/<tag>.router-comparison.json from the judged sample (no calls)")
    parser.add_argument('--replace', action='store_true', help='with --write-calibration: overwrite an existing file')
    parser.add_argument('--judge-set', choices=('router', *JUDGE_SETS), default='router',
                        help="router: the spec's 32B pair (default); command-a: the Tiny Aya report's judge, alone, "
                             'written to <run>-command-a-sample/')
    args = parser.parse_args(argv)
    if args.judge_set != 'router' and args.write_calibration:
        raise SystemExit("--write-calibration takes the spec's router pair, not --judge-set " + args.judge_set)
    run = args.run.resolve()
    router_out = run.parent / (run.name + '-router-sample')
    out = router_out if args.judge_set == 'router' else run.parent / f'{run.name}-{args.judge_set}-sample'
    os.environ['LANGSAFE_RUNS_DIR'] = str(out)  # the router cache lives with the sample, not in the run
    os.environ.setdefault('LANGSAFE_SPEC', str(ROOT / 'configs' / 'spec.yaml'))
    from langsafe.config import load_spec
    from langsafe.data.records import read_prompts
    from langsafe.eval.judge_run import compare_judge_sets, generation_key, judge_generations

    spec = load_spec()
    judges = spec['judging']['judges'] if args.judge_set == 'router' else JUDGE_SETS[args.judge_set]
    prompts = {prompt.id: prompt for prompt in read_prompts(run / 'data' / 'heldout.jsonl')}
    plan = {}
    for tag in args.tags.split(','):
        sample = stratified_sample(read_rows(run / 'gens' / f'{tag}.jsonl'), args.per_cell, spec['data']['seed'],
                                   generation_key)
        if args.pilot:
            sample = sample[:args.pilot]
        chars = sum(len(prompts[row['id']].text) + min(len(row['response'] or ''), 3000) for row in sample)
        if args.judge_set == 'router':
            plan[tag] = {'answers': len(sample), 'calls': 4 * len(sample), 'answer_and_prompt_chars': chars}
        else:  # the refusal rubric on every answer, the content rubric on harmful ones; before cache hits
            plan[tag] = {'answers': len(sample), 'calls': len(sample) + sum(row['kind'] == 'harmful' for row in sample),
                         'answer_and_prompt_chars': chars}
        if not args.execute:
            continue
        if not os.environ.get('HF_TOKEN'):
            from huggingface_hub import get_token
            token = get_token()
            if not token:
                raise SystemExit('No Hugging Face login on this machine; run `hf auth login` yourself first')
            os.environ['HF_TOKEN'] = token
        record_usage()
        judged = out / 'judged' / f'{tag}.{args.judge_set}-sample.jsonl'
        judged.parent.mkdir(parents=True, exist_ok=True)
        judge_generations(sample, prompts, judges, judged, audit_size=0,
                          min_kappa=spec['judging']['min_judge_kappa'], min_ac1=spec['judging']['min_judge_ac1'],
                          primary_safety_axis=spec.get('experiment', {}).get('primary_safety_axis', 'refusal'))
        router = read_rows(judged)
        # A tag has one answer per prompt and preamble; judged rows carry other fields than generations, so match on these.
        keys = {(row['id'], row['preamble']) for row in sample}
        local = [row for row in read_rows(run / 'judged' / f'{tag}.local.jsonl') if (row['id'], row['preamble']) in keys]
        if len(local) != len(sample) or len(router) != len(sample):
            raise SystemExit(f'{tag}: {len(local)} local and {len(router)} router verdicts for {len(sample)} answers')
        if args.judge_set == 'router':
            comparison = {'tag': tag, 'per_cell': args.per_cell, 'pilot': args.pilot or None,
                          'refusal_axis': compare_judge_sets(local, router),
                          'content_axis': content_agreement(local, router)}
        else:
            comparison = {'tag': tag, 'per_cell': args.per_cell, 'pilot': args.pilot or None,
                          'judge_set': args.judge_set, 'judges': judges,
                          'note': 'one judge: its own refusal vote is the label; verified is false by design, '
                                  'since the spec calibrates pair against pair',
                          'vs_local_pair': compare_single_judge(local, router, ('local', args.judge_set))}
            pair_path = router_out / 'judged' / f'{tag}.router-sample.jsonl'  # read only; never written here
            pair = ([row for row in read_rows(pair_path) if (row['id'], row['preamble']) in keys]
                    if pair_path.is_file() else [])
            comparison['vs_router_pair'] = (compare_single_judge(pair, router, ('router', args.judge_set))
                                            if len(pair) == len(sample) else
                                            {'missing': f'{pair_path.name} has {len(pair)} of these {len(sample)} answers'})
        target = out / f'{tag}.{args.judge_set}-sample-comparison.json'
        target.write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        plan[tag]['comparison'] = str(target)
    if args.write_calibration:
        if args.pilot:
            raise SystemExit('A pilot is too small to calibrate with; run the full sample first')
        for tag in args.tags.split(','):
            plan.setdefault(tag, {})['calibration'] = write_calibration(run, out, tag, args.per_cell, args.replace)
    if args.execute:
        usage_path = out / 'usage.json'
        earlier = json.loads(usage_path.read_text(encoding='utf-8')) if usage_path.is_file() else []
        entry = {'tags': args.tags, 'per_cell': args.per_cell, 'pilot': args.pilot or None, 'usage': dict(USAGE)}
        earlier.append(entry if args.judge_set == 'router' else {'judge_set': args.judge_set, **entry})
        usage_path.write_text(json.dumps(earlier, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'plan': plan, 'usage': dict(USAGE)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
