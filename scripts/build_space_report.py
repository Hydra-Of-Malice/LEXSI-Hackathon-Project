"""Build space/results.json, the only numbers the demo Space shows, from the sanitised public export.

    python scripts/build_space_report.py            # write space/results.json
    python scripts/build_space_report.py --check    # exit 1 if space/results.json is out of date

The input is the export that `scripts/export_public.py` wrote and the Hugging Face dataset publishes
(`runs/experiments/public/earth-full` by default). Its `reports/report.json` is already sanitised. This script keeps a
smaller allowlist on top of it: per-language rates with their intervals and sample sizes, the gap tests, the paired
comparisons, the acceptance gates and which knob each arm chose. It adds the pre-registered acceptance limits from
`configs/spec.yaml`, only when that spec's digest matches the report's. Every value must be a number, a flag or a short
identifier, so no model answer, prompt, local path or per-item vector can reach the Space, even if the export changes.

Nobody copies numbers by hand: rerun this after a new export, and the Space follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / 'runs' / 'experiments' / 'public' / 'earth-full'
DEFAULT_RUN_REPORT = ROOT / 'runs' / 'experiments' / 'full' / 'earth' / 'report.json'
DEFAULT_OUT = ROOT / 'space' / 'results.json'
DEFAULT_SPEC = ROOT / 'configs' / 'spec.yaml'

SCHEMA = 'langsafe.space-results/1'
DATASET = 'HydraOfMalice/tiny-aya-earth-langsafe-eval'
REPORT_FILE = 'reports/report.json'
MANIFEST_FILE = 'EXPORT_MANIFEST.json'
DIGITS = 6      # Enough for any displayed figure, and it drops float noise such as -1e-17 at a zero bound.

_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.\-/]{0,79}')
_GAP = re.compile(r'[\w.\-/]+: [a-z0-9 ,/_\-]{1,120}|No repair-split selection of the final weight candidate was recorded')
_FORBIDDEN_WORDS = frozenset({'response', 'raw', 'completion', 'answer', 'prompt', 'text', 'output', 'generation',
                              'reason', 'rationale', 'explanation', 'item', 'path', 'model', 'host'})


class BuildError(Exception):
    """The export is missing, stale or not in the expected shape. Nothing is written."""


# ---- allowlist ------------------------------------------------------------------------------------------------------

NUM, FLAG, NAME, GAP, INTERVAL = 'number', 'flag', 'name', 'gap', 'interval'


class Each:
    """A mapping with free keys (models, languages, conditions), each value following `spec`."""

    def __init__(self, spec):
        self.spec = spec


class ListOf:
    def __init__(self, spec):
        self.spec = spec


RATE = {'rate': NUM, 'ci': INTERVAL, 'n': NUM, 'partial': NUM, 'unclear': NUM, 'unsafe': NUM, 'invalid': NUM,
        'disputed': NUM}
METRICS = {metric: RATE for metric in ('content_safe', 'harmful_refusal', 'harmful_compliance', 'over_refusal',
                                       'degenerate')}
CAPABILITY = {'accuracy': NUM, 'ci95': INTERVAL, 'n': NUM}
PAIRED = {'delta': NUM, 'ci': INTERVAL, 'p_mcnemar': NUM, 'n': NUM}
COMPARISON = {'baseline': NAME, 'pairing_complete': FLAG,
              'safety': Each({metric: PAIRED for metric in ('content_safe', 'harmful_refusal', 'over_refusal')}),
              'capability': Each(PAIRED)}
GATE = {'pass': FLAG, 'status': NAME, 'exploratory': FLAG,
        'harmful_gap_closed': {'value': NUM, 'lang': NAME, 'gap_ci_low': NUM, 'status': NAME, 'pass': FLAG},
        'over_refusal_increase': {'by_lang': Each(NUM), 'pass': FLAG},
        'capability_drop': {'by_lang': Each(NUM), 'pass': FLAG},
        'coverage': {'pass': FLAG}, 'eligibility': {'pass': FLAG}}
ARM_CHOICE = {'arm': NAME, 'knob': NUM, 'tag': NAME}
MODEL = {'status': NAME,
         'eligibility': {'eligible': FLAG, 'status': NAME, 'statistic': NAME, 'min_agreement': NUM,
                         'by_lang': Each({'status': NAME, 'measurable': FLAG, 'agreement': NUM})},
         'conditions': Each(Each(METRICS)),
         'capability_by_condition': Each(Each(CAPABILITY))}
REPORT = {
    'family': NAME, 'profile': NAME, 'spec_digest': NAME, 'spec_frozen': FLAG, 'judge_set': NAME, 'validated': FLAG,
    'empirical_repair_success': FLAG, 'acceptance_candidate': NAME, 'confirmed_conditions': ListOf(NAME),
    'translation_quality_status': NAME, 'validation_gaps': ListOf(GAP),
    'models': Each(MODEL),
    'gap_confirmation': {'metric': NAME, 'tests': NUM, 'alpha': NUM,
                         'conditions': Each(Each({**PAIRED, 'confirmed': FLAG, 'measurable': FLAG}))},
    'paired_comparisons': Each({'vs_drifted': COMPARISON, 'vs_reference': COMPARISON}),
    'thresholds': Each(GATE),
    'selection': {'split': NAME, 'preamble': NAME, 'worst_language': NAME,
                  'arms': Each({'knob': NUM, 'selected': ARM_CHOICE}),
                  'selected_for_quantization': ARM_CHOICE,
                  'controls': ListOf({**ARM_CHOICE, 'random_seed': NUM})}}


def _number(value, where: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BuildError(f'{where}: expected a number')
    return value if isinstance(value, int) else round(value, DIGITS) + 0.0     # + 0.0 turns -0.0 into 0.0.


def _leaf(kind: str, value, where: str):
    if kind == NUM:
        return _number(value, where)
    if kind == FLAG:
        if value is None or isinstance(value, bool):
            return value
        raise BuildError(f'{where}: expected true or false')
    if kind == INTERVAL:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise BuildError(f'{where}: expected a [low, high] interval')
        return [_number(bound, where) for bound in value]
    pattern = _NAME if kind == NAME else _GAP
    if value is None or (isinstance(value, str) and pattern.fullmatch(value)):
        return value
    raise BuildError(f'{where}: not a short identifier')


def reduce(spec, value, where: str = 'report'):
    """Keep only what `spec` names. A named field holding the wrong kind of value stops the build."""
    if isinstance(spec, str):
        return _leaf(spec, value, where)
    if isinstance(spec, ListOf):
        if not isinstance(value, list):
            raise BuildError(f'{where}: expected a list')
        return [reduce(spec.spec, item, f'{where}[{index}]') for index, item in enumerate(value)]
    if not isinstance(value, dict):
        raise BuildError(f'{where}: expected an object')
    if isinstance(spec, Each):
        for key in value:
            if not (isinstance(key, str) and _NAME.fullmatch(key)):
                raise BuildError(f'{where}: a key is not a short identifier')
        return {key: reduce(spec.spec, item, f'{where}.{key}') for key, item in value.items()}
    return {key: reduce(spec[key], item, f'{where}.{key}') for key, item in value.items() if key in spec}


def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keys(value)


def check_keys(result: dict) -> None:
    """Belt and braces: no key may even sound like it carries text. Language and model keys are short codes."""
    for key in _keys(result.get('results', {})):
        if _FORBIDDEN_WORDS.intersection(re.split(r'[^a-z]+', key.casefold())):
            raise BuildError(f'refused key {key!r}')


# ---- inputs ---------------------------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise BuildError(f'cannot read {path.name}: {type(error).__name__}') from error
    if not isinstance(document, dict):
        raise BuildError(f'{path.name} is not a JSON object')
    return document


LIMITS = {'harmful_gap_closed_min': NUM, 'over_refusal_increase_max': NUM, 'capability_drop_max': NUM}


def acceptance_limits(spec_path: Path | None, digest) -> dict | None:
    """The pre-registered acceptance limits, but only from the very spec the run used (same digest)."""
    if spec_path is None or not Path(spec_path).is_file():
        return None
    from langsafe.config import load_spec, spec_digest

    spec = load_spec(spec_path)
    if spec_digest(spec) != digest:
        raise BuildError("the spec's digest differs from the report's; pass the spec the run used, or --spec ''")
    return reduce(LIMITS, spec.get('thresholds') or {}, 'spec.thresholds')


def build(export_dir: Path, run_report: Path | None = None, spec_path: Path | None = None) -> dict:
    """Return the Space's results document. Fails if the export's report does not match its own manifest, or if the
    run's report.json has changed since the export was made."""
    export_dir = Path(export_dir)
    manifest = read_json(export_dir / MANIFEST_FILE)
    report_path = export_dir / REPORT_FILE
    entry = next((item for item in manifest.get('exported', []) if item.get('file') == REPORT_FILE), None)
    if entry is None:
        raise BuildError(f'{MANIFEST_FILE} does not list {REPORT_FILE}')
    if not report_path.is_file() or sha256(report_path) != entry.get('sha256'):
        raise BuildError(f'{REPORT_FILE} does not match the hash in {MANIFEST_FILE}; re-run the export')
    if run_report is not None and Path(run_report).is_file() and sha256(Path(run_report)) != entry.get('source_sha256'):
        raise BuildError("the run's report.json changed after the export was made; re-run scripts/export_public.py")
    report = read_json(report_path)
    if not isinstance(report.get('models'), dict) or not report['models']:
        raise BuildError('the report has no models')
    result = {
        'schema': SCHEMA,
        'notice': 'Aggregate results only: rates, intervals, sample sizes and paired comparisons. No model answers or '
                  'prompts. Built by scripts/build_space_report.py from the public dataset; do not edit by hand.',
        'source': {'dataset': DATASET, 'file': REPORT_FILE, 'sha256': entry['sha256'],
                   'exporter_sha256': manifest.get('exporter_sha256') if _NAME.fullmatch(
                       str(manifest.get('exporter_sha256'))) else None},
        'results': reduce(REPORT, report),
    }
    limits = acceptance_limits(spec_path, report.get('spec_digest'))
    if limits:
        result['results']['acceptance_limits'] = limits
    check_keys(result)
    return result


def render(result: dict) -> str:
    return json.dumps(result, ensure_ascii=True, indent=1) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--export', type=Path, default=DEFAULT_EXPORT, help='public export folder (default: %(default)s)')
    parser.add_argument('--run-report', type=Path, default=DEFAULT_RUN_REPORT,
                        help="the run's own report.json, to check the export is current; skipped if absent")
    parser.add_argument('--spec', default=str(DEFAULT_SPEC),
                        help="the run's frozen spec, for the acceptance limits; '' leaves them out (default: %(default)s)")
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='output file (default: %(default)s)')
    parser.add_argument('--check', action='store_true', help='compare with --out instead of writing it')
    args = parser.parse_args(argv)
    try:
        text = render(build(args.export, args.run_report, Path(args.spec) if args.spec else None))
    except BuildError as error:
        print(f'build_space_report: {error}', file=sys.stderr)
        return 2
    if args.check:
        current = args.out.read_text(encoding='utf-8') if args.out.is_file() else None
        if current != text:
            print(f'{args.out} is out of date; run scripts/build_space_report.py', file=sys.stderr)
            return 1
        print(f'{args.out} is up to date')
        return 0
    args.out.write_text(text, encoding='utf-8', newline='\n')
    print(f'Wrote {args.out} ({len(text.encode())} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
