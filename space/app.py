"""Public demo for "Same model, different language, different safety" (Lexsi hackathon, Track 2b).

A results explorer for the final Tiny Aya Earth run. It reads `results.json`, a file of aggregate numbers that
`scripts/build_space_report.py` builds from the sanitised public dataset, and draws them per language and per repair.
It never shows a model's answer, and the file holds no model text to show.

An optional side-by-side tab can run two 4-bit models on a fixed list of six everyday benign prompts. It is off unless
LANGSAFE_ENABLE_LIVE=1 is set and llama-cpp-python is installed, and the Space's requirements leave that package out,
so the public Space never loads a model. Even when it is on there is no free-text input: the click handler accepts a
language code and a prompt ID, and looks the text up here.

This folder is self-contained because a Space receives only this folder, so nothing here imports langsafe.
"""
from __future__ import annotations

import html
import json
import logging
import math
import os
import re
from pathlib import Path

import gradio as gr

logger = logging.getLogger('langsafe.space')

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / 'results.json'
PROMPTS_PATH = HERE / 'prompts.json'
RESULTS_SCHEMA = 'langsafe.space-results/1'

REPORT_ENV = 'LANGSAFE_REPORT_PATH'
LIVE_ENV = 'LANGSAFE_ENABLE_LIVE'
THREADS_ENV = 'LANGSAFE_N_THREADS'
MAX_TOKENS_ENV = 'LANGSAFE_MAX_NEW_TOKENS'

DATASET_URL = 'https://huggingface.co/datasets/HydraOfMalice/tiny-aya-earth-langsafe-eval'
CODE_URL = 'https://github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project'
BASE_MODEL_URLS = {'global': 'https://huggingface.co/CohereLabs/tiny-aya-global',
                   'earth': 'https://huggingface.co/CohereLabs/tiny-aya-earth'}
_HF = 'https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-'
# Report tag -> the published repository. Random-layer controls and the NF4 copy are deliberately not published.
MODEL_REPOS = {'earth-circuit_merge-0p5': _HF + 'circuit-merge-0p5', 'earth-circuit_merge-0p5-q4': _HF + 'circuit-merge-0p5-gguf',
               'earth-resta-0p1': _HF + 'resta-0p1', 'earth-circuit-0p1': _HF + 'circuit-0p1',
               'earth-merge_back-0p5': _HF + 'merge-back-0p5', 'earth-steer-1p0': _HF + 'steer-1p0'}
REPO_CONTENTS = {'earth-circuit_merge-0p5-q4': 'Q4_K_M GGUF', 'earth-steer-1p0': 'refusal direction, no weights'}

ALL = 'all'
# Earth's languages, in the order the page lists them. Indonesian only ever has a capability value.
LANGUAGE_NAMES = {'en': 'English', 'ar': 'Arabic', 'sw': 'Swahili', 'jv': 'Javanese', 'bn': 'Bengali', 'th': 'Thai',
                  'ko': 'Korean', 'id': 'Indonesian (stand-in for Javanese)'}
PRECISION_NAMES = {'fp16': 'FP16', 'q4': 'Q4_K_M', 'nf4': 'NF4'}
PREAMBLE_NAMES = {'default': 'Default preamble', 'none': 'Preamble removed'}
TRANSLATION_OK = {'human_reviewed', 'not_applicable'}
# name, knob symbol, what changes. From the article's arm table.
ARMS = {'circuit_merge': ('Circuit merge', 'λ', 'Earth moved toward Global, only in the top 20% of layers by circuit score'),
        'resta': ('ReSta', 'α', 'a safety vector (Global minus Base) added across the whole model'),
        'circuit': ('Circuit (C-ΔΘ)', 'strength', 'the same safety vector, only in the top 20% of layers'),
        'merge_back': ('Merge-back (WiSE-FT)', 'λ', 'a weighted average of Earth and Global across the whole model'),
        'steer': ('Steering', 'strength', 'a refusal direction added to activations at run time; no weights change')}
TAG = re.compile(r'(?P<base>[a-z]+)(?:-(?P<arm>circuit_merge|merge_back|resta|circuit|steer)-(?P<knob>\d+p\d+)'
                 r'(?:-rand(?P<seed>\d+))?)?(?:-(?P<precision>q4|nf4))?')
ROLE_ORDER = ('reference', 'original', 'selected', 'arm', 'control', 'other')
ROLE_FILTERS = {'main': ('Global, Earth and the selected repair', ('reference', 'original', 'selected')),
                'arm': ('Other repair arms', ('arm',)), 'control': ('Random-layer controls', ('control',))}

NOT_MEASURED = 'not measured'
NOT_MEASURABLE = 'not measurable'
TABLE_HEADERS = ['Model', 'Precision', 'Preamble', 'Language', 'Measurement status',
                 'Content-safe on harmful prompts (primary, provisional; higher is safer)',
                 'Harmful: unsafe / invalid / disputed',
                 'Degenerate answers, harmful and benign (lower is better)', 'Harmful refusal', 'Harmful compliance',
                 'Harmful: partial / unclear', 'Benign over-refusal (lower is better)',
                 'Benign: partial / unclear', 'Capability (higher is better)']
COLUMN_WIDTHS = ['220px', '90px', '120px', '110px', '110px', '200px', '130px', '180px', '180px', '180px', '120px',
                 '180px', '120px', '180px']
REFUSAL_MIX = ('partial', 'unclear')
CONTENT_MIX = ('unsafe', 'invalid', 'disputed')
SAFETY_METRICS = ('content_safe', 'harmful_refusal', 'harmful_compliance', 'over_refusal', 'degenerate')

# metric: (title, how to read it, +1 when higher is better)
CHART_METRICS = {'content_safe': ('Content-safe rate on harmful prompts', 'higher is safer', 1),
                 'over_refusal': ('Over-refusal on benign prompts', 'lower is better', -1),
                 'capability': ('Capability: Global-MMLU-Lite accuracy', 'higher is better', 1),
                 'degenerate': ('Degenerate answers, harmful and benign prompts', 'lower is better', -1)}
DELTA_METRICS = ('content_safe', 'over_refusal', 'capability')
# role, legend label, marker, colour variable, light fallback. The first three slots of the validated categorical
# palette (blue, orange, aqua); the markers differ too, so colour never carries identity alone.
SERIES = (('reference', 'Global (reference)', 'circle', '--ls-s1', '#2a78d6'),
          ('original', 'Earth (original)', 'square', '--ls-s2', '#eb6834'),
          ('selected', 'Selected repair', 'diamond', '--ls-s3', '#1baf7a'))
REPAIR_COLOUR = ('--ls-s3', '#1baf7a')

ROLES = ('original', 'repaired')
ROLE_LABELS = {'original': 'original Earth model', 'repaired': 'selected repair'}
N_CTX = 1024             # The longest prompt renders to well under 400 tokens with the default preamble, plus 256 new ones.
N_BATCH = 128            # Saves about 0.8 GB per model over the default of 512, with no measured slowdown.
DEFAULT_MAX_NEW_TOKENS = 200
SPEC_MAX_NEW_TOKENS = 256
# Tiny Aya's tokenizer calls token 6 EOS, but generation stops at token 3. A file exported with 6 runs past the end of its
# answer, which would make a repaired model look worse for a reason unrelated to the repair, so both models are pinned.
GENERATION_EOS_TOKEN = 3
QUEUE_SIZE = 16

PAGE_CSS = '''
.virtual-table-viewport .header-cell.first-column, .virtual-table-viewport .body-cell.first-column {
    position: sticky !important; left: 0; z-index: 3; background: var(--background-fill-primary);
}
.ls-viz { --ls-s1: #2a78d6; --ls-s2: #eb6834; --ls-s3: #1baf7a; --ls-good: #0ca30c; --ls-bad: #d03b3b; }
.dark .ls-viz { --ls-s1: #3987e5; --ls-s2: #d95926; --ls-s3: #199e70; }
.ls-viz svg.ls-plot { display: block; width: 100%; height: auto; }
.ls-legend svg { width: 16px; height: 16px; flex: none; }
.ls-viz .ls-row:hover .ls-hit { fill: var(--border-color-primary); fill-opacity: .35; }
.ls-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 6px 0 2px; font-size: 13px; }
.ls-legend span { display: inline-flex; align-items: center; gap: 6px; }
.ls-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 4px 0 8px; }
.ls-tile { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 14px;
           background: var(--background-fill-secondary); }
.ls-tile .ls-kicker { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; opacity: .75; }
.ls-tile .ls-big { font-size: 22px; font-weight: 600; margin: 2px 0 4px; }
.ls-tile p { margin: 0; font-size: 13px; line-height: 1.45; }
.ls-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.ls-table th, .ls-table td { border-bottom: 1px solid var(--border-color-primary); padding: 4px 8px; text-align: left; }
'''
# Gradio keeps a generic upload route even on a page with no upload component. Nothing here needs it, and a public Space
# should not offer anonymous disk storage, so the limit refuses every real file.
LAUNCH_OPTIONS = {'mcp_server': False, 'ssr_mode': False, 'show_error': False, 'max_file_size': '1b', 'css': PAGE_CSS,
                  'footer_links': ['gradio', 'settings']}


# ---------------------------------------------------------------------------------------------------------------------
# Report parsing. Only the aggregate fields named here are ever read or displayed.
# ---------------------------------------------------------------------------------------------------------------------

class ReportError(ValueError):
    """The file is unreadable, or is not an aggregate report."""


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def finite(value, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and low <= value <= high else None


def proportion(value) -> float | None:
    """A rate in [0, 1], or None. Anything else counts as missing, never as zero."""
    return finite(value, 0, 1)


def interval(bounds, low: float = 0, high: float = 1) -> tuple[float, float] | None:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        first, second = finite(bounds[0], low, high), finite(bounds[1], low, high)
        if first is not None and second is not None and first <= second:
            return first, second
    return None


def count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def metric_cell(raw, value_key: str = 'rate', ci_key: str = 'ci') -> dict | None:
    raw = as_dict(raw)
    value = proportion(raw.get(value_key))
    if value is None:
        return None
    return {'rate': value, 'ci': interval(raw.get(ci_key)), 'n': count(raw.get('n')),
            **{key: proportion(raw.get(key)) for key in (*REFUSAL_MIX, *CONTENT_MIX)}}


def paired_cell(raw) -> dict | None:
    """A paired change in rate units, with its interval and McNemar p."""
    raw = as_dict(raw)
    delta = finite(raw.get('delta'), -1, 1)
    if delta is None:
        return None
    return {'delta': delta, 'ci': interval(raw.get('ci'), -1, 1), 'p': proportion(raw.get('p_mcnemar')),
            'n': count(raw.get('n')), 'confirmed': raw.get('confirmed') is True}


def precision_of(tag: str) -> str:
    return 'q4' if tag.endswith('-q4') else 'nf4' if tag.endswith('-nf4') else 'fp16'


def base_tag(tag: str) -> str:
    return tag.removesuffix('-q4').removesuffix('-nf4')


def describe_model(tag: str, family: str | None, selected: str | None) -> dict:
    """Role and plain-language label for a report tag, such as earth-circuit_merge-0p5-q4."""
    match = TAG.fullmatch(tag)
    parts = match.groupdict() if match else {}
    base = base_tag(tag)
    arm, knob, seed = parts.get('arm'), parts.get('knob'), parts.get('seed')
    name, symbol, _ = ARMS.get(arm, (arm, 'knob', ''))
    knob_text = knob.replace('p', '.') if knob else ''
    if base == 'global':
        role, label = 'reference', 'Global (reference)'
    elif base == family:
        role, label = 'original', f'{str(family).title()} (original)'
    elif seed:
        role, label = 'control', f'Random-layer control, seed {seed}'
    elif arm and selected and base == base_tag(selected):
        role, label = 'selected', f'{name} {symbol}={knob_text} (selected repair)'
    elif arm:
        role, label = 'arm', f'{name} {symbol}={knob_text}'
    else:
        role, label = 'other', tag
    return {'tag': tag, 'base': base, 'role': role, 'label': label, 'arm': arm, 'precision': precision_of(tag),
            'repo': MODEL_REPOS.get(tag)}


def has_content_safe(raw: dict) -> bool:
    return any('content_safe' in as_dict(cells)
               for model in as_dict(raw.get('models')).values()
               for condition in as_dict(as_dict(model).get('conditions')).values()
               for cells in as_dict(condition).values())


def provisional_reasons(raw: dict) -> list[str]:
    """Why this report cannot be read as a validated result. Every check fails closed when a field is absent."""
    reasons = []
    if raw.get('validated') is not True:
        reasons.append('The report does not say it is validated.')
    if raw.get('profile') != 'full':
        reasons.append(f"Profile is {raw.get('profile', 'unknown')!s}, not the full experiment.")
    if raw.get('spec_frozen') is not True:
        reasons.append('The pre-registered spec is not frozen.')
    if raw.get('translation_quality_status') not in TRANSLATION_OK:
        reasons.append('Translations have not been reviewed by native readers.')
    models = as_dict(raw.get('models'))
    ineligible = [tag for tag, model in models.items()
                  if as_dict(as_dict(model).get('eligibility')).get('eligible') is not True]
    if ineligible:
        reasons.append(f'{len(ineligible)} of {len(models)} models lack complete judge calibration '
                       '(the report holds no human calibration labels).')
    gaps = raw.get('validation_gaps')
    if gaps or not isinstance(gaps, list):
        reasons.append(f'{len(gaps) if isinstance(gaps, list) else "Unknown number of"} validation gap(s) recorded.')
    # SPEC.md, amendments 10 and 11, and the report's own limits: the report's router calibration checks refusal labels
    # only, so `validated` does not fully cover the content axis, whatever the report says. The separate content check
    # on the router sample is described in the article, not in the report, so its numbers are not shown here.
    if has_content_safe(raw):
        reasons.append("Content-safe numbers are outside the report's calibration against the 32B router judges, which "
                       'covers refusal labels only. A separate sample, described in the article, checked content labels.')
    return reasons


def unwrap(raw: dict) -> tuple[dict, dict]:
    """`results.json` wraps the report with its source; a bare report.json (set through the environment) also works."""
    if raw.get('schema') == RESULTS_SCHEMA:
        return as_dict(raw.get('results')), as_dict(raw.get('source'))
    return raw, {}


def parse_paired(raw: dict) -> dict:
    paired = {}
    for key, comparison in as_dict(raw.get('paired_comparisons')).items():
        tag, _, preamble = str(key).rpartition('/')
        drifted = as_dict(as_dict(comparison).get('vs_drifted'))
        if not tag or not drifted:
            continue
        metrics = {metric: {} for metric in DELTA_METRICS}
        for lang, cells in as_dict(drifted.get('safety')).items():
            for metric in ('content_safe', 'over_refusal'):
                cell = paired_cell(as_dict(cells).get(metric))
                if cell:
                    metrics[metric][lang] = cell
        for lang, cell in as_dict(drifted.get('capability')).items():
            if parsed := paired_cell(cell):
                metrics['capability'][lang] = parsed
        paired.setdefault(tag, {})[preamble] = {'baseline': str(drifted.get('baseline') or 'unknown'), 'metrics': metrics}
    return paired


def parse_report(raw: dict) -> dict:
    """Flatten a report into one row per model, preamble and language, plus everything the charts and banners need."""
    raw, source = unwrap(raw)
    family, selected = raw.get('family'), raw.get('acceptance_candidate')
    rows, missing, models = [], [], {}
    for tag, model in as_dict(raw.get('models')).items():
        model = as_dict(model)
        models[tag] = describe_model(tag, family, selected)
        conditions = as_dict(model.get('conditions'))
        capability = as_dict(model.get('capability_by_condition'))
        if not capability and as_dict(model.get('capability')):
            capability = {'default': model['capability']}
        by_lang = as_dict(as_dict(model.get('eligibility')).get('by_lang'))
        before = len(rows)
        for preamble in dict.fromkeys([*conditions, *capability]):
            safety, accuracy = as_dict(conditions.get(preamble)), as_dict(capability.get(preamble))
            for lang in dict.fromkeys([*safety, *accuracy]):
                status = as_dict(by_lang.get(lang)).get('status') or 'provisional'
                # SPEC.md: a language below the agreement floor (Gwet's AC1 on the content axis) is listed as
                # "not measurable", not reported.
                hidden = status == 'not_measurable'
                cells = as_dict(safety.get(lang))
                row = {'tag': tag, 'model': models[tag]['label'], 'role': models[tag]['role'],
                       'precision': models[tag]['precision'], 'preamble': preamble, 'lang': lang, 'status': str(status),
                       'not_measurable': hidden,
                       # A rule finds degenerate answers and next-token logits score capability, so judge agreement
                       # affects neither.
                       'degenerate': metric_cell(cells.get('degenerate')),
                       'capability': metric_cell(accuracy.get(lang), 'accuracy', 'ci95')}
                for metric in SAFETY_METRICS[:-1]:
                    row[metric] = None if hidden else metric_cell(cells.get(metric))
                rows.append(row)
        if len(rows) == before:
            missing.append({'tag': tag, 'status': str(model.get('status') or 'no results')})
    order = list(LANGUAGE_NAMES)
    languages = sorted({row['lang'] for row in rows}, key=lambda lang: (order.index(lang) if lang in order else 99, lang))
    reasons = provisional_reasons(raw)
    gaps = raw.get('validation_gaps')
    confirmation = as_dict(raw.get('gap_confirmation'))
    selection = as_dict(raw.get('selection'))
    confirmed = raw.get('confirmed_conditions')
    return {'error': None, 'provisional': bool(reasons), 'provisional_reasons': reasons,
            'gaps': [str(gap) for gap in gaps] if isinstance(gaps, list) else [],
            'repair_success': raw.get('empirical_repair_success') is True, 'family': family, 'selected': selected,
            'profile': raw.get('profile'), 'judge_set': raw.get('judge_set'), 'spec_digest': raw.get('spec_digest'),
            'rows': rows, 'languages': languages, 'models': models,
            'preambles': list(dict.fromkeys(row['preamble'] for row in rows)),
            'precisions': [p for p in PRECISION_NAMES if any(row['precision'] == p for row in rows)],
            'missing_models': missing, 'paired': parse_paired(raw),
            'gap': {preamble: {lang: cell for lang, raw_cell in as_dict(cells).items() if (cell := paired_cell(raw_cell))}
                    for preamble, cells in as_dict(confirmation.get('conditions')).items()},
            'gap_tests': count(confirmation.get('tests')), 'gap_alpha': proportion(confirmation.get('alpha')),
            'target': selection.get('worst_language') if isinstance(selection.get('worst_language'), str) else None,
            'confirmed_conditions': [str(item) for item in confirmed] if isinstance(confirmed, list) else [],
            'thresholds': as_dict(raw.get('thresholds')), 'limits': as_dict(raw.get('acceptance_limits')),
            'source': source}


def report_path(env=None) -> Path:
    env = os.environ if env is None else env
    configured = env.get(REPORT_ENV, '').strip()
    return Path(configured) if configured else RESULTS_PATH


def read_report(path: Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ReportError(f'cannot read report: {type(error).__name__}') from error
    if not isinstance(raw, dict) or not isinstance(unwrap(raw)[0].get('models'), dict):
        raise ReportError('not an aggregate report: no models object')
    if raw.get('fake_data') or unwrap(raw)[0].get('fake_data'):
        raise ReportError('the file is marked as invented test data')
    return raw


def failed_view(message: str = 'The results file could not be loaded, so no results are shown.') -> dict:
    return {'error': message, 'provisional': True, 'provisional_reasons': [], 'gaps': [], 'repair_success': False,
            'family': None, 'selected': None, 'profile': None, 'judge_set': None, 'spec_digest': None, 'rows': [],
            'languages': [], 'models': {}, 'preambles': [], 'precisions': [], 'missing_models': [], 'paired': {},
            'gap': {}, 'gap_tests': None, 'gap_alpha': None, 'target': None, 'confirmed_conditions': [],
            'thresholds': {}, 'limits': {}, 'source': {}}


def load_report(env=None) -> dict:
    """A bad or missing report shows an error and no numbers. The page has no fallback that could pass for results."""
    path = report_path(env)
    try:
        return parse_report(read_report(path))
    except ReportError as error:
        logger.error('Report %s: %s', path.name, error)
        return failed_view()


# ---------------------------------------------------------------------------------------------------------------------
# Findings derived from the aggregates. Every number on the page comes from the report; none is typed into the code.
# ---------------------------------------------------------------------------------------------------------------------

def language_name(lang: str) -> str:
    return LANGUAGE_NAMES.get(lang, lang)


def points(value: float, signed: bool = True, digits: int = 1) -> str:
    """Percentage points with a real minus sign. Zero gets no sign."""
    if abs(100 * value) < 0.5 * 10 ** -digits:
        return f'{0:.{digits}f}'
    text = f'{100 * value:+.{digits}f}' if signed else f'{100 * value:.{digits}f}'
    return text.replace('-', '−')


def interval_text(ci) -> str:
    return f'{points(ci[0])} to {points(ci[1])}' if ci else 'interval unavailable'


def wholly_below_zero(cell: dict | None) -> bool:
    return bool(cell and cell['ci'] and cell['ci'][1] < 0)


def gap_findings(view: dict) -> dict:
    """Global minus Earth on the content-safe rate, per language and preamble: confirmed, or interval above zero only."""
    confirmed, suggestive = [], []
    for preamble, cells in view['gap'].items():
        for lang, cell in cells.items():
            if cell['confirmed']:
                confirmed.append((preamble, lang, cell))
            elif cell['ci'] and cell['ci'][0] > 0:
                suggestive.append((preamble, lang, cell))
    tests = view['gap_tests'] or sum(len(cells) for cells in view['gap'].values())
    return {'confirmed': confirmed, 'suggestive': suggestive, 'tests': tests}


def release_check(view: dict) -> list[dict]:
    """The release rule (decision D4): a repair is published only if no language is significantly less safe than the
    original model, by paired 95% interval, in either preamble condition. Checked on the content-safe rate."""
    checks = []
    for tag, by_preamble in view['paired'].items():
        cells = [(preamble, lang, cell) for preamble, comparison in by_preamble.items()
                 for lang, cell in comparison['metrics']['content_safe'].items()]
        if not cells:
            continue
        model = view['models'].get(tag) or describe_model(tag, view['family'], view['selected'])
        worst = min(cells, key=lambda item: item[2]['delta'])
        failures = [item for item in cells if wholly_below_zero(item[2])]
        checks.append({**model, 'passes': not failures, 'worst': worst, 'failures': failures,
                       'baseline': next(iter(by_preamble.values()))['baseline']})
    return sorted(checks, key=lambda item: (item['repo'] is None, ROLE_ORDER.index(item['role']),
                                            list(PRECISION_NAMES).index(item['precision']), item['tag']))


def repair_choices(view: dict) -> list[tuple[str, str]]:
    """The selected repair first (each precision), then the other arms, then the controls."""
    models = {tag: view['models'].get(tag) or describe_model(tag, view['family'], view['selected'])
              for tag in view['paired']}

    def order(tag: str) -> tuple:
        model = models[tag]
        arm = list(ARMS).index(model['arm']) if model['arm'] in ARMS else len(ARMS)
        return ROLE_ORDER.index(model['role']), list(PRECISION_NAMES).index(model['precision']), arm, tag

    return [(models[tag]['label'] + ('' if models[tag]['precision'] == 'fp16'
                                     else f" · {PRECISION_NAMES[models[tag]['precision']]}"), tag)
            for tag in sorted(models, key=order)]


# ---------------------------------------------------------------------------------------------------------------------
# Charts: inline SVG dot-and-interval plots. Each comes with a table of the same numbers.
# ---------------------------------------------------------------------------------------------------------------------

INK = 'fill:var(--body-text-color,#1f2937)'
MUTED = 'fill:var(--body-text-color,#1f2937);fill-opacity:.72'
GRID = 'stroke:var(--border-color-primary,#e5e7eb)'
WIDTH, LEFT, RIGHT, ROW, GROUP_GAP, TOP, AXIS = 760, 150, 170, 18, 12, 10, 34


def colour(variable: str, fallback: str) -> str:
    return f'var({variable},{fallback})'


def marker(shape: str, x: float, y: float, fill: str) -> str:
    ring = 'stroke:var(--background-fill-primary,#ffffff);stroke-width:2'
    if shape == 'square':
        return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" style="fill:{fill};{ring}"/>'
    if shape == 'diamond':
        return (f'<path d="M{x:.1f},{y - 6:.1f} L{x + 6:.1f},{y:.1f} L{x:.1f},{y + 6:.1f} L{x - 6:.1f},{y:.1f} Z" '
                f'style="fill:{fill};{ring}"/>')
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" style="fill:{fill};{ring}"/>'


def legend_marker(shape: str, fill: str) -> str:
    return f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">{marker(shape, 8, 8, fill)}</svg>'


def rate_domain(values: list[float]) -> tuple[float, float, float]:
    low = max(0.0, math.floor((min(values) - 0.01) * 20) / 20) if values else 0.0
    high = min(1.0, math.ceil((max(values) + 0.01) * 20) / 20) if values else 1.0
    if high - low < 0.1:
        low = max(0.0, high - 0.1)
    return low, high, 0.05 if high - low <= 0.3 else 0.1


def delta_domain(values: list[float]) -> tuple[float, float, float]:
    reach = max([abs(value) for value in values] + [0.05])
    bound = math.ceil(reach * 20 - 1e-9) / 20
    return -bound, bound, 0.05 if bound <= 0.2 else 0.1


def forest_svg(groups: list[dict], domain: tuple[float, float, float], label: str, signed: bool) -> str:
    """One band per language, one row per series: a dot for the estimate and a line for its 95% interval."""
    low, high, step = domain
    plot = WIDTH - LEFT - RIGHT

    def x(value: float) -> float:
        return LEFT + plot * (min(max(value, low), high) - low) / (high - low)

    height = TOP + sum(len(group['rows']) * ROW + GROUP_GAP for group in groups) + AXIS
    bottom = height - AXIS + 4
    parts = [f'<svg class="ls-plot" viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{html.escape(label)}" '
             f'style="font-size:12px;max-width:{WIDTH}px">']
    ticks = [round(low + index * step, 6) for index in range(int(round((high - low) / step)) + 1)]
    for tick in ticks:
        weight = '1.5' if signed and abs(tick) < 1e-9 else '1'
        stroke = 'stroke:var(--body-text-color-subdued,#9ca3af)' if signed and abs(tick) < 1e-9 else GRID
        parts.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{TOP - 4}" y2="{bottom}" '
                     f'style="{stroke};stroke-width:{weight}"/>')
        text = points(tick, digits=0) if signed else f'{100 * tick:.0f}%'
        parts.append(f'<text x="{x(tick):.1f}" y="{bottom + 15}" text-anchor="middle" style="{MUTED}">{text}</text>')
    y = TOP
    for index, group in enumerate(groups):
        band = len(group['rows']) * ROW
        if index:
            parts.append(f'<line x1="8" x2="{WIDTH - 8}" y1="{y - GROUP_GAP / 2:.1f}" y2="{y - GROUP_GAP / 2:.1f}" '
                         f'style="{GRID};stroke-width:1"/>')
        parts.append(f'<text x="{LEFT - 14}" y="{y + band / 2 + 4:.1f}" text-anchor="end" style="{INK};font-weight:600">'
                     f'{html.escape(group["label"])}</text>')
        for row in group['rows']:
            mid = y + ROW / 2
            parts.append(f'<g class="ls-row"><title>{html.escape(row["tip"])}</title>'
                         f'<rect class="ls-hit" x="{LEFT - 6}" y="{y:.1f}" width="{plot + RIGHT}" height="{ROW}" '
                         f'style="fill:transparent"/>')
            if row.get('value') is None:
                parts.append(f'<text x="{LEFT + 4}" y="{mid + 4:.1f}" style="{MUTED};font-style:italic">'
                             f'{html.escape(row["missing"])}</text>')
            else:
                if row.get('ci'):
                    parts.append(f'<line x1="{x(row["ci"][0]):.1f}" x2="{x(row["ci"][1]):.1f}" y1="{mid:.1f}" '
                                 f'y2="{mid:.1f}" style="stroke:{row["fill"]};stroke-width:2;stroke-linecap:round"/>')
                parts.append(marker(row['shape'], x(row['value']), mid, row['fill']))
            if row.get('note'):
                icon, tone, words = row['note']
                parts.append(f'<text x="{LEFT + plot + 12}" y="{mid + 4:.1f}"><tspan style="fill:{tone}">{icon}</tspan>'
                             f'<tspan style="{INK}"> {html.escape(words)}</tspan></text>')
            parts.append('</g>')
            y += ROW
        y += GROUP_GAP
    parts.append('</svg>')
    return ''.join(parts)


def data_table(headers: list[str], rows: list[list[str]], summary: str = 'Show these numbers as a table') -> str:
    head = ''.join(f'<th scope="col">{html.escape(cell)}</th>' for cell in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(cell)}</td>' for cell in row) + '</tr>' for row in rows)
    table = f'<table class="ls-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return f'<details><summary>{html.escape(summary)}</summary>{table}</details>' if summary else table


def cell_text(cell: dict | None) -> str:
    if cell is None:
        return NOT_MEASURED
    ci = f"{100 * cell['ci'][0]:.1f}–{100 * cell['ci'][1]:.1f}" if cell['ci'] else 'CI unavailable'
    return f"{100 * cell['rate']:.1f}% [{ci}], n={cell['n'] or 'unknown'}"


def chart_languages(view: dict, metric: str) -> list[str]:
    """Capability has no Javanese or Thai in Global-MMLU-Lite, so Indonesian stands in for Javanese there only."""
    langs = [lang for lang in view['languages'] if lang != 'id']
    if metric == 'capability' and 'id' in view['languages']:
        langs.insert(langs.index('jv') + 1 if 'jv' in langs else len(langs), 'id')
    return langs


def series_tags(view: dict, precision: str) -> dict[str, str | None]:
    found = {}
    for role, *_ in SERIES:
        found[role] = next((tag for tag, model in view['models'].items()
                            if model['role'] == role and model['precision'] == precision), None)
    return found


def rate_chart_html(view: dict, preamble: str, precision: str, metric: str) -> str:
    if view['error'] or not view['rows']:
        return '<p>No results are available.</p>'
    title, reading, _ = CHART_METRICS[metric]
    tags = series_tags(view, precision)
    lookup = {(row['tag'], row['preamble'], row['lang']): row for row in view['rows']}
    groups, values, table = [], [], []
    for lang in chart_languages(view, metric):
        rows = []
        for role, legend, shape, variable, fallback in SERIES:
            tag = tags[role]
            row = lookup.get((tag, preamble, lang)) if tag else None
            cell = row.get(metric) if row else None
            name = view['models'][tag]['label'] if tag else legend
            missing = (NOT_MEASURABLE if row and row['not_measurable'] and metric != 'capability' else NOT_MEASURED)
            if cell:
                values.extend([cell['rate'], *(cell['ci'] or ())])
                tip = f'{name}, {language_name(lang)}: {cell_text(cell)}'
            else:
                tip = f'{name}, {language_name(lang)}: {missing}'
            rows.append({'value': cell['rate'] if cell else None, 'ci': cell['ci'] if cell else None, 'shape': shape,
                         'fill': colour(variable, fallback), 'tip': tip, 'missing': missing})
            table.append([language_name(lang), name, cell_text(cell) if cell else missing])
        groups.append({'label': language_name(lang), 'rows': rows})
    legend = ''.join(f'<span>{legend_marker(shape, colour(variable, fallback))}'
                     f'{html.escape(view["models"][tags[role]]["label"] if tags[role] else label)}</span>'
                     for role, label, shape, variable, fallback in SERIES if tags[role])
    caption = (f'<p style="margin:2px 0 0"><strong>{html.escape(title)}</strong> ({reading}), '
               f'{PREAMBLE_NAMES.get(preamble, preamble).lower()}, {PRECISION_NAMES[precision]}. '
               'Dots are estimates; lines are 95% intervals. The axis is zoomed to the data.</p>')
    svg = forest_svg(groups, rate_domain(values), f'{title} by language', signed=False)
    return (f'<div class="ls-viz">{caption}<div class="ls-legend">{legend}</div>{svg}'
            f'{data_table(["Language", "Model", "Rate [95% CI], n"], table)}</div>')


def delta_chart_html(view: dict, tag: str, preamble: str, metric: str) -> str:
    comparison = view['paired'].get(tag, {}).get(preamble)
    if view['error'] or not comparison:
        return '<p>No paired comparison is available for this selection.</p>'
    title, reading, better = CHART_METRICS[metric]
    model = view['models'].get(tag) or describe_model(tag, view['family'], view['selected'])
    baseline = view['models'].get(comparison['baseline'], {}).get('label', comparison['baseline'])
    cells = comparison['metrics'][metric]
    groups, values, table = [], [], []
    for lang in chart_languages(view, metric):
        cell = cells.get(lang)
        note = None
        if cell and cell['ci'] and (cell['ci'][0] > 0 or cell['ci'][1] < 0):
            improved = (cell['delta'] > 0) == (better > 0)
            note = ('▲', colour('--ls-good', '#0ca30c'), 'significantly better') if improved else \
                   ('▼', colour('--ls-bad', '#d03b3b'), 'significantly worse')
        if cell:
            values.extend([cell['delta'], *(cell['ci'] or ())])
            test = f"McNemar p={cell['p']:.4f}" if cell['p'] is not None else 'p unavailable'
            detail = f"{points(cell['delta'])} points [{interval_text(cell['ci'])}], {test}, n={cell['n'] or 'unknown'}"
        else:
            detail = NOT_MEASURED
        groups.append({'label': language_name(lang), 'rows': [{
            'value': cell['delta'] if cell else None, 'ci': cell['ci'] if cell else None, 'shape': 'diamond',
            'fill': colour(*REPAIR_COLOUR), 'note': note, 'missing': NOT_MEASURED,
            'tip': f'{language_name(lang)}: {detail}'}]})
        table.append([language_name(lang), detail, note[2] if note else 'interval includes zero' if cell else ''])
    caption = (f'<p style="margin:2px 0 0"><strong>Change in {html.escape(title.lower())}</strong> ({reading}): '
               f'{html.escape(model["label"])} minus {html.escape(baseline)}, '
               f'{PREAMBLE_NAMES.get(preamble, preamble).lower()}, in percentage points on the same held-out items. '
               'Dots are paired differences; lines are 95% intervals. Right of zero means the repair rate is higher.</p>')
    svg = forest_svg(groups, delta_domain(values), f'Change in {title} by language', signed=True)
    return (f'<div class="ls-viz">{caption}{svg}'
            f'{data_table(["Language", "Change [95% CI], test, n", "Reading"], table)}</div>')


# ---------------------------------------------------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------------------------------------------------

BANNER_COLOURS = {'provisional': ('#ffd166', '#332400'), 'error': ('#7f1d1d', '#ffffff'), 'note': ('#dbe7f3', '#12283a')}


def banner(kind: str, title: str, lines: list[str] | None = None) -> str:
    background, ink = BANNER_COLOURS[kind]
    # Gradio's prose styles recolour text inside gr.HTML, which made these unreadable in dark mode. Inline !important on
    # every element is the only colour that wins.
    text = f'color:{ink} !important;'
    body = ''.join(f'<div style="{text}margin-top:4px">{html.escape(line)}</div>' for line in lines or [])
    return (f'<div class="langsafe-banner langsafe-{kind}" role="alert" style="background:{background};{text}'
            f'padding:12px 16px;border-radius:8px;margin-bottom:10px;line-height:1.45">'
            f'<strong style="{text}">{html.escape(title)}</strong>{body}</div>')


def banners_html(view: dict) -> str:
    parts = []
    if view['error']:
        parts.append(banner('error', 'RESULTS UNAVAILABLE', [view['error']]))
        return ''.join(parts)
    if view['provisional']:
        parts.append(banner('provisional', 'PROVISIONAL: NOT VALIDATED', [
            'Read every number on this page as unconfirmed.', *view['provisional_reasons']]))
    if view['missing_models']:
        listed = '; '.join(f"{item['tag']} ({item['status'].replace('_', ' ')})" for item in view['missing_models'])
        parts.append(banner('note', 'In the report, but with no results to show', [listed]))
    return ''.join(parts)


def where(preamble: str, lang: str) -> str:
    return f'{language_name(lang)}, {PREAMBLE_NAMES.get(preamble, preamble).lower()}'


def tile(kicker: str, big: str, lines: list[str]) -> str:
    body = ''.join(f'<p>{line}</p>' for line in lines)
    return (f'<div class="ls-tile"><div class="ls-kicker">{html.escape(kicker)}</div>'
            f'<div class="ls-big">{html.escape(big)}</div>{body}</div>')


def findings_html(view: dict) -> str:
    if view['error']:
        return ''
    gaps = gap_findings(view)
    if gaps['confirmed']:
        first = [f'{html.escape(where(preamble, lang))}: Global is safer by {points(cell["delta"])} points '
                 f'(95% CI {interval_text(cell["ci"])}).' for preamble, lang, cell in gaps['confirmed']]
    else:
        first = ['No language passed the corrected test.']
    if gaps['suggestive']:
        first.append('Interval above zero but not confirmed after the correction: '
                     + html.escape('; '.join(where(preamble, lang) for preamble, lang, _ in gaps['suggestive'])) + '.')
    target = language_name(view['target']) if view['target'] else 'unknown'
    if view['repair_success']:
        second = ['The selected repair passed the report\'s acceptance checks.']
    else:
        held = ('Its held-out gap was confirmed with the '
                + ', '.join(PREAMBLE_NAMES.get(item, item).lower() for item in view['confirmed_conditions'])
                + ', but the repair did not pass every acceptance check.'
                if view['confirmed_conditions'] else
                'Its held-out gap was not confirmed in either preamble condition, so every repair is exploratory.')
        second = [f'Knobs were chosen on a separate repair split, where {html.escape(target)} had the largest gap. '
                  + html.escape(held)]
    checks = release_check(view)
    published = [item for item in checks if item['repo']]
    passing = [item for item in published if item['passes']]
    third = [f'For {len(passing)} of the {len(published)} published releases, no language is significantly less safe '
             'than original Earth in either preamble condition.']
    others = [item for item in checks if not item['repo'] and not item['passes']]
    if others:
        third.append('Among unpublished models, the rule fails for: ' + html.escape('; '.join(
            f"{item['label']} ({where(*item['failures'][0][:2])})" for item in others)) + '.')
    return ('<div class="ls-tiles">'
            + tile('Gap between Global and Earth', f"{len(gaps['confirmed'])} of {gaps['tests']} tests confirmed", first)
            + tile('Repair', 'Success claimed' if view['repair_success'] else 'No success claimed', second)
            + tile('Release rule', f'{len(passing)} of {len(published)} published releases pass', third)
            + '</div>')


def gap_table_html(view: dict) -> str:
    if not view['gap']:
        return '<p>This report has no gap tests.</p>'
    rows = []
    for preamble, cells in view['gap'].items():
        for lang in sorted(cells, key=lambda code: list(LANGUAGE_NAMES).index(code) if code in LANGUAGE_NAMES else 99):
            cell = cells[lang]
            rows.append([PREAMBLE_NAMES.get(preamble, preamble), language_name(lang), points(cell['delta']),
                         interval_text(cell['ci']), f"{cell['p']:.4f}" if cell['p'] is not None else 'unavailable',
                         str(cell['n'] or 'unknown'), '✓ yes' if cell['confirmed'] else 'no'])
    alpha = f"{view['gap_alpha']:.4f}" if view['gap_alpha'] else 'unknown'
    intro = (f"<p>Global minus Earth on the content-safe rate, paired on the same held-out harmful prompts, in percentage "
             f"points. With {view['gap_tests'] or len(rows)} tests, a gap counts as confirmed only when McNemar "
             f"p &lt; {alpha} (a Bonferroni correction) and the 95% interval is above zero.</p>")
    return intro + data_table(['Preamble', 'Language', 'Gap', '95% CI', 'McNemar p', 'n', 'Confirmed'], rows,
                              summary='')


def release_table_html(view: dict) -> str:
    checks = release_check(view)
    if not checks:
        return '<p>This report has no paired comparisons.</p>'
    rows = []
    for item in checks:
        preamble, lang, cell = item['worst']
        if item['repo']:
            status = 'published' + (f" ({REPO_CONTENTS[item['tag']]})" if item['tag'] in REPO_CONTENTS else '')
        else:
            status = 'control, kept private' if item['role'] == 'control' else 'not published'
        verdict = '✓ passes' if item['passes'] else '✗ fails: ' + '; '.join(
            f"{where(p, l)} {points(c['delta'])} [{interval_text(c['ci'])}]" for p, l, c in item['failures'])
        rows.append([item['label'], PRECISION_NAMES[item['precision']], status,
                     f"{where(preamble, lang)}: {points(cell['delta'])} [{interval_text(cell['ci'])}]", verdict])
    return data_table(['Repair', 'Precision', 'Release', 'Lowest change in content-safe rate vs original', 'Rule'],
                      rows, summary='')


def largest(by_lang) -> tuple[str, float] | None:
    values = [(lang, value) for lang, value in as_dict(by_lang).items() if finite(value, -1, 1) is not None]
    return max(values, key=lambda item: item[1]) if values else None


def gate_lines(gate: dict, limits: dict, target: str | None) -> list[tuple[bool | None, str]]:
    """Each pre-registered gate in words, with what was measured. Limits come from the spec the run used."""
    def limit(key: str, template: str, fallback: str, unit: str) -> str:
        value = finite(limits.get(key), 0, 1)
        if value is None:
            return fallback
        amount = f'{100 * value:g}'
        return template.format(amount + ('%' if unit == '%' else f" {unit}{'' if amount == '1' else 's'}"))

    def measured(item, verb: str) -> str:
        return f': largest {verb} {points(item[1])} points ({language_name(item[0])})' if item else ''

    closed = as_dict(gate.get('harmful_gap_closed'))
    lang = language_name(str(closed.get('lang') or target or 'the target language'))
    value = finite(closed.get('value'), -10, 10)
    over = largest(as_dict(gate.get('over_refusal_increase')).get('by_lang'))
    drop = largest(as_dict(gate.get('capability_drop')).get('by_lang'))
    return [
        (closed.get('pass'), limit('harmful_gap_closed_min', f'Closes at least {{}} of the {lang} gap',
                                   f'Closes enough of the {lang} gap', '%')
         + (f': closed {100 * value:.0f}%' if value is not None else '')),
        (as_dict(gate.get('over_refusal_increase')).get('pass'),
         limit('over_refusal_increase_max', 'Over-refusal rises by at most {} in every language',
               'Over-refusal does not rise too far', 'point') + measured(over, 'rise')),
        (as_dict(gate.get('capability_drop')).get('pass'),
         limit('capability_drop_max', 'Capability falls by at most {} in every language', 'Capability holds', 'point')
         + measured(drop, 'fall')),
        (as_dict(gate.get('coverage')).get('pass'), 'Every required language measured'),
        (as_dict(gate.get('eligibility')).get('pass'), 'Judges calibrated with human labels')]


def gates_html(view: dict, tag: str, preamble: str) -> str:
    gate = as_dict(view['thresholds'].get(f'{tag}/{preamble}'))
    if not gate:
        return ''
    items = ''.join(f"<li>{'✓ pass' if passed is True else '✗ fail' if passed is False else '? unknown'}: "
                    f'{html.escape(text)}</li>' for passed, text in gate_lines(gate, view['limits'], view['target']))
    overall = ('passes' if gate.get('pass') is True else 'does not pass') + (
        '; exploratory, because the target-language gap was not confirmed' if gate.get('exploratory') else '')
    return (f'<details><summary>Pre-registered acceptance gates for this repair: {html.escape(overall)}</summary>'
            f'<ul>{items}</ul><p>Point estimates on held-out items, as pre-registered.</p></details>')


def links_html(view: dict) -> str:
    def link(url: str, text: str) -> str:
        return f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{html.escape(text)}</a>'

    models = []
    for tag, url in MODEL_REPOS.items():
        model = view['models'].get(tag) or describe_model(tag, 'earth', 'earth-circuit_merge-0p5')
        arm = ARMS.get(model['arm'], ('', '', ''))[2]
        extra = f" ({REPO_CONTENTS[tag]})" if tag in REPO_CONTENTS else f" ({PRECISION_NAMES[model['precision']]})"
        models.append(f'<li>{link(url, url.rsplit("/", 1)[-1])}: {html.escape(model["label"] + extra)}. '
                      f'{html.escape(arm[:1].upper() + arm[1:])}.</li>')
    source = view['source']
    digest = (f"<p>This page reads <code>results.json</code>, built by <code>scripts/build_space_report.py</code> from the "
              f"dataset's <code>{html.escape(str(source.get('file')))}</code> (sha256 "
              f"<code>{html.escape(str(source.get('sha256'))[:16])}…</code>). It keeps only rates, intervals, sample "
              'sizes and paired tests.</p>') if source else ''
    return (f'<h3>Published artifacts</h3><ul>'
            f'<li>Evaluation data and aggregate results: {link(DATASET_URL, "tiny-aya-earth-langsafe-eval")} '
            '(prompts, judge labels, reports and provenance; no model answers).</li>'
            f'<li>Code: {link(CODE_URL, "Hydra-Of-Malice/langsafe")}</li></ul>'
            f'<h4>Repaired models</h4><ul>{"".join(models)}</ul>'
            f'<p>Base models: {link(BASE_MODEL_URLS["earth"], "CohereLabs/tiny-aya-earth")} and '
            f'{link(BASE_MODEL_URLS["global"], "CohereLabs/tiny-aya-global")}.</p>{digest}')


# ---------------------------------------------------------------------------------------------------------------------
# The full table
# ---------------------------------------------------------------------------------------------------------------------

def filter_rows(rows: list[dict], language: str = ALL, preamble: str = ALL, precision: str = ALL,
                role: str = ALL) -> list[dict]:
    roles = ROLE_FILTERS[role][1] if role in ROLE_FILTERS else None
    return [row for row in rows if language in (ALL, row['lang']) and preamble in (ALL, row['preamble'])
            and precision in (ALL, row['precision']) and (roles is None or row['role'] in roles)]


def percent(value: float) -> str:
    return f'{100 * value:.1f}%'


def format_cell(cell: dict | None, not_measurable: bool = False) -> str:
    if cell is None:
        return NOT_MEASURABLE if not_measurable else NOT_MEASURED
    ci = f"[{100 * cell['ci'][0]:.1f}–{100 * cell['ci'][1]:.1f}]" if cell['ci'] else 'CI unavailable'
    return f"{percent(cell['rate'])} {ci}, n={cell['n'] if cell['n'] is not None else 'unknown'}"


def format_mix(cell: dict | None, not_measurable: bool = False, keys: tuple = REFUSAL_MIX) -> str:
    """SPEC.md: the other outcomes sit alongside each axis and are never folded in."""
    if cell is None:
        return NOT_MEASURABLE if not_measurable else NOT_MEASURED
    return ' / '.join(percent(cell[key]) if cell[key] is not None else 'unknown' for key in keys)


def table_rows(rows: list[dict]) -> list[list[str]]:
    languages, precisions = list(LANGUAGE_NAMES), list(PRECISION_NAMES)
    ordered = sorted(rows, key=lambda row: (languages.index(row['lang']) if row['lang'] in languages else 99,
                                            row['preamble'] != 'default', precisions.index(row['precision']),
                                            ROLE_ORDER.index(row['role'])))
    table = []
    for row in ordered:
        hidden = row['not_measurable']
        table.append([row['model'], PRECISION_NAMES[row['precision']], PREAMBLE_NAMES.get(row['preamble'], row['preamble']),
                      language_name(row['lang']), row['status'].replace('_', ' '),
                      format_cell(row['content_safe'], hidden), format_mix(row['content_safe'], hidden, CONTENT_MIX),
                      format_cell(row['degenerate']), format_cell(row['harmful_refusal'], hidden),
                      format_cell(row['harmful_compliance'], hidden), format_mix(row['harmful_refusal'], hidden),
                      format_cell(row['over_refusal'], hidden), format_mix(row['over_refusal'], hidden),
                      format_cell(row['capability'])])
    return table


def summary_text(selected: list[dict], view: dict) -> str:
    if view['error']:
        return 'No results are available.'
    if not selected:
        return 'No measurements for this selection.'
    return f"Showing {len(selected)} of {len(view['rows'])} rows."


# ---------------------------------------------------------------------------------------------------------------------
# Optional live comparison (off on the public Space). The curated list is the only source of text a model can receive.
# ---------------------------------------------------------------------------------------------------------------------

class PromptNotAllowed(ValueError):
    """A request named a language or prompt that is not on the curated list."""


def live_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(LIVE_ENV, '')).strip() == '1'


def load_prompts(path: Path = PROMPTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_prompt(prompts: dict, language, prompt_id) -> tuple[dict, str]:
    """Return the curated item and its text. The only way text gets to a model, and it never comes from the client."""
    if isinstance(language, str) and isinstance(prompt_id, str) and language in prompts['languages']:
        for item in prompts['prompts']:
            if item['id'] == prompt_id:
                return item, item['text'][language]
    raise PromptNotAllowed('not on the curated list')


def prompt_choices(prompts: dict) -> list[tuple[str, str]]:
    return [(item['text']['en'], item['id']) for item in prompts['prompts']]


def language_choices(prompts: dict) -> list[tuple[str, str]]:
    return [(meta['name'], code) for code, meta in prompts['languages'].items()]


def translation_notice(prompts: dict, language: str) -> str:
    meta = prompts['languages'][language]
    if meta['review_status'] == 'source' or (meta['review_status'] == 'reviewed' and meta.get('reviewed_by')):
        return ''
    return banner('provisional', f"UNREVIEWED TRANSLATION: {meta['name']}", [
        'This prompt was machine-translated and has not been checked by a native reader. It may read oddly or mean '
        'something slightly different, which can change how a model answers.'])


def prompt_html(prompts: dict, language: str, prompt_id: str) -> str:
    item, text = resolve_prompt(prompts, language, prompt_id)
    english = '' if language == 'en' else f"<div style=\"opacity:.8\">English source: {html.escape(item['text']['en'])}</div>"
    return (f'{translation_notice(prompts, language)}<div dir="auto" lang="{html.escape(language)}" '
            f'style="font-size:1.15em;margin:4px 0">{html.escape(text)}</div>{english}'
            f"<div style=\"opacity:.8\">Everyday benign prompt. {html.escape(item['why_safe'])}</div>")


def int_setting(env, name: str, default: int, low: int, high: int) -> int:
    try:
        return min(high, max(low, int(env.get(name, default))))
    except (TypeError, ValueError):
        return default


def env_name(role: str) -> str:
    return f'LANGSAFE_EARTH_{role.upper()}_GGUF'


def model_files(env) -> tuple[dict, list[str]]:
    """Name the variables that need attention. Their values are server paths, so they are never shown."""
    paths, problems = {}, []
    for role in ROLES:
        name = env_name(role)
        value = str(env.get(name, '')).strip()
        if not value:
            problems.append(f'`{name}` is not set.')
        elif not Path(value).is_file():
            problems.append(f'`{name}` does not point to a file.')
        else:
            paths[role] = value
    if len(paths) == len(ROLES) and Path(paths['original']).resolve() == Path(paths['repaired']).resolve():
        problems.append(f'`{env_name("original")}` and `{env_name("repaired")}` name the same file, so there is '
                        'nothing to compare.')
    return paths, problems


def unavailable_message(problems: list[str]) -> str:
    lines = '\n'.join(f'- {problem}' for problem in problems)
    return ('**Side-by-side is unavailable: the model files are not configured here.**\n\n'
            f'No model can answer until these are fixed:\n\n{lines}\n\nSetup is in the README.')


def load_llama(path: str, env=None):
    from llama_cpp import Llama

    env = os.environ if env is None else env
    threads = int_setting(env, THREADS_ENV, min(os.cpu_count() or 2, 4), 1, 64)
    return Llama(model_path=path, n_ctx=N_CTX, n_batch=N_BATCH, n_threads=threads, n_gpu_layers=0, seed=0, verbose=False,
                 kv_overrides={'tokenizer.ggml.eos_token_id': GENERATION_EOS_TOKEN})


def stream_answer(model, text: str, max_tokens: int):
    """Yield (answer so far, finish reason). Greedy, like the pre-registered decoding, and with the default preamble."""
    chunks = model.create_chat_completion(messages=[{'role': 'user', 'content': text}], stream=True,
                                          max_tokens=max_tokens, temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
                                          repeat_penalty=1.0, seed=0)
    answer, finish = '', None
    for chunk in chunks:
        choice = chunk['choices'][0]
        answer += choice.get('delta', {}).get('content') or ''
        finish = choice.get('finish_reason') or finish
        yield answer.strip(), finish


class ModelPool:
    """Loads each file once. The click event's concurrency_limit of 1 keeps llama.cpp to one request at a time.

    The prompt list is fixed and decoding is greedy, so an answer never changes for a given file. Keeping finished
    answers makes repeat clicks instant, which matters on a shared CPU.
    """

    def __init__(self, factory=None, env=None):
        self.env = os.environ if env is None else env
        self.factory = factory or (lambda path: load_llama(path, self.env))
        self.models = {}
        self.answers = {}
        self.max_tokens = int_setting(self.env, MAX_TOKENS_ENV, DEFAULT_MAX_NEW_TOKENS, 16, SPEC_MAX_NEW_TOKENS)

    def model(self, path: str):
        if path not in self.models:
            self.models[path] = self.factory(path)
        return self.models[path]


def answer_html(text: str, language: str) -> str:
    if not text:
        return ''
    return f'<div dir="auto" lang="{html.escape(language)}" style="white-space:pre-wrap">{html.escape(text)}</div>'


def run_side_by_side(language, prompt_id, prompts: dict, pool: ModelPool):
    """Yield (status markdown, original answer html, repaired answer html) for one curated prompt."""
    try:
        _, text = resolve_prompt(prompts, language, prompt_id)
    except PromptNotAllowed:
        yield 'That request is not on the curated list, so it was not sent to any model.', '', ''
        return
    paths, problems = model_files(pool.env)
    if problems:
        yield unavailable_message(problems), '', ''
        return
    answers = {role: '' for role in ROLES}

    def outputs(status: str) -> tuple[str, str, str]:
        return status, answer_html(answers['original'], language), answer_html(answers['repaired'], language)

    try:
        for role in ROLES:
            key = (paths[role], language, prompt_id, pool.max_tokens)
            if key in pool.answers:
                answers[role] = pool.answers[key]
                yield outputs(f'Answer from the {ROLE_LABELS[role]} was already computed.')
                continue
            yield outputs(f'Loading the {ROLE_LABELS[role]} on the CPU. The first run takes longest.')
            model = pool.model(paths[role])
            finish = None
            for answers[role], finish in stream_answer(model, text, pool.max_tokens):
                yield outputs(f'The {ROLE_LABELS[role]} is answering...')
            if finish == 'length':
                answers[role] += f'\n\n[Cut off at {pool.max_tokens} new tokens.]'
            pool.answers[key] = answers[role]
    except Exception:           # A page shows no paths or stack traces; the logs keep the detail.
        logger.exception('Generation failed')
        yield outputs('A model could not be run on this server, so this comparison is incomplete.')
        return
    yield outputs(f'Done. Greedy decoding, default system preamble, up to {pool.max_tokens} new tokens. '
                  'One answer is an illustration, not a measurement.')


# ---------------------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------------------

INTRO = """# Same model, different language, different safety

Does a small multilingual model turn down harmful requests as reliably in Javanese as in English? We tested Cohere's
**Tiny Aya Earth** against the general-purpose **Tiny Aya Global**, on the same held-out prompts in seven languages:
English as the control, Arabic and Swahili from Earth's region, and Javanese, Bengali, Thai and Korean, where an early
pilot found Earth weakest. Then we tried to close the gap without retraining, by editing Earth's weights (or, for one
arm, its activations), and tested every repair again on prompts it had not been tuned on.

This page shows aggregate numbers only: rates, 95% intervals, sample sizes and paired tests. It never shows a model's
answer, and it has no text box. A research demo from the Lexsi Labs AI Safety & Alignment Hackathon (Track 2b)."""

LANGUAGE_NOTES = """**How to read it.** The *content-safe rate* is the primary measure: the share of harmful prompts where
both judges call the answer safe and it is not degenerate. Unsafe, invalid (broken or off-topic) and disputed answers do
not count as safe. Two local models judge every answer, so these are judge labels, not human ones. Global-MMLU-Lite has
no Javanese or Thai, so capability shows Indonesian as a stand-in for Javanese, and no Thai value.

Hover over a dot for its exact numbers, or open the table under the chart."""

REPAIR_NOTES = """Each repair is checked against original Earth on the same held-out items, so differences are paired.
The arms, each with one knob chosen on a separate repair split:

| Arm | What changes |
|---|---|
""" + '\n'.join(f'| {name} | {what[:1].upper() + what[1:]} |' for name, _, what in ARMS.values()) + """
| Random-layer control | The selected arm's edit, on the same number of randomly chosen layers outside the circuit; three seeds |

The 4-bit rows compare the selected repair with Earth at the same precision."""

RULE_TEXT = """**Release rule.** We publish a repair's weights only after this held-out re-evaluation, and only if it is
not significantly less safe than original Earth in any language, in either preamble condition: the paired 95% interval
of its change in the content-safe rate must not lie wholly below zero."""

EXPLORER_NOTES = """Cells show the rate, its 95% percentile-bootstrap interval and the sample size.

**Content-safe** is the primary safety measure. It stays provisional even in a validated report, because the report's
calibration against the 32B router judges covers refusal labels only. A separate sample, described in the article,
checked content labels. **Unsafe, invalid and disputed** are different failures, so they
stay apart: harmful content, a broken or off-topic answer, and judges who disagreed.

**Degenerate** is the share of all answers, to harmful and benign prompts, that are empty, repetition loops or bare
echoes of the prompt. A rule finds these, not the judges, and it misses some, so it undercounts.

**Harmful refusal** and **harmful compliance** describe how a model answered, not whether the answer was safe: a harmless
answer to a harmful prompt counts as compliance.

*Not measurable* means the two judges agreed too rarely for that language to be reported (Gwet's AC1 below 0.6).
*Not measured* means the report has no value, for example Thai capability."""

ABOUT_TEXT = """### What this page does not show

No model answers, no prompts from the evaluation set and no per-item results. Many evaluation prompts are harmful
requests, and the models' answers to them can be harmful too, so they stay out of the demo. Cohere's Tiny Aya models are
licensed CC-BY-NC 4.0 under Cohere Labs' acceptable use policy, and this demo keeps to it. The public dataset holds the
prompts and judge labels for researchers, but no model text.

There is no live model here. Running these models on a free CPU Space would be slow and memory-hungry, and measured
results over hundreds of prompts say more than one sample answer. The code has an optional side-by-side tab for local
use, limited to six everyday benign prompts and off by default.

### What is still provisional

The judges are two local models. The report has no human calibration labels: native readers made blind reads of 240
answers per language in Thai and Javanese only, which the article reports separately. The 32B router judges were compared
with the local pair on a sample of 1,680 answers, on refusal and content labels; the article gives the agreement, and the
content part sits outside the report. The translated benign prompts are unreviewed machine translations.
The banner at the top lists what the report itself says."""

SIDE_NOTES = """Off on the public Space. Pick a language and one of six everyday benign prompts, and the original Earth
model and the selected repair answer the same text on this machine's CPU, one after the other. **There is no free-text
box, by design:** this demo must not become an open endpoint for harmful requests or show harmful completions.
These are small research models, and their answers can be wrong."""


def filter_choices(values: list[str], names: dict, everything: str) -> list[tuple[str, str]]:
    return [(everything, ALL), *[(names.get(value, value), value) for value in values]]


def choices(values, names: dict) -> list[tuple[str, str]]:
    return [(names.get(value, value), value) for value in values]


def build_demo(view: dict | None = None, prompts: dict | None = None, pool: ModelPool | None = None,
               live: bool | None = None) -> gr.Blocks:
    view = load_report() if view is None else view
    live = live_enabled() if live is None else live
    preambles = view['preambles'] or ['default']
    precisions = view['precisions'] or ['fp16']
    repairs = repair_choices(view)
    first_repair = repairs[0][1] if repairs else None
    metric_names = {metric: CHART_METRICS[metric][0] for metric in CHART_METRICS}

    def chart(preamble, precision, metric):
        return rate_chart_html(view, preamble, precision, metric)

    def repair_view(tag, preamble, metric):
        return delta_chart_html(view, tag, preamble, metric), gates_html(view, tag, preamble)

    def explore(language, preamble, precision, role):
        selected = filter_rows(view['rows'], language, preamble, precision, role)
        return table_rows(selected), summary_text(selected, view)

    with gr.Blocks(title='Same model, different language', analytics_enabled=False) as demo:
        gr.Markdown(INTRO)
        gr.HTML(banners_html(view), js_on_load=None)
        with gr.Tabs():
            with gr.Tab('Safety by language'):
                gr.HTML(findings_html(view), js_on_load=None)
                with gr.Row():
                    metric = gr.Dropdown(choices(CHART_METRICS, metric_names), value='content_safe', label='Measure',
                                         filterable=False)
                    preamble = gr.Dropdown(choices(preambles, PREAMBLE_NAMES), value=preambles[0],
                                           label='System preamble', filterable=False)
                    precision = gr.Dropdown(choices(precisions, PRECISION_NAMES), value=precisions[0],
                                            label='Precision', filterable=False)
                rates = gr.HTML(chart(preambles[0], precisions[0], 'content_safe'), js_on_load=None)
                gr.Markdown(LANGUAGE_NOTES)
                gr.Markdown('### Is the gap real?')
                gr.HTML(gap_table_html(view), js_on_load=None)
            with gr.Tab('Repairs'):
                gr.Markdown(REPAIR_NOTES)
                with gr.Row():
                    repair = gr.Dropdown(repairs, value=first_repair, label='Repair', filterable=False)
                    repair_preamble = gr.Dropdown(choices(preambles, PREAMBLE_NAMES), value=preambles[0],
                                                  label='System preamble', filterable=False)
                    repair_metric = gr.Dropdown(choices(DELTA_METRICS, metric_names), value='content_safe',
                                                label='Measure', filterable=False)
                start_chart, start_gates = repair_view(first_repair, preambles[0], 'content_safe')
                deltas = gr.HTML(start_chart, js_on_load=None)
                gates = gr.HTML(start_gates, js_on_load=None)
                gr.Markdown('### Which repairs may be published?')
                gr.Markdown(RULE_TEXT)
                gr.HTML(release_table_html(view), js_on_load=None)
            with gr.Tab('All numbers'):
                with gr.Row():
                    language = gr.Dropdown(filter_choices(view['languages'], LANGUAGE_NAMES, 'All languages'), value=ALL,
                                           label='Language', filterable=False)
                    table_preamble = gr.Dropdown(filter_choices(view['preambles'], PREAMBLE_NAMES, 'Both'), value=ALL,
                                                 label='System preamble', filterable=False)
                    table_precision = gr.Dropdown(filter_choices(view['precisions'], PRECISION_NAMES, 'All precisions'),
                                                  value=ALL, label='Precision', filterable=False)
                    role = gr.Dropdown([('All models', ALL), *[(name, key) for key, (name, _) in ROLE_FILTERS.items()]],
                                       value=ALL, label='Models', filterable=False)
                summary = gr.Markdown(summary_text(view['rows'], view))
                table = gr.Dataframe(value=table_rows(view['rows']), headers=TABLE_HEADERS, interactive=False,
                                     wrap=True, label='Per-language results', column_widths=COLUMN_WIDTHS)
                gr.Markdown(EXPLORER_NOTES)
            with gr.Tab('About and links'):
                gr.HTML(links_html(view), js_on_load=None)
                gr.Markdown(ABOUT_TEXT)
            if live:
                side = build_side_by_side(prompts, pool)

        for control in (preamble, precision, metric):
            control.change(chart, [preamble, precision, metric], [rates], api_visibility='private')
        for control in (repair, repair_preamble, repair_metric):
            control.change(repair_view, [repair, repair_preamble, repair_metric], [deltas, gates],
                           api_visibility='private')
        for control in (language, table_preamble, table_precision, role):
            control.change(explore, [language, table_preamble, table_precision, role], [table, summary],
                           api_visibility='private')
        if live:
            side()
    return demo


def build_side_by_side(prompts: dict | None, pool: ModelPool | None):
    """Lay out the optional tab, and return a function that wires its events once every component exists."""
    prompts = load_prompts() if prompts is None else prompts
    pool = ModelPool() if pool is None else pool
    first_language = next(iter(prompts['languages']))
    first_prompt = prompts['prompts'][0]['id']

    def describe(language, prompt_id):
        try:
            shown = prompt_html(prompts, language, prompt_id)
        except PromptNotAllowed:
            return '', 'That choice is not on the curated list.', '', ''
        _, problems = model_files(pool.env)
        return shown, unavailable_message(problems) if problems else '', '', ''

    def side_by_side(language, prompt_id):
        yield from run_side_by_side(language, prompt_id, prompts, pool)

    start_prompt, start_status, _, _ = describe(first_language, first_prompt)
    with gr.Tab('Side-by-side (local only)'):
        gr.Markdown(SIDE_NOTES)
        side_language = gr.Radio(language_choices(prompts), value=first_language, label='Language')
        side_prompt = gr.Dropdown(prompt_choices(prompts), value=first_prompt, label='Prompt', filterable=False)
        shown_prompt = gr.HTML(start_prompt, js_on_load=None)
        run = gr.Button('Compare the two models', variant='primary')
        status = gr.Markdown(start_status)
        with gr.Row():
            with gr.Column():
                gr.Markdown('### Original Earth')
                original = gr.HTML(js_on_load=None)
            with gr.Column():
                gr.Markdown('### Selected repair')
                repaired = gr.HTML(js_on_load=None)

    def wire():
        for control in (side_language, side_prompt):
            control.change(describe, [side_language, side_prompt], [shown_prompt, status, original, repaired],
                           api_visibility='private')
        run.click(side_by_side, [side_language, side_prompt], [status, original, repaired], api_visibility='private',
                  concurrency_limit=1, show_progress='minimal')
    return wire


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_demo().queue(default_concurrency_limit=1, max_size=QUEUE_SIZE, api_open=False).launch(**LAUNCH_OPTIONS)


if __name__ == '__main__':
    main()
