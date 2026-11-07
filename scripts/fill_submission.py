"""Fills the measured values in the submission drafts from a finished run, so nobody copies numbers by hand.

A value goes in only if the run's own files hold it. Everything else stays as its {{NAME}}: links, team decisions, and
the sentences that need someone to read the result first. `remaining.md` lists what is left in each draft. Which
[ALT-*] block survives follows the rule at the top of the article, and a case that rule can't settle keeps its markers.

    python scripts/fill_submission.py runs/experiments/full/earth docs/submission/filled
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ('article.md', 'model_card_template.md', 'dataset_card_template.md', 'roi.md')
TEAM_SECTION = re.compile(r'^## (Appendix for the team|Placeholder list)', re.M)
PLACEHOLDER = re.compile(r'\{\{([A-Z0-9_]+)\}\}')
LIST_ROW = re.compile(r'^\| `\{\{([A-Z0-9_]+)\}\}` \| (.+?) \|', re.M)

LANGUAGES = {'AR': 'ar', 'SW': 'sw', 'JV': 'jv', 'BN': 'bn', 'TH': 'th', 'KO': 'ko', 'EN': 'en'}
LANGUAGE_NAMES = {'ar': 'Arabic', 'sw': 'Swahili', 'jv': 'Javanese', 'bn': 'Bengali', 'th': 'Thai', 'ko': 'Korean',
                  'en': 'English'}
CAPABILITY_LANGUAGE = {'AR': 'ar', 'SW': 'sw', 'JV': 'id', 'BN': 'bn', 'KO': 'ko', 'EN': 'en'}   # no Javanese or Thai set
CONDITIONS = {'default': 'with the preamble', 'none': 'without the preamble'}
ARMS = {'RESTA': 'resta', 'CIRCUIT': 'circuit', 'MERGE': 'merge_back', 'CMERGE': 'circuit_merge', 'STEER': 'steer'}
ARM_DETAILS = {   # name, formula, SafeTune function, donors; these match the table in the model card
    'resta': ('ReSta', 'Earth + α(Global − Base)', 'apply_resta', 'Tiny Aya Global, Tiny Aya Base'),
    'circuit': ('Circuit', 'Earth + s(Global − Base)', 'apply_ctheta', 'Tiny Aya Global, Tiny Aya Base'),
    'merge_back': ('Merge-back', '(1 − λ)Earth + λGlobal', 'apply_wise_ft', 'Tiny Aya Global'),
    'circuit_merge': ('Circuit merge', 'Earth + λ(Global − Earth)', 'apply_ctheta', 'Tiny Aya Global'),
    'steer': ('Steering', 'Refusal direction added at layer 24 at run time', None, None)}
PRECISIONS = {'FP16': '', 'Q4': '-q4', 'NF4': '-nf4'}
JUDGE_NAMES = {'local': ('local 8B', 'Aya Expanse 8B and Qwen3-8B')}


def dig(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def percent(value: float | None) -> str | None:
    return None if value is None else f'{100 * value:.1f}'


def signed(value: float | None) -> str | None:
    return None if value is None else f'{100 * value:+.1f}'


def difference(cell: dict | None) -> str | None:
    """A paired change in points, with its 95% CI."""
    if not cell or cell.get('delta') is None:
        return None
    return f"{signed(cell['delta'])} ({signed(cell['ci'][0])} to {signed(cell['ci'][1])})"


def rate_with_ci(rate: float | None, ci: list | None) -> str | None:
    if rate is None or not ci:
        return None
    return f'{percent(rate)}% ({percent(ci[0])}–{percent(ci[1])})'


def p_value(value: float | None) -> str | None:
    if value is None:
        return None
    return '<0.0001' if value < 0.0001 else f'{value:.4f}'


def unclear_share(cell: dict | None) -> float | None:
    """Unclear labels over harmful and benign rows together, the share the knob score in repair/select.py uses."""
    harmful, benign = dig(cell, 'harmful_refusal'), dig(cell, 'over_refusal')
    if not harmful or not benign or harmful.get('unclear') is None or benign.get('unclear') is None:
        return None
    return (harmful['unclear'] * harmful['n'] + benign['unclear'] * benign['n']) / (harmful['n'] + benign['n'])


def numbers_pass(gate: dict | None) -> bool:
    return bool(gate) and all(dig(gate, name, 'pass') for name in ('harmful_gap_closed', 'over_refusal_increase',
                                                                   'capability_drop', 'coverage'))


def gate_word(gate: dict | None) -> str | None:
    """`pass` also needs human calibration, so a bare 'no' would hide an arm that met every number."""
    if not gate:
        return None
    if gate.get('pass'):
        return 'yes'
    return 'numbers pass, not validated' if numbers_pass(gate) else 'no'


def largest(gate: dict | None, name: str) -> str | None:
    values = dig(gate, name, 'by_lang')
    return signed(max(values.values())) if values else None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


@dataclass
class Results:
    run: Path
    report: dict
    selection: dict
    localisation: dict
    manifest: dict

    @property
    def drifted(self) -> str:
        return self.report.get('family', 'earth')

    @property
    def target(self) -> str | None:
        return self.selection.get('worst_language')

    @property
    def condition(self) -> str:
        """The confirmed condition for the target language; with none, the condition the knobs were chosen in."""
        return (self.report.get('confirmed_conditions') or ['default'])[0]

    @property
    def best(self) -> dict:
        return self.selection.get('selected_for_quantization') or {}

    def gaps(self, condition: str) -> dict:
        return dig(self.report, 'gap_confirmation', 'conditions', condition) or {}

    def cell(self, tag: str, lang: str | None, condition: str | None = None) -> dict:
        return dig(self.report, 'models', tag, 'conditions', condition or self.condition, lang) or {}

    def capability(self, tag: str, lang: str) -> dict:
        return dig(self.report, 'models', tag, 'capability_by_condition', self.condition, lang) or {}

    def gate(self, tag: str | None) -> dict | None:
        return dig(self.report, 'thresholds', f'{tag}/{self.condition}')


def _safe_rate(results: Results, tag: str, lang: str | None, condition: str | None = None) -> float | None:
    return dig(results.cell(tag, lang, condition), 'content_safe', 'rate')


def gap_values(results: Results) -> dict:
    values = {}
    for suffix, condition in (('', 'default'), ('_NONE', 'none')):
        for code, lang in LANGUAGES.items():
            gap = results.gaps(condition).get(lang, {})
            confirmed = 'not measurable' if gap.get('measurable') is False else 'yes' if gap.get('confirmed') else 'no'
            values.update({f'GAP_{code}_GLOBAL{suffix}': percent(_safe_rate(results, 'global', lang, condition)),
                           f'GAP_{code}_EARTH{suffix}': percent(_safe_rate(results, results.drifted, lang, condition)),
                           f'GAP_{code}_FIRE{suffix}': percent(dig(results.report, 'comparison_models', 'fire', 'conditions',
                                                                   condition, lang, 'rate')),
                           f'GAP_{code}_DIFF{suffix}': difference(gap), f'GAP_{code}_P{suffix}': p_value(gap.get('p_mcnemar')),
                           f'GAP_{code}_OK{suffix}': confirmed if gap else None})
        for model, tag in (('GLOBAL', 'global'), ('EARTH', results.drifted)):
            rates = [_safe_rate(results, tag, lang, condition) for lang in LANGUAGES.values()]
            if all(rate is not None for rate in rates):
                name = 'DEFAULT' if condition == 'default' else 'NONE'
                values[f'HELDOUT_MEAN_{model}_{name}'] = percent(sum(rates) / len(rates))
                values[f'HELDOUT_WORST_{model}_{name}'] = percent(min(rates))
    tests = dig(results.report, 'gap_confirmation', 'tests')
    cells = [(lang, condition, cell) for condition in CONDITIONS for lang, cell in results.gaps(condition).items()]
    if tests:
        values.update({'N_GAP_TESTS': str(tests), 'GAP_P_THRESHOLD': f"{results.report['gap_confirmation']['alpha']:.4f}"})
    if cells:
        lang, condition, top = max(cells, key=lambda item: item[2]['delta'])
        values.update({'LARGEST_GAP_POINTS': f"{100 * top['delta']:.1f}",
                       'LARGEST_GAP_LANGUAGE': f'{LANGUAGE_NAMES[lang]}, {CONDITIONS[condition]}',
                       'LARGEST_GAP_CI': f"95% CI {signed(top['ci'][0])} to {signed(top['ci'][1])}",
                       'GAP_UPPER_BOUND': f"{100 * max(cell['ci'][1] for _, _, cell in cells):.1f}",
                       'SCALE_JUDGE_BEFORE': f"{100 * top['delta']:.1f} points"})
        confirmed = [f'{LANGUAGE_NAMES[lang]} {CONDITIONS[condition]}' for lang, condition, cell in cells if cell.get('confirmed')]
        values['CONFIRMED_CELLS'] = ', '.join(confirmed) if confirmed else None
    for model, tag in (('GLOBAL', 'global'), ('EARTH', results.drifted)):
        rates = [_safe_rate(results, tag, lang, condition) for condition in CONDITIONS for lang in LANGUAGES.values()]
        rates = [rate for rate in rates if rate is not None]
        if rates:
            values.update({f'{model}_RANGE_LOW': percent(min(rates)), f'{model}_RANGE_HIGH': percent(max(rates))})
    return values


def target_values(results: Results) -> dict:
    target = results.target
    if not target:
        return {}
    gap = results.gaps(results.condition).get(target, {})
    values = {'TARGET_LANGUAGE': LANGUAGE_NAMES[target], 'TARGET_CONDITION': CONDITIONS[results.condition],
              'TARGET_SAFE_EARTH': percent(_safe_rate(results, results.drifted, target)),
              'TARGET_SAFE_GLOBAL': percent(_safe_rate(results, 'global', target)),
              'TARGET_GAP_POINTS': None if not gap else f"{100 * gap['delta']:.1f}",
              'TARGET_GAP_CI': None if not gap else f"95% CI {signed(gap['ci'][0])} to {signed(gap['ci'][1])}",
              'TARGET_GAP_P': p_value(gap.get('p_mcnemar')), 'TARGET_GAP_N': None if not gap else str(gap['n'])}
    for letter, word, tag in (('G', 'GLOBAL', 'global'), ('E', 'EARTH', results.drifted), ('R', None, results.best.get('tag'))):
        split = dig(results.cell(tag, target), 'content_safe') or {}
        for name, key in (('SAFE', 'rate'), ('UNSAFE', 'unsafe'), ('INVALID', 'invalid'), ('DISPUTED', 'disputed')):
            share = percent(split.get(key))
            values[f'TARGET_{letter}_{name}'] = None if share is None else f'{share}%'
            if word:
                values[f'R_{word}_{name}'] = share
        if word:
            values[f'R_{word}_REFUSAL'] = percent(dig(results.cell(tag, target), 'harmful_refusal', 'rate'))
    for letter, tag in (('G', 'global'), ('E', results.drifted)):
        for word, condition in (('ON', 'default'), ('OFF', 'none')):
            cell = results.cell(tag, target, condition)
            split = cell.get('content_safe', {})
            values.update({f'FAIL_{letter}_{word}_SAFE': percent(split.get('rate')),
                           f'FAIL_{letter}_{word}_UNSAFE': percent(split.get('unsafe')),
                           f'FAIL_{letter}_{word}_INVALID': percent(split.get('invalid')),
                           f'FAIL_{letter}_{word}_DISPUTED': percent(split.get('disputed')),
                           f'FAIL_{letter}_{word}_DEGENERATE': percent(dig(cell, 'degenerate', 'rate'))})
    return values


def repair_values(results: Results) -> dict:
    values = {}
    target = results.target
    earth_unclear = {lang: unclear_share(results.cell(results.drifted, lang)) for lang in LANGUAGES.values()}
    for key, arm in ARMS.items():
        selected = dig(results.selection, 'arms', arm, 'selected')
        if not selected:
            continue
        tag = selected['tag']
        gate = results.gate(tag)
        split = dig(results.cell(tag, target), 'content_safe') or {}
        rises = [unclear_share(results.cell(tag, lang)) - base for lang, base in earth_unclear.items()
                 if base is not None and unclear_share(results.cell(tag, lang)) is not None]
        l2 = dig(results.localisation, 'edits', arm, 'relative_edit_l2')
        values.update({f'R_{key}_KNOB': str(selected['knob']), f'R_{key}_SAFE': percent(split.get('rate')),
                       f'R_{key}_UNSAFE': percent(split.get('unsafe')), f'R_{key}_INVALID': percent(split.get('invalid')),
                       f'R_{key}_DISPUTED': percent(split.get('disputed')),
                       f'R_{key}_REFUSAL': percent(dig(results.cell(tag, target), 'harmful_refusal', 'rate')),
                       f'R_{key}_OVER': largest(gate, 'over_refusal_increase'), f'R_{key}_CAP': largest(gate, 'capability_drop'),
                       f'R_{key}_UNCLEAR': signed(sum(rises) / len(rises)) if rises else None,
                       f'R_{key}_PASS': gate_word(gate)})
        if arm != 'steer':
            values[f'R_{key}_L2'] = None if l2 is None else f'{l2:.5f}'
    return values


def selected_arm_values(results: Results) -> dict:
    best = results.best
    if not best:
        return {}
    arm, tag = best['arm'], best['tag']
    name, formula, function, donors = ARM_DETAILS[arm]
    spec = yaml.safe_load((ROOT / 'configs' / 'spec.yaml').read_text(encoding='utf-8'))
    knobs = next((row for row in spec['repair_arms'] if row['id'] == arm), {})
    edit = dig(results.localisation, 'edits', arm) or {}
    layers = edit.get('edited_layers')
    gate = results.gate(tag)
    closed = dig(gate, 'harmful_gap_closed', 'value')
    change = dig(results.report, 'paired_comparisons', f'{tag}/{results.condition}', 'vs_drifted', 'safety', results.target,
                 'content_safe')
    values = {'BEST_ARM': name, 'ARM_NAME': name, 'ARM_FORMULA': formula, 'SAFETUNE_FUNCTION': function, 'DONOR_MODELS': donors,
              'KNOB_NAME': knobs.get('knob'), 'KNOB_VALUE': str(best['knob']),
              'KNOB_GRID': ', '.join(str(value) for value in knobs['values']) if knobs else None,
              'PRECISION': 'FP16', 'EDITED_TENSORS': None if edit.get('edited_tensors') is None else str(edit['edited_tensors']),
              'EDITED_LAYERS': None if not layers else ', '.join(str(layer) for layer in layers),
              'RELATIVE_EDIT_L2': None if edit.get('relative_edit_l2') is None else f"{edit['relative_edit_l2']:.5f}",
              'CIRCUIT_SOURCE': None if arm in ('circuit', 'circuit_merge') else 'not used',
              'GAP_CLOSED_PCT': None if closed is None else f'{100 * closed:.1f}',
              'MAX_OVER_REFUSAL_RISE': largest(gate, 'over_refusal_increase'),
              'MAX_CAPABILITY_DROP': largest(gate, 'capability_drop'), 'SAFE_CHANGE': difference(change),
              'PASS_GAP': _yes_no(dig(gate, 'harmful_gap_closed', 'pass')),
              'PASS_OVER': _yes_no(dig(gate, 'over_refusal_increase', 'pass')),
              'PASS_CAP': _yes_no(dig(gate, 'capability_drop', 'pass'))}
    for word, suffix in PRECISIONS.items():
        gate = results.gate(tag + suffix)
        earth = percent(_safe_rate(results, results.drifted + suffix, results.target))
        repaired = percent(_safe_rate(results, tag + suffix, results.target))
        over, cap = largest(gate, 'over_refusal_increase'), largest(gate, 'capability_drop')
        values.update({f'Q_EARTH_{word}_SAFE': earth, f'Q_REP_{word}_SAFE': repaired, f'Q_REP_{word}_OVER': over,
                       f'Q_REP_{word}_CAP': cap, f'{word}_E_SAFE': earth, f'{word}_R_SAFE': repaired, f'{word}_OVER': over,
                       f'{word}_CAP': cap})
        if suffix:
            verdict = gate_word(gate)
            values[f'PASS_{word}'] = verdict
            values[f'{word}_RESULT'] = {'yes': 'held', 'no': 'did not hold'}.get(verdict, verdict)
    return values


def _yes_no(flag: bool | None) -> str | None:
    return None if flag is None else 'yes' if flag else 'no'


def language_values(results: Results) -> dict:
    values = {}
    tag = results.best.get('tag')
    models = (('G', 'global'), ('E', results.drifted), ('R', tag))
    for code, lang in LANGUAGES.items():
        capability_lang = CAPABILITY_LANGUAGE.get(code)
        for letter, model in models:
            cell = results.cell(model, lang)
            refusal, over = cell.get('harmful_refusal', {}), cell.get('over_refusal', {})
            capability = results.capability(model, capability_lang) if capability_lang else {}
            unclear = percent(unclear_share(cell))
            values.update({f'{code}_{letter}_SAFE': rate_with_ci(dig(cell, 'content_safe', 'rate'), dig(cell, 'content_safe', 'ci')),
                           f'{code}_{letter}_REF': rate_with_ci(refusal.get('rate'), refusal.get('ci')),
                           f'{code}_{letter}_OVER': rate_with_ci(over.get('rate'), over.get('ci')),
                           f'{code}_{letter}_UNC': None if unclear is None else f'{unclear}%'})
            if capability_lang:
                values[f'{code}_{letter}_CAP'] = rate_with_ci(capability.get('accuracy'), capability.get('ci95'))
            if letter != 'G':
                values.update({f'L_{code}_SAFE_{letter}': percent(dig(cell, 'content_safe', 'rate')),
                               f'L_{code}_REF_{letter}': percent(refusal.get('rate')),
                               f'L_{code}_OVER_{letter}': percent(over.get('rate'))})
                if capability_lang:
                    values[f'L_{code}_CAP_{letter}'] = percent(capability.get('accuracy'))
        reference = results.cell('global', lang)
        for name, key in (('HARM', 'content_safe'), ('BEN', 'over_refusal')):
            n = dig(reference, key, 'n')
            values[f'{code}_N_{name}'] = None if n is None else str(n)
    return values


def circuit_values(results: Results) -> dict:
    values = {}
    examples = []
    for letter, model in (('G', 'global'), ('E', results.drifted)):
        for code in ('EN', 'JV', 'BN'):
            summary = _read(results.run / 'circuits' / f'{model}-circuit__{code.lower()}' / 'summary.json')
            faithfulness = summary.get('faithfulness', {})
            for name, key in (('PATCH', 'patching_score'), ('ABL', 'ablation_score')):
                score = faithfulness.get(key)
                values[f'CIRC_{name}_{letter}_{code}'] = None if score is None else f'{score:.2f}'
            if summary.get('task_examples') is not None:
                examples.append(summary['task_examples'])
    if examples:
        low, high = min(examples), max(examples)
        values['CIRCUIT_N_PAIRS'] = str(low) if low == high else f'{low} to {high}'
    overlap = dig(_read(results.run / 'circuits' / 'overlap.json'), 'jaccard') or {}
    drifted = results.drifted
    pairs = [('en', 'jv'), ('en', 'bn'), ('jv', 'bn')]
    across = [(a, b, dig(overlap, f'global/{a}', f'global/{b}')) for a, b in pairs]
    between = [(lang, dig(overlap, f'global/{lang}', f'{drifted}/{lang}')) for lang in ('en', 'jv', 'bn')]
    if all(score is not None for *_, score in across):
        values['CIRC_JACCARD_LANGS'] = ', '.join(f'{a}–{b} {score:.2f}' for a, b, score in across)
    if all(score is not None for _, score in between):
        values['CIRC_JACCARD_MODELS'] = ', '.join(f'{lang} {score:.2f}' for lang, score in between)
    layers = dig(results.localisation, 'edits', 'circuit', 'layer_subset')
    values['CIRCUIT_LAYERS'] = None if not layers else ', '.join(str(layer) for layer in layers)
    return values


def localisation_values(results: Results) -> dict:
    values = {}
    rows = [('C_TWIN', dig(results.localisation, 'whole_model_twin', 'circuit')),
            ('CM_TWIN', dig(results.localisation, 'whole_model_twin', 'circuit_merge'))]
    controls = (results.localisation.get('random_control') or {}).values()
    rows += [(f"RAND{control['seed']}", control) for control in controls]
    for name, block in rows:
        cells = dig(block, 'conditions', results.condition) or {}
        values.update({f'LOC_{name}_SAFE': difference(cells.get('target_content_safe')),
                       f'LOC_{name}_OVER': difference(cells.get('over_refusal')),
                       f'LOC_{name}_CAP': difference(cells.get('capability'))})
    return values


def data_values(results: Results) -> dict:
    manifest = results.manifest
    if not manifest:
        return {}
    counts = manifest.get('counts', {})
    files = manifest.get('files', {})
    translation = manifest.get('translation', {})
    leakage = manifest.get('leakage_check', {})
    values = {f'C_{code}_{word}_{split.upper()}': str(counts[f'{split}/{kind}/{lang}'])
              for code, lang in LANGUAGES.items() for word, kind in (('HARM', 'harmful'), ('BEN', 'benign'))
              for split in ('heldout', 'repair') if f'{split}/{kind}/{lang}' in counts}
    rows = {name: dig(files, name, 'rows') for name in ('heldout', 'repair')}
    if all(rows.values()):
        values.update({'N_PROMPTS_TOTAL': f'{sum(rows.values()):,}', 'N_PROMPTS_HELDOUT': f"{rows['heldout']:,}",
                       'N_PROMPTS_REPAIR': f"{rows['repair']:,}"})
    harmful = [counts[f'heldout/harmful/{lang}'] for lang in LANGUAGES.values() if f'heldout/harmful/{lang}' in counts]
    if harmful:
        values['GAP_PROMPTS_NOW'] = f'{min(harmful)} to {max(harmful)}'
    minhash = [dig(cell, 'stage_counts', 'MinHashDeduplicator') or {} for cell in manifest.get('curatorkit', {}).values()]
    if minhash and all('output_count' in stage for stage in minhash):
        values['MINHASH_REMOVED'] = str(sum(stage['input_count'] - stage['output_count'] for stage in minhash))
    values.update({'TRANSLATION_REQUESTED': _text(translation.get('requested')), 'TRANSLATION_FAILED': _text(translation.get('failed')),
                   'TRANSLATION_QUALITY_STATUS': translation.get('quality_status'),
                   'LEAKAGE_REMOVED_ROWS': _text(leakage.get('removed_rows')),
                   'LEAKAGE_REMOVED_GROUPS': _text(leakage.get('removed_groups')),
                   'CARD_DATE': manifest.get('built_at', '')[:10] or None,
                   'DATA_MANIFEST_SHA256': hashlib.sha256((results.run / 'data' / 'manifest.json').read_bytes()).hexdigest()})
    if translation.get('quality_status') == 'unreviewed':
        values.update({'TRANSLATION_REVIEWED_ROWS': '0', 'TRANSLATION_REJECTED_ROWS': '0', 'TRANSLATION_REVIEWERS': 'none yet'})
    return values


def _text(value) -> str | None:
    return None if value is None else str(value)


def run_values(results: Results) -> dict:
    report = results.report
    judges = JUDGE_NAMES.get(report.get('judge_set'), (None, None))
    gaps = [results.gaps(condition) for condition in CONDITIONS]
    missing = sorted({lang for cells in gaps for lang, cell in cells.items() if cell.get('measurable') is False})
    tested = [LANGUAGE_NAMES[lang] for lang in LANGUAGES.values() if lang not in missing]
    agreement = dig(report, 'models', results.drifted, 'eligibility', 'by_lang') or {}
    scores = [(lang, agreement.get(lang, {}).get('agreement')) for lang in LANGUAGES.values()]
    confirmed = report.get('confirmed_conditions') or []
    frozen = re.search(r'frozen on (\d{1,2} \w+ \d{4})', (ROOT / 'SPEC.md').read_text(encoding='utf-8'))
    values = {'SPEC_DIGEST': report.get('spec_digest'), 'JUDGE_SET_USED': judges[0], 'JUDGE_PAIR': judges[1],
              'SPEC_FREEZE_DATE': frozen.group(1) if frozen and report.get('spec_frozen') else None,
              'LANGUAGES_TESTED': ', '.join(tested) if any(gaps) else None,
              'LANGUAGES_NOT_MEASURABLE': (', '.join(LANGUAGE_NAMES[lang] for lang in missing) or 'none') if any(gaps) else None}
    if all(score is not None for _, score in scores):
        values['AC1_BY_LANGUAGE'] = ', '.join(f'{lang} {score:.2f}' for lang, score in scores)
    if 'empirical_repair_success' in report:
        values['ACCEPTANCE_STATUS'] = ('passed acceptance' if report['empirical_repair_success']
                                       else 'failed acceptance' if confirmed else 'no repair claim')
    return values


def measured(run: Path) -> dict[str, str]:
    """Every value the run's files settle. A value they don't hold is left out, so its placeholder stays."""
    results = Results(run, _read(run / 'report.json'), _read(run / 'selection.json'), _read(run / 'localisation.json'),
                      _read(run / 'data' / 'manifest.json'))
    values = {}
    for part in (gap_values, target_values, repair_values, selected_arm_values, language_values, circuit_values,
                 localisation_values, data_values, run_values):
        values.update(part(results))
    return {name: value for name, value in values.items() if value is not None}


def kept_blocks(report: dict, selection: dict) -> dict[str, bool]:
    """Which [ALT-*] blocks stay. A block missing here keeps its markers, because the rule can't settle it from the files."""
    conditions = dig(report, 'gap_confirmation', 'conditions')
    if not conditions:
        return {}
    if not any(cell.get('confirmed') for cells in conditions.values() for cell in cells.values()):
        return {'ALT-A': False, 'ALT-A1': False, 'ALT-A2': False, 'ALT-B': True}
    blocks = {'ALT-A': True, 'ALT-B': False}
    if report.get('empirical_repair_success'):
        return {**blocks, 'ALT-A1': True, 'ALT-A2': False}
    tag = dig(selection, 'selected_for_quantization', 'tag')
    gates = [dig(report, 'thresholds', f'{tag}/{condition}') for condition in report.get('confirmed_conditions') or []]
    if any(numbers_pass(gate) for gate in gates):
        return blocks     # every number passed but nothing is validated: a person decides how to word that
    return {**blocks, 'ALT-A1': False, 'ALT-A2': True}


def apply_blocks(text: str, blocks: dict[str, bool]) -> str:
    for name in sorted(blocks, key=len, reverse=True):     # the nested ALT-A1 and ALT-A2 go first
        pattern = rf'\*\*\[{name}\]\*\*\n(.*?)\*\*\[END {name}\]\*\*\n?'
        text = re.sub(pattern, lambda match: match.group(1) if blocks[name] else '', text, flags=re.S)
    return text


def fill(text: str, values: dict[str, str], blocks: dict[str, bool]) -> tuple[str, dict[str, str]]:
    """The filled draft, without its team section, and what is still open, with each item's description from that section."""
    section = TEAM_SECTION.search(text)
    body, team = (text[:section.start()].rstrip() + '\n', text[section.start():]) if section else (text, '')
    described = dict(LIST_ROW.findall(team))
    body = PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), apply_blocks(body, blocks))
    return body, {name: described.get(name, '') for name in dict.fromkeys(PLACEHOLDER.findall(body))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    values = measured(args.run)
    report, selection = _read(args.run / 'report.json'), _read(args.run / 'selection.json')
    blocks = kept_blocks(report, selection)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ['# What is still open', '',
             f"Run `{args.run.as_posix()}`, spec `{report.get('spec_digest')}`, frozen: {report.get('spec_frozen')}, "
             f"validated: {report.get('validated')}. Blocks kept: "
             f"{', '.join(name for name, keep in sorted(blocks.items()) if keep) or 'none settled'}.", '']
    for name in DRAFTS:
        text = (ROOT / 'docs' / 'submission' / name).read_text(encoding='utf-8')
        filled, left = fill(text, values, blocks)
        (args.out / name.replace('_template', '')).write_text(filled, encoding='utf-8', newline='\n')
        lines += [f'## {name} ({len(left)} open)', ''] + [f'- `{key}`: {about}'.rstrip(': ') for key, about in left.items()] + ['']
        print(f'{name}: {len(left)} open')
    (args.out / 'values.json').write_text(json.dumps(values, indent=1, ensure_ascii=False, sort_keys=True) + '\n',
                                          encoding='utf-8', newline='\n')
    (args.out / 'remaining.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(f'{len(values)} values filled into {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
