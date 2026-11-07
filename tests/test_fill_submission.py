"""scripts/fill_submission.py: only what a run's files hold goes into the drafts, and everything else stays visible."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fill_submission', ROOT / 'scripts' / 'fill_submission.py')
fill_submission = importlib.util.module_from_spec(SPEC)
sys.modules['fill_submission'] = fill_submission
SPEC.loader.exec_module(fill_submission)

LANGS = ['ar', 'bn', 'en', 'jv', 'ko', 'sw', 'th']
CAPABILITY_LANGS = ['ar', 'bn', 'en', 'id', 'ko', 'sw']


def cell(safe: float) -> dict:
    return {'harmful_refusal': {'rate': 0.4, 'ci': [0.3, 0.5], 'n': 100, 'partial': 0.0, 'unclear': 0.1},
            'over_refusal': {'rate': 0.2, 'ci': [0.1, 0.3], 'n': 50, 'partial': 0.0, 'unclear': 0.3},
            'content_safe': {'rate': safe, 'ci': [safe - 0.05, safe + 0.05], 'n': 100, 'unsafe': 1 - safe - 0.1,
                             'invalid': 0.06, 'disputed': 0.04},
            'degenerate': {'rate': 0.02, 'ci': [0.0, 0.04], 'n': 150}}


def model(safe: float, capability: float = 0.5) -> dict:
    per_lang = {lang: {'accuracy': capability, 'ci95': [capability - 0.05, capability + 0.05]} for lang in CAPABILITY_LANGS}
    return {'conditions': {condition: {lang: cell(safe) for lang in LANGS} for condition in ('default', 'none')},
            'capability_by_condition': {'default': per_lang, 'none': per_lang},
            'eligibility': {'by_lang': {lang: {'agreement': 0.8} for lang in LANGS}}}


def gate(passed: bool = False, numbers: bool = False) -> dict:
    return {'harmful_gap_closed': {'value': 0.6, 'pass': numbers},
            'over_refusal_increase': {'by_lang': {'jv': 0.01, 'ar': -0.02}, 'pass': numbers},
            'capability_drop': {'by_lang': {'jv': 0.005}, 'pass': numbers}, 'coverage': {'pass': numbers}, 'pass': passed}


def report(confirmed: bool = True, **extra) -> dict:
    gap = {'delta': 0.2, 'ci': [0.1, 0.3], 'p_mcnemar': 0.00001, 'n': 176, 'confirmed': confirmed, 'measurable': True}
    other = {**gap, 'delta': 0.01, 'ci': [-0.05, 0.07], 'p_mcnemar': 0.5, 'confirmed': False}
    cells = {lang: dict(gap if lang == 'jv' else other) for lang in LANGS}
    base = {'family': 'earth', 'spec_digest': 'abc123', 'spec_frozen': True, 'judge_set': 'local', 'validated': False,
            'empirical_repair_success': False, 'confirmed_conditions': ['none'] if confirmed else [],
            'models': {'global': model(0.9), 'earth': model(0.7), 'earth-resta-0p25': model(0.85, 0.49)},
            'gap_confirmation': {'metric': 'content_safe', 'tests': 14, 'alpha': 0.05 / 14,
                                 'conditions': {'default': cells, 'none': cells}},
            'thresholds': {'earth-resta-0p25/default': gate(), 'earth-resta-0p25/none': gate()},
            'paired_comparisons': {}, 'comparison_models': {'fire': {'conditions': {'default': {'jv': {'rate': 0.8}}}}}}
    return {**base, **extra}


SELECTION = {'worst_language': 'jv', 'selected_for_quantization': {'arm': 'resta', 'knob': 0.25, 'tag': 'earth-resta-0p25'},
             'arms': {'resta': {'selected': {'arm': 'resta', 'knob': 0.25, 'tag': 'earth-resta-0p25'}}}}


def write_run(run: Path, report_json: dict, selection: dict = SELECTION) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    (run / 'report.json').write_text(json.dumps(report_json), encoding='utf-8')
    (run / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    return run


def test_gap_cells_come_straight_from_the_report(tmp_path):
    values = fill_submission.measured(write_run(tmp_path, report()))
    assert values['GAP_JV_GLOBAL'] == '90.0' and values['GAP_JV_EARTH'] == '70.0'
    assert values['GAP_JV_DIFF'] == '+20.0 (+10.0 to +30.0)'
    assert values['GAP_JV_P'] == '<0.0001' and values['GAP_AR_P'] == '0.5000'
    assert values['GAP_JV_OK'] == 'yes' and values['GAP_AR_OK'] == 'no'
    assert values['GAP_JV_FIRE'] == '80.0'
    assert values['GAP_P_THRESHOLD'] == '0.0036' and values['N_GAP_TESTS'] == '14'
    assert values['LARGEST_GAP_POINTS'] == '20.0' and values['GAP_UPPER_BOUND'] == '30.0'


def test_a_language_below_the_agreement_floor_is_never_called_confirmed(tmp_path):
    data = report()
    data['gap_confirmation']['conditions']['default']['jv'] = {**data['gap_confirmation']['conditions']['default']['jv'],
                                                               'confirmed': False, 'measurable': False}
    values = fill_submission.measured(write_run(tmp_path, data))
    assert values['GAP_JV_OK'] == 'not measurable'
    assert values['LANGUAGES_NOT_MEASURABLE'] == 'Javanese' and 'Javanese' not in values['LANGUAGES_TESTED']


def test_the_headline_uses_the_confirmed_condition_and_falls_back_to_the_selection_one(tmp_path):
    assert fill_submission.measured(write_run(tmp_path / 'a', report()))['TARGET_CONDITION'] == 'without the preamble'
    values = fill_submission.measured(write_run(tmp_path / 'b', report(confirmed=False)))
    assert values['TARGET_CONDITION'] == 'with the preamble'


def test_capability_uses_indonesian_for_javanese_and_leaves_thai_out(tmp_path):
    data = report()
    data['models']['earth']['capability_by_condition']['none']['id']['accuracy'] = 0.42
    values = fill_submission.measured(write_run(tmp_path, data))
    assert values['L_JV_CAP_E'] == '42.0'
    assert 'L_TH_CAP_E' not in values and 'TH_E_CAP' not in values


def test_unclear_share_weights_harmful_and_benign_rows_like_knob_selection():
    assert fill_submission.unclear_share(cell(0.9)) == pytest.approx((0.1 * 100 + 0.3 * 50) / 150)


def test_an_arm_that_met_every_number_without_validation_is_not_called_a_plain_no():
    assert fill_submission.gate_word(gate(passed=True, numbers=True)) == 'yes'
    assert fill_submission.gate_word(gate(numbers=True)) == 'numbers pass, not validated'
    assert fill_submission.gate_word(gate()) == 'no'


def test_nothing_is_invented_for_files_the_run_lacks(tmp_path):
    values = fill_submission.measured(write_run(tmp_path, report()))
    missing = {name for name in values if name.startswith(('CIRC_', 'LOC_', 'C_', 'Q_REP_Q4', 'Q_REP_NF4', 'N_PROMPTS'))}
    assert not missing


def test_an_empty_run_fills_nothing(tmp_path):
    assert fill_submission.measured(tmp_path) == {}


def test_every_value_names_a_real_placeholder(tmp_path):
    localisation = {'edits': {'resta': {'relative_edit_l2': 0.01, 'edited_tensors': 9, 'edited_layers': [1, 2]},
                              'circuit': {'layer_subset': [1, 2]}},
                    'whole_model_twin': {'circuit': {'conditions': {'none': {}}}},
                    'random_control': {'x': {'seed': 1, 'conditions': {'none': {}}}}}
    run = write_run(tmp_path, report())
    (run / 'localisation.json').write_text(json.dumps(localisation), encoding='utf-8')
    known = set()
    for name in fill_submission.DRAFTS:
        known |= set(fill_submission.PLACEHOLDER.findall((ROOT / 'docs' / 'submission' / name).read_text(encoding='utf-8')))
    assert set(fill_submission.measured(run)) <= known


@pytest.mark.parametrize('kwargs, expected', [
    ({'confirmed': False}, {'ALT-A': False, 'ALT-A1': False, 'ALT-A2': False, 'ALT-B': True}),
    ({'confirmed': True}, {'ALT-A': True, 'ALT-B': False, 'ALT-A1': False, 'ALT-A2': True}),
    ({'confirmed': True, 'empirical_repair_success': True}, {'ALT-A': True, 'ALT-B': False, 'ALT-A1': True, 'ALT-A2': False}),
])
def test_the_article_rule_picks_the_blocks(kwargs, expected):
    assert fill_submission.kept_blocks(report(**kwargs), SELECTION) == expected


def test_a_confirmed_gap_outside_the_target_language_means_no_arm_passed():
    data = report()
    data['confirmed_conditions'] = []
    assert fill_submission.kept_blocks(data, SELECTION)['ALT-A2'] is True


def test_numbers_that_pass_unvalidated_leave_the_wording_to_a_person():
    data = report()
    data['thresholds']['earth-resta-0p25/none'] = gate(numbers=True)
    assert fill_submission.kept_blocks(data, SELECTION) == {'ALT-A': True, 'ALT-B': False}


def test_a_report_without_gap_results_settles_no_block():
    assert fill_submission.kept_blocks({}, {}) == {}


def test_nested_blocks_are_resolved_and_their_markers_removed():
    text = ('**[ALT-A]**\nA\n**[ALT-A1]**\none\n**[END ALT-A1]**\n**[ALT-A2]**\ntwo\n**[END ALT-A2]**\n**[END ALT-A]**\n'
            '**[ALT-B]**\nB\n**[END ALT-B]**\nafter\n')
    assert fill_submission.apply_blocks(text, {'ALT-A': True, 'ALT-A1': False, 'ALT-A2': True, 'ALT-B': False}) == 'A\ntwo\nafter\n'


def test_open_items_stay_visible_and_keep_their_description():
    text = 'Rate {{KNOWN}} and {{OPEN}}.\n\n## Placeholder list\n\n| `{{OPEN}}` | A link for later | Team |\n| `{{KNOWN}}` | x | y |\n'
    body, left = fill_submission.fill(text, {'KNOWN': '5.0'}, {})
    assert body == 'Rate 5.0 and {{OPEN}}.\n'
    assert left == {'OPEN': 'A link for later'}


def test_filling_a_real_draft_leaves_no_double_percent_signs_or_team_section(tmp_path):
    values = fill_submission.measured(write_run(tmp_path, report()))
    for name in fill_submission.DRAFTS:
        text = (ROOT / 'docs' / 'submission' / name).read_text(encoding='utf-8')
        body, _ = fill_submission.fill(text, values, fill_submission.kept_blocks(report(), SELECTION))
        assert '%%' not in body
        assert not re.search(r'^## (Appendix for the team|Placeholder list)', body, re.M)


def test_the_command_writes_the_drafts_and_a_list_of_what_is_left(tmp_path, monkeypatch):
    run, out = write_run(tmp_path / 'run', report()), tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['fill_submission.py', str(run), str(out)])
    assert fill_submission.main() == 0
    assert {path.name for path in out.iterdir()} == {'article.md', 'model_card.md', 'dataset_card.md', 'roi.md', 'values.json',
                                                     'remaining.md'}
    remaining = (out / 'remaining.md').read_text(encoding='utf-8')
    assert '`REPO_URL`' in remaining and 'spec `abc123`' in remaining


def test_the_freeze_date_appears_only_for_a_frozen_spec(tmp_path):
    assert fill_submission.measured(write_run(tmp_path / 'a', report()))['SPEC_FREEZE_DATE'] == '7 November 2026'
    assert 'SPEC_FREEZE_DATE' not in fill_submission.measured(write_run(tmp_path / 'b', report(spec_frozen=False)))


def test_written_sentences_apply_only_to_their_own_run(tmp_path):
    path = tmp_path / 'sentences.json'
    path.write_text(json.dumps({'spec_digest': 'abc123', 'sentences': {'REPO_URL': 'x'}}), encoding='utf-8')
    assert fill_submission.written(report(), path) == {'REPO_URL': 'x'}
    assert fill_submission.written(report(spec_digest='other'), path) == {}
    assert fill_submission.written(report(), tmp_path / 'missing.json') == {}


def test_the_sentences_on_file_name_real_placeholders_and_stay_plain_text():
    data = json.loads(fill_submission.SENTENCES.read_text(encoding='utf-8'))
    known = set()
    for name in fill_submission.DRAFTS:
        known |= set(fill_submission.PLACEHOLDER.findall((ROOT / 'docs' / 'submission' / name).read_text(encoding='utf-8')))
    assert set(data['sentences']) <= known
    assert not any('{{' in text or '%%' in text for text in data['sentences'].values())


def test_fertility_fills_each_language_but_english(tmp_path):
    run = write_run(tmp_path, report())
    languages = {lang: {'closed_per_tiny_aya': 1.0 + index / 10} for index, lang in enumerate(LANGS)}
    (run / 'fertility.json').write_text(json.dumps({'languages': languages}), encoding='utf-8')
    values = fill_submission.measured(run)
    assert values['FERTILITY_AR'] == '1.00' and values['FERTILITY_TH'] == '1.60'
    assert 'FERTILITY_EN' not in values
    assert 'FERTILITY_AR' not in fill_submission.measured(write_run(tmp_path / 'none', report()))


def write_circuits(run: Path, overlap: float = 1.0, max_records: int = 16) -> Path:
    for model_name in ('global', 'earth'):
        for lang in ('en', 'jv', 'bn'):
            folder = run / 'circuits' / f'{model_name}-circuit__{lang}'
            folder.mkdir(parents=True)
            summary = {'task_examples': 16, 'settings': {'n_examples': 16, 'data_max_records': max_records},
                       'faithfulness': {'patching_score': 0.99, 'ablation_score': 0.9,
                                        'baseline_comparison': {'circuit_score': 6.0, 'best_baseline_score': 0.5},
                                        'stability': {'mean_jaccard': overlap, 'n_runs': 3}}}
            (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return run


def test_circuit_rows_include_the_baseline_ratio(tmp_path):
    values = fill_submission.measured(write_circuits(write_run(tmp_path, report())))
    assert values['CIRC_PATCH_G_JV'] == '0.99' and values['CIRC_BASE_E_BN'] == '12.0×'
    assert values['CIRCUIT_N_PAIRS'] == '16'


def test_perfect_stability_on_the_same_pairs_is_called_deterministic_not_robust(tmp_path):
    sentence = fill_submission.measured(write_circuits(write_run(tmp_path, report())))['CIRC_STABILITY_SENTENCE']
    assert 'deterministic' in sentence and '3 seeds' in sentence


def test_stability_that_varies_reports_its_range(tmp_path):
    values = fill_submission.measured(write_circuits(write_run(tmp_path, report()), overlap=0.8, max_records=64))
    assert 'from 0.80 to 0.80' in values['CIRC_STABILITY_SENTENCE']
    assert 'same pairs' not in values['CIRC_STABILITY_SENTENCE']


def test_stability_waits_for_all_six_circuits(tmp_path):
    run = write_circuits(write_run(tmp_path, report()))
    (run / 'circuits' / 'earth-circuit__bn' / 'summary.json').unlink()
    values = fill_submission.measured(run)
    assert 'CIRC_STABILITY_SENTENCE' not in values and 'CIRC_BASE_E_BN' not in values and 'CIRC_BASE_G_EN' in values
