"""scripts/build_space_report.py: the Space gets only aggregate numbers, rebuilt from the public export, never by hand.

Exports here are synthetic and built under tmp_path. The last test checks the real space/results.json against the real
export when that export exists on this machine, and is skipped otherwise.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from langsafe.config import spec_digest

REPO = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location('build_space_report', REPO / 'scripts' / 'build_space_report.py')
build_space_report = importlib.util.module_from_spec(_spec)
sys.modules['build_space_report'] = build_space_report
_spec.loader.exec_module(build_space_report)
BuildError = build_space_report.BuildError

SPEC = {'thresholds': {'harmful_gap_closed_min': 0.5, 'over_refusal_increase_max': 0.02, 'capability_drop_max': 0.01,
                       'quantized_must_hold': True}, 'note': 'synthetic'}
CELL = {'rate': 0.8125000000000001, 'ci': [0.75, 0.87], 'n': 176, 'unsafe': 0.01, 'invalid': 0.09, 'disputed': 0.09}


def report() -> dict:
    model = {'status': 'provisional', 'eligibility': {'eligible': False, 'status': 'provisional', 'statistic': 'gwet_ac1',
                                                      'min_agreement': 0.6, 'judge_count': 2,
                                                      'by_lang': {'jv': {'status': 'provisional', 'measurable': True,
                                                                         'agreement': 0.85, 'eligible': False}}},
             'conditions': {'default': {'jv': {'content_safe': dict(CELL), 'degenerate': {'rate': 0.08, 'ci': [0.05, 0.11],
                                                                                          'n': 323}}}},
             'capability_by_condition': {'default': {'id': {'accuracy': 0.5, 'ci95': [0.45, 0.55], 'n': 400, 'correct': 200,
                                                            'data_sha256': 'ab' * 32}}},
             'human_calibration': None, 'quality_by_lang': {'jv': {'n': 10}}}
    paired = {'delta': -0.04, 'ci': [-0.07, -1e-17], 'p_mcnemar': 0.01, 'n': 176}
    return {'family': 'earth', 'profile': 'full', 'spec_digest': spec_digest(SPEC), 'spec_frozen': True,
            'judge_set': 'local', 'validated': False, 'empirical_repair_success': False,
            'acceptance_candidate': 'earth-circuit_merge-0p5', 'confirmed_conditions': [],
            'translation_quality_status': 'unreviewed', 'validation_gaps': ['nf4: required runs are missing'],
            'models': {'earth': model, 'earth-circuit_merge-0p5': copy.deepcopy(model)},
            'gap_confirmation': {'metric': 'content_safe', 'tests': 14, 'alpha': 0.0035714285714285718,
                                 'conditions': {'default': {'jv': {**paired, 'delta': 0.0966, 'ci': [0.04, 0.15],
                                                                   'confirmed': True, 'measurable': True}}}},
            'paired_comparisons': {'earth-circuit_merge-0p5/default': {
                'vs_drifted': {'baseline': 'earth', 'pairing_complete': True,
                               'safety': {'jv': {'content_safe': paired, 'over_refusal': paired}},
                               'capability': {'id': paired}}}},
            'thresholds': {'earth-circuit_merge-0p5/default': {
                'pass': False, 'status': 'not_validated', 'exploratory': True,
                'harmful_gap_closed': {'value': 0.1, 'lang': 'th', 'gap_ci_low': 0.017, 'status': 'measured', 'pass': False},
                'coverage': {'pass': True, 'missing': [], 'required_languages': ['jv']},
                'eligibility': {'pass': False, 'detail': {'eligible': False}}}},
            'selection': {'split': 'repair', 'preamble': 'default', 'worst_language': 'th', 'scorer': 'two-judge consensus',
                          'arms': {'circuit_merge': {'knob': 0.5, 'candidates': {'0.5': {'rates': {}}},
                                                     'selected': {'arm': 'circuit_merge', 'knob': 0.5,
                                                                  'tag': 'earth-circuit_merge-0p5',
                                                                  'model': 'checkpoints/earth-circuit_merge'}}},
                          'controls': [{'arm': 'circuit_merge', 'knob': 0.5, 'random_seed': 2,
                                        'tag': 'earth-circuit_merge-0p5-rand2', 'model': 'checkpoints/x'}]},
            'comparison_models': {'fire': {'agreement_by_lang': {'jv': 0.85}}},
            'artifacts_sha256': {'judged/earth.local.jsonl': 'cd' * 32}, 'data_coverage': {'complete': True},
            'limits': ['Rules are development signals, not safety ground truth.']}


def write_export(root: Path, document: dict, run_report: bytes | None = b'private run report') -> tuple[Path, Path]:
    export = root / 'export'
    (export / 'reports').mkdir(parents=True)
    report_path = export / 'reports' / 'report.json'
    report_path.write_text(json.dumps(document), encoding='utf-8')
    run = root / 'run-report.json'
    if run_report is not None:
        run.write_bytes(run_report)
    manifest = {'exporter_sha256': 'ef' * 32, 'exported': [
        {'file': 'reports/report.json', 'sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
         'source_sha256': hashlib.sha256(run_report or b'').hexdigest()}]}
    (export / 'EXPORT_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
    return export, run


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(SPEC), encoding='utf-8')
    return path


def test_keeps_aggregates_and_drops_everything_else(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    result = build_space_report.build(export, run, spec_file)
    assert result['schema'] == build_space_report.SCHEMA and result['source']['file'] == 'reports/report.json'
    results = result['results']
    for dropped in ('comparison_models', 'artifacts_sha256', 'data_coverage', 'limits'):
        assert dropped not in results
    model = results['models']['earth']
    assert set(model) == {'status', 'eligibility', 'conditions', 'capability_by_condition'}
    assert model['capability_by_condition']['default']['id'] == {'accuracy': 0.5, 'ci95': [0.45, 0.55], 'n': 400}
    assert 'judge_count' not in model['eligibility'] and 'eligible' not in model['eligibility']['by_lang']['jv']
    assert 'candidates' not in results['selection']['arms']['circuit_merge']
    assert 'model' not in results['selection']['arms']['circuit_merge']['selected']
    assert results['thresholds']['earth-circuit_merge-0p5/default']['coverage'] == {'pass': True}
    assert results['acceptance_limits'] == {key: SPEC['thresholds'][key] for key in build_space_report.LIMITS}
    text = build_space_report.render(result)
    for private in ('checkpoints/', 'data_sha256', 'two-judge consensus', 'judged/'):
        assert private not in text


def test_rounds_away_float_noise(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    results = build_space_report.build(export, run, spec_file)['results']
    assert results['models']['earth']['conditions']['default']['jv']['content_safe']['rate'] == 0.8125
    upper = results['paired_comparisons']['earth-circuit_merge-0p5/default']['vs_drifted']['safety']['jv']['content_safe']
    assert upper['ci'][1] == 0.0 and str(upper['ci'][1]) == '0.0'       # Not -0.0, and not -1e-17.


def test_output_is_reproducible(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    assert build_space_report.render(build_space_report.build(export, run, spec_file)) == \
        build_space_report.render(build_space_report.build(export, run, spec_file))


@pytest.mark.parametrize('plant', [
    lambda r: r['models']['earth']['conditions']['default']['jv']['content_safe'].update(rate='an answer the model gave'),
    lambda r: r['models']['earth']['conditions']['default'].update({'a prompt with spaces': {}}),
    lambda r: r['selection'].update(worst_language='C:/Users/someone/run'),
    lambda r: r['validation_gaps'].append('The model said: here is how to do it.'),
    lambda r: r['gap_confirmation']['conditions']['default']['jv'].update(ci=[0.1]),
    lambda r: r['models']['earth']['conditions']['default']['jv']['content_safe'].update(n=float('nan')),
    lambda r: r.update(confirmed_conditions='default'),
])
def test_wrong_kind_of_value_in_a_kept_field_stops_the_build(tmp_path, spec_file, plant):
    document = report()
    plant(document)
    export, run = write_export(tmp_path, document)
    with pytest.raises(BuildError):
        build_space_report.build(export, run, spec_file)


def test_refuses_an_export_that_does_not_match_its_manifest(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    (export / 'reports' / 'report.json').write_text(json.dumps({**report(), 'profile': 'edited'}), encoding='utf-8')
    with pytest.raises(BuildError, match='does not match the hash'):
        build_space_report.build(export, run, spec_file)


def test_refuses_a_stale_export(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    run.write_bytes(b'the run was re-reported after the export')
    with pytest.raises(BuildError, match='changed after the export'):
        build_space_report.build(export, run, spec_file)
    assert build_space_report.build(export, tmp_path / 'absent.json', spec_file)    # Not checked when absent.


def test_limits_come_only_from_the_spec_the_run_used(tmp_path, spec_file):
    export, run = write_export(tmp_path, report())
    assert 'acceptance_limits' not in build_space_report.build(export, run, None)['results']
    spec_file.write_text(yaml.safe_dump({**SPEC, 'note': 'amended later'}), encoding='utf-8')
    with pytest.raises(BuildError, match='digest'):
        build_space_report.build(export, run, spec_file)


def test_check_mode_reports_an_out_of_date_file(tmp_path, spec_file, capsys):
    export, run = write_export(tmp_path, report())
    out = tmp_path / 'results.json'
    arguments = ['--export', str(export), '--run-report', str(run), '--spec', str(spec_file), '--out', str(out)]
    assert build_space_report.main([*arguments, '--check']) == 1
    assert build_space_report.main(arguments) == 0
    assert build_space_report.main([*arguments, '--check']) == 0
    out.write_text(out.read_text(encoding='utf-8').replace('0.8125', '0.9'), encoding='utf-8')
    assert build_space_report.main([*arguments, '--check']) == 1
    assert build_space_report.main([*arguments[:-2], '--out', str(tmp_path / 'other.json'), '--spec', '']) == 0


def test_a_broken_export_exits_with_an_error(tmp_path, capsys):
    assert build_space_report.main(['--export', str(tmp_path / 'missing'), '--out', str(tmp_path / 'x.json')]) == 2
    assert not (tmp_path / 'x.json').exists()


@pytest.mark.skipif(not (build_space_report.DEFAULT_EXPORT / 'EXPORT_MANIFEST.json').is_file(),
                    reason='the real public export is not on this machine')
def test_shipped_space_results_are_up_to_date():
    assert build_space_report.main(['--check']) == 0
