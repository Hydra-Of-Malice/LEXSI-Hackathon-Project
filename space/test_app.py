"""Tests for the public demo. None of them need model files, a GPU or the network.

    python -m pytest space/test_app.py -q

Most tests read the real `results.json` that ships with the Space, so they also check that file.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
import sys
import types

import gradio as gr
import pytest

import app


@pytest.fixture(scope='module')
def shipped():
    return json.loads(app.RESULTS_PATH.read_text(encoding='utf-8'))


@pytest.fixture
def raw(shipped):
    return copy.deepcopy(shipped)


@pytest.fixture
def report(raw):
    return raw['results']


@pytest.fixture(scope='module')
def view(shipped):
    return app.parse_report(copy.deepcopy(shipped))


@pytest.fixture(scope='module')
def prompts():
    return app.load_prompts()


def clean_report(report) -> dict:
    """A report that passes every check, to prove a banner is earned and not always on."""
    report = copy.deepcopy(report)
    report.update(validated=True, profile='full', spec_frozen=True, translation_quality_status='human_reviewed',
                  validation_gaps=[], empirical_repair_success=False)
    for model in report['models'].values():
        model['eligibility']['eligible'] = True
        for condition in model['conditions'].values():
            for cells in condition.values():
                cells.pop('content_safe', None)   # SPEC.md amendments 10 and 11 keep this axis provisional.
    return report


def cell(view, tag, preamble, lang):
    return next(row for row in view['rows'] if (row['tag'], row['preamble'], row['lang']) == (tag, preamble, lang))


def text_of(page: str) -> str:
    return re.sub(r'<[^>]+>', ' ', page)


EARTH = ['en', 'ar', 'sw', 'jv', 'bn', 'th', 'ko']


# The shipped results file -------------------------------------------------------------------------------------------

def test_results_file_is_the_built_aggregate(shipped):
    assert shipped['schema'] == app.RESULTS_SCHEMA
    assert shipped['source']['dataset'] == 'HydraOfMalice/tiny-aya-earth-langsafe-eval'
    assert shipped['source']['file'] == 'reports/report.json' and re.fullmatch(r'[0-9a-f]{64}', shipped['source']['sha256'])
    assert 'build_space_report.py' in shipped['notice'] and 'fake_data' not in json.dumps(shipped)


def test_results_file_holds_only_numbers_flags_and_short_identifiers(shipped):
    """Nothing in the file can be a model answer, a prompt or a local path."""
    name = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.\-/]{0,79}')
    gap = re.compile(r'[\w.\-/]+: [a-z0-9 ,/_\-]{1,120}')

    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                assert name.fullmatch(key), where
                walk(value, f'{where}.{key}')
        elif isinstance(node, list):
            for value in node:
                walk(value, where)
        elif isinstance(node, str):
            assert name.fullmatch(node) or (where.endswith('validation_gaps') and gap.fullmatch(node)), where
        else:
            assert node is None or isinstance(node, (bool, int, float)), where

    walk(shipped['results'], 'results')


def test_results_file_has_no_per_item_detail_or_paths(shipped):
    text = json.dumps(shipped['results'])
    for marker in ('per_item', 'data_sha256', 'artifacts_sha256', 'checkpoints/', ':\\\\', '/Users/', '/home/',
                   'response', 'completion', 'answer', 'prompt'):
        assert marker not in text, marker
    assert len(text) < 1_000_000


def test_results_cover_earths_seven_languages_and_every_arm(report):
    models = report['models']
    assert set(models['earth']['conditions']['default']) == set(EARTH)
    assert set(app.MODEL_REPOS) <= set(models) and {'global', 'earth'} <= set(models)
    assert report['acceptance_candidate'] == 'earth-circuit_merge-0p5'


def test_acceptance_limits_come_from_the_runs_spec(report):
    assert set(report['acceptance_limits']) == {'harmful_gap_closed_min', 'over_refusal_increase_max',
                                                'capability_drop_max'}


# Report parsing ------------------------------------------------------------------------------------------------------

def test_every_metric_is_read_with_its_interval(report, view):
    row = cell(view, 'earth', 'default', 'jv')
    source = report['models']['earth']['conditions']['default']['jv']
    for metric in ('content_safe', 'degenerate', 'harmful_refusal', 'harmful_compliance', 'over_refusal'):
        assert row[metric]['rate'] == source[metric]['rate']
        assert list(row[metric]['ci']) == source[metric]['ci']
        assert row[metric]['n'] == source[metric]['n']
    assert row['harmful_refusal']['unclear'] == source['harmful_refusal']['unclear']


def test_capability_uses_accuracy_ci95_and_its_own_condition(report, view):
    for preamble in ('default', 'none'):
        source = report['models']['earth']['capability_by_condition'][preamble]['ar']
        row = cell(view, 'earth', preamble, 'ar')
        assert row['capability']['rate'] == source['accuracy'] and list(row['capability']['ci']) == source['ci95']


def test_thai_capability_is_not_measured_rather_than_zero(view):
    assert cell(view, 'earth', 'default', 'th')['capability'] is None
    table = app.table_rows(app.filter_rows(view['rows'], 'th', 'default', 'fp16'))
    assert table and all(line[-1] == app.NOT_MEASURED for line in table)


def test_indonesian_is_a_capability_stand_in_and_nothing_else(view):
    """Global-MMLU-Lite has no Javanese, so `id` reaches the report through capability alone."""
    row = cell(view, 'earth', 'default', 'id')
    assert row['capability'] is not None and row['content_safe'] is None and row['degenerate'] is None
    assert cell(view, 'earth', 'default', 'jv')['capability'] is None
    assert 'stand-in for Javanese' in app.LANGUAGE_NAMES['id']


def test_languages_are_earths_seven_plus_the_stand_in(view):
    assert view['languages'] == [*EARTH, 'id'] and set(view['languages']) <= set(app.LANGUAGE_NAMES)


def test_precision_comes_from_the_tag_suffix():
    assert [app.precision_of(tag) for tag in ('earth', 'earth-q4', 'earth-x-nf4')] == ['fp16', 'q4', 'nf4']


def test_models_are_labelled_by_role():
    labels = {tag: app.describe_model(tag, 'earth', 'earth-circuit_merge-0p5') for tag in (
        'global-q4', 'earth', 'earth-circuit_merge-0p5-nf4', 'earth-resta-0p1', 'earth-steer-1p0',
        'earth-circuit_merge-0p5-rand2', 'mystery')}
    assert [(model['role'], model['label']) for model in labels.values()] == [
        ('reference', 'Global (reference)'), ('original', 'Earth (original)'),
        ('selected', 'Circuit merge λ=0.5 (selected repair)'), ('arm', 'ReSta α=0.1'), ('arm', 'Steering strength=1.0'),
        ('control', 'Random-layer control, seed 2'), ('other', 'mystery')]
    assert labels['earth-circuit_merge-0p5-nf4']['precision'] == 'nf4' and labels['earth-circuit_merge-0p5-nf4']['repo'] is None


def test_not_measurable_language_hides_judged_numbers_but_keeps_capability(report):
    """SPEC.md: a language below the agreement floor is listed as not measurable, not reported."""
    report['models']['earth-resta-0p1']['eligibility']['by_lang']['th']['status'] = 'not_measurable'
    report['models']['earth-resta-0p1']['capability_by_condition']['default']['th'] = {'accuracy': 0.5, 'n': 10}
    row = cell(app.parse_report(report), 'earth-resta-0p1', 'default', 'th')
    assert row['content_safe'] is row['harmful_refusal'] is row['harmful_compliance'] is row['over_refusal'] is None
    assert row['capability']['rate'] == 0.5 and row['degenerate'] is not None   # Neither uses the judges.
    line = app.table_rows([row])[0]
    assert line[5:7] == [app.NOT_MEASURABLE] * 2 and line[8:13] == [app.NOT_MEASURABLE] * 5
    assert line[7] == app.format_cell(row['degenerate'])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, 1.5, True, '0.5', None])
def test_invalid_rates_count_as_missing(bad):
    assert app.metric_cell({'rate': bad, 'ci': [0, 1], 'n': 5}) is None
    assert app.paired_cell({'delta': bad if bad != -0.1 else 2.0, 'ci': [0, 1]}) is None


@pytest.mark.parametrize('bad', [None, [0.1], [0.9, 0.1], ['a', 'b'], 'nope'])
def test_invalid_interval_is_shown_as_unavailable(bad):
    parsed = app.metric_cell({'rate': 0.5, 'ci': bad, 'n': 5})
    assert parsed['ci'] is None and 'CI unavailable' in app.format_cell(parsed)


def test_cell_format_shows_rate_interval_and_n():
    assert app.format_cell({'rate': 0.5, 'ci': (0.25, 0.75), 'n': 40}) == '50.0% [25.0–75.0], n=40'


def test_other_outcomes_stay_separate_from_the_axes(report, view):
    source = report['models']['earth']['conditions']['default']['jv']['content_safe']
    line = app.table_rows([cell(view, 'earth', 'default', 'jv')])[0]
    assert line[6] == ' / '.join(app.percent(source[key]) for key in ('unsafe', 'invalid', 'disputed'))


def test_table_headers_keep_the_measures_apart():
    assert app.TABLE_HEADERS[5].startswith('Content-safe') and 'primary, provisional' in app.TABLE_HEADERS[5]
    assert app.TABLE_HEADERS[6] == 'Harmful: unsafe / invalid / disputed'
    assert app.TABLE_HEADERS[8] == 'Harmful refusal' and app.TABLE_HEADERS[9] == 'Harmful compliance'
    assert len(app.COLUMN_WIDTHS) == len(app.TABLE_HEADERS)


def test_a_bare_report_also_loads(report):
    """LANGSAFE_REPORT_PATH may point at the dataset's own report.json rather than results.json."""
    assert app.parse_report(copy.deepcopy(report))['rows']


@pytest.mark.parametrize('content', ['{"id": "x", "response": "text"}\n{"id": "y"}', '[1, 2]', '{"family": "earth"}',
                                     'not json', '{"schema": "langsafe.space-results/1", "results": []}'])
def test_non_reports_are_rejected(tmp_path, content):
    path = tmp_path / 'report.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(app.ReportError):
        app.read_report(path)


@pytest.mark.parametrize('where', ['top', 'inside'])
def test_a_file_marked_as_invented_data_is_refused(tmp_path, raw, where):
    (raw if where == 'top' else raw['results'])['fake_data'] = True
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    view = app.load_report({app.REPORT_ENV: str(path)})
    assert view['error'] and view['rows'] == []


def test_only_aggregate_fields_reach_the_page(raw):
    secret = 'PER-ITEM-DETAIL'
    report = raw['results']
    report['models']['earth']['human_calibration'] = {'note': secret}
    report['models']['earth']['capability_by_condition']['default']['en']['per_item'] = [secret]
    report['models']['earth']['conditions']['default']['en']['content_safe']['note'] = secret
    report['selection']['note'] = secret
    report['paired_comparisons']['earth-resta-0p1/default']['vs_drifted']['note'] = secret
    view = app.parse_report(raw)
    pages = [app.banners_html(view), app.findings_html(view), app.gap_table_html(view), app.release_table_html(view),
             app.links_html(view), json.dumps(app.table_rows(view['rows'])),
             app.rate_chart_html(view, 'default', 'fp16', 'content_safe'),
             app.delta_chart_html(view, 'earth-resta-0p1', 'default', 'content_safe'),
             app.gates_html(view, 'earth-resta-0p1', 'default')]
    assert secret not in ''.join(pages)


# Choosing a report ---------------------------------------------------------------------------------------------------

def test_the_shipped_results_are_the_default():
    assert app.report_path({}) == app.RESULTS_PATH
    view = app.load_report({})
    assert view['error'] is None and view['rows']


def test_missing_results_show_an_error_and_no_numbers(monkeypatch, tmp_path):
    monkeypatch.setattr(app, 'RESULTS_PATH', tmp_path / 'absent.json')
    view = app.load_report({})
    assert view['rows'] == [] and 'RESULTS UNAVAILABLE' in app.banners_html(view)
    app.build_demo(view=view, live=False)


def test_environment_variable_wins(tmp_path):
    path = tmp_path / 'chosen.json'
    assert app.report_path({app.REPORT_ENV: str(path)}) == path


@pytest.mark.parametrize('content', [None, 'not json'])
def test_broken_configured_report_shows_an_error_and_hides_the_path(tmp_path, content):
    path = tmp_path / 'secret-path-report.json'
    if content is not None:
        path.write_text(content, encoding='utf-8')
    view = app.load_report({app.REPORT_ENV: str(path)})
    page = app.banners_html(view) + app.findings_html(view)
    assert view['rows'] == [] and 'RESULTS UNAVAILABLE' in page and 'secret-path-report' not in page
    app.build_demo(view=view, live=False)


# Findings: every number comes from the report ------------------------------------------------------------------------

def test_gap_findings_match_the_report(report, view):
    expected = sorted((preamble, lang) for preamble, cells in report['gap_confirmation']['conditions'].items()
                      for lang, item in cells.items() if item['confirmed'])
    found = app.gap_findings(view)
    assert sorted(item[:2] for item in found['confirmed']) == expected and expected
    assert found['tests'] == report['gap_confirmation']['tests']
    for preamble, lang, item in found['suggestive']:
        assert item['ci'][0] > 0 and not item['confirmed']


def test_findings_show_the_confirmed_gap_as_measured(report, view):
    page = text_of(app.findings_html(view))
    for preamble, lang, item in app.gap_findings(view)['confirmed']:
        source = report['gap_confirmation']['conditions'][preamble][lang]
        assert app.LANGUAGE_NAMES[lang] in page and app.points(source['delta']) in page
        assert app.interval_text(source['ci']) in page


def test_no_repair_success_is_claimed_when_the_report_says_false(report, view):
    assert report['empirical_repair_success'] is False and view['repair_success'] is False
    page = text_of(app.findings_html(view))
    assert 'No success claimed' in page and 'exploratory' in page
    assert app.LANGUAGE_NAMES[report['selection']['worst_language']] in page


def test_repair_success_needs_a_literal_true(report):
    report['empirical_repair_success'] = 'yes'
    assert app.parse_report(copy.deepcopy(report))['repair_success'] is False
    report['empirical_repair_success'] = True
    assert 'Success claimed' in app.findings_html(app.parse_report(report))


def test_release_rule_matches_decision_d4(view):
    """D4: all published arms pass; of the controls, random layers seed 2 fails in Swahili without the preamble."""
    checks = {item['tag']: item for item in app.release_check(view)}
    assert set(app.MODEL_REPOS) <= set(checks)
    assert all(checks[tag]['passes'] for tag in app.MODEL_REPOS)
    failures = {(item['tag'], preamble, lang) for item in checks.values() for preamble, lang, _ in item['failures']}
    assert failures == {('earth-circuit_merge-0p5-rand2', 'none', 'sw')}
    page = text_of(app.findings_html(view))
    assert f'{len(app.MODEL_REPOS)} of {len(app.MODEL_REPOS)} published releases pass' in page


@pytest.mark.parametrize('high, passes', [(0.0, True), (0.01, True), (-0.001, False)])
def test_release_rule_fails_only_when_the_interval_is_wholly_below_zero(view, high, passes):
    edited = copy.deepcopy(view)
    edited['paired'] = {'earth-resta-0p1': {'none': {'baseline': 'earth', 'metrics': {
        'content_safe': {'sw': {'delta': -0.04, 'ci': (-0.07, high), 'p': 0.01, 'n': 176, 'confirmed': False}},
        'over_refusal': {}, 'capability': {}}}}}
    assert app.release_check(edited)[0]['passes'] is passes


def test_release_table_names_every_model_and_its_status(view):
    page = text_of(app.release_table_html(view))
    for item in app.release_check(view):
        assert item['label'] in page
    assert 'control, kept private' in page and '✗ fails' in page and 'Q4_K_M GGUF' in page


def test_paired_comparisons_use_same_precision_baselines(view):
    assert view['paired']['earth-circuit_merge-0p5-q4']['default']['baseline'] == 'earth-q4'
    assert view['paired']['earth-circuit_merge-0p5-nf4']['none']['baseline'] == 'earth-nf4'
    assert view['paired']['earth-resta-0p1']['default']['baseline'] == 'earth'


def test_gates_show_the_spec_limits_and_what_was_measured(report, view):
    page = text_of(app.gates_html(view, 'earth-circuit_merge-0p5', 'default'))
    gate = report['thresholds']['earth-circuit_merge-0p5/default']
    assert 'at most 2 points' in page and 'at most 1 point' in page and 'at least 50%' in page
    assert f"closed {100 * gate['harmful_gap_closed']['value']:.0f}%" in page
    assert ('does not pass' in page) is (gate['pass'] is False) and 'exploratory' in page


def test_gates_without_limits_still_read(view):
    edited = {**view, 'limits': {}}
    assert 'Closes enough of the' in app.gates_html(edited, 'earth-circuit_merge-0p5', 'default')


def test_repair_choices_put_the_selected_repair_first(view):
    tags = [tag for _, tag in app.repair_choices(view)]
    assert tags[:3] == ['earth-circuit_merge-0p5', 'earth-circuit_merge-0p5-q4', 'earth-circuit_merge-0p5-nf4']
    assert tags[-3:] == [f'earth-circuit_merge-0p5-rand{seed}' for seed in (1, 2, 3)] and len(tags) == 10


# Charts --------------------------------------------------------------------------------------------------------------

def test_rate_chart_has_a_row_per_language_and_series(view):
    page = app.rate_chart_html(view, 'default', 'fp16', 'content_safe')
    assert page.count('<g class="ls-row">') == len(EARTH) * len(app.SERIES)
    assert '<svg' in page and '<details>' in page and 'Global (reference)' in page and 'Earth (original)' in page


def test_rate_chart_numbers_match_the_report(view):
    page = app.rate_chart_html(view, 'none', 'q4', 'content_safe')
    for tag in ('global-q4', 'earth-q4', 'earth-circuit_merge-0p5-q4'):
        for lang in EARTH:
            assert app.cell_text(cell(view, tag, 'none', lang)['content_safe']) in page


def test_capability_chart_shows_the_stand_in_and_no_thai_value(view):
    page = text_of(app.rate_chart_html(view, 'default', 'fp16', 'capability'))
    assert 'Indonesian (stand-in for Javanese)' in page and f'Thai: {app.NOT_MEASURED}' in page


def test_delta_chart_flags_the_failing_control(view):
    page = app.delta_chart_html(view, 'earth-circuit_merge-0p5-rand2', 'none', 'content_safe')
    assert page.count('significantly worse') == 2      # Once on the chart, once in its table.
    assert 'Random-layer control, seed 2 minus Earth (original)' in page
    assert 'significantly worse' not in app.delta_chart_html(view, 'earth-circuit_merge-0p5', 'default', 'content_safe')


def test_delta_chart_reads_direction_by_measure(view):
    """A rise in over-refusal is worse, a rise in content-safe is better."""
    edited = copy.deepcopy(view)
    rise = {'delta': 0.05, 'ci': (0.01, 0.09), 'p': 0.01, 'n': 150, 'confirmed': False}
    comparison = edited['paired']['earth-resta-0p1']['default']['metrics']
    comparison['over_refusal']['en'] = comparison['content_safe']['en'] = rise
    assert 'significantly worse' in app.delta_chart_html(edited, 'earth-resta-0p1', 'default', 'over_refusal')
    assert 'significantly better' in app.delta_chart_html(edited, 'earth-resta-0p1', 'default', 'content_safe')


def test_unknown_selection_says_so(view):
    assert 'No paired comparison' in app.delta_chart_html(view, 'not-a-model', 'default', 'content_safe')
    assert 'No results' in app.rate_chart_html(app.failed_view(), 'default', 'fp16', 'content_safe')


def test_axes_hold_the_data():
    low, high, step = app.rate_domain([0.62, 0.99])
    assert 0 <= low <= 0.62 and 0.99 <= high <= 1 and step in (0.05, 0.1)
    low, high, _ = app.delta_domain([-0.091, 0.034])
    assert low == -high and high >= 0.091


def test_series_use_the_validated_palette_and_distinct_markers():
    assert [fallback for *_, fallback in app.SERIES] == ['#2a78d6', '#eb6834', '#1baf7a']
    assert len({shape for _, _, shape, _, _ in app.SERIES}) == len(app.SERIES)
    assert '.dark .ls-viz' in app.PAGE_CSS


def test_chart_and_table_text_is_escaped(view):
    edited = copy.deepcopy(view)
    edited['paired']['<script>x</script>'] = edited['paired']['earth-resta-0p1']
    page = app.release_table_html(edited) + app.delta_chart_html(edited, '<script>x</script>', 'default', 'content_safe')
    assert '<script>' not in page and '&lt;script&gt;' in page


# Banners -------------------------------------------------------------------------------------------------------------

def test_real_report_is_provisional_and_says_why(view):
    page = app.banners_html(view)
    assert view['provisional'] and 'PROVISIONAL' in page
    assert 'Translations have not been reviewed' in page and 'Content-safe numbers are outside' in page


def test_clean_report_shows_no_warning_banner(report):
    view = app.parse_report(clean_report(report))
    assert view['provisional'] is False and 'PROVISIONAL' not in app.banners_html(view)


@pytest.mark.parametrize('change', [{'validated': False}, {'validated': 'true'}, {'validated': 1}, {'validated': None},
                                    {'profile': 'development'}, {'spec_frozen': False},
                                    {'translation_quality_status': 'unreviewed'}, {'validation_gaps': ['x: missing']}])
def test_provisional_banner_appears_when_anything_is_unvalidated(report, change):
    edited = clean_report(report)
    edited.update(change)
    view = app.parse_report(edited)
    assert view['provisional'] is True and 'PROVISIONAL' in app.banners_html(view)


@pytest.mark.parametrize('field', ['validated', 'profile', 'spec_frozen', 'translation_quality_status', 'validation_gaps'])
def test_absent_fields_fail_closed(report, field):
    edited = clean_report(report)
    edited.pop(field)
    assert app.parse_report(edited)['provisional'] is True


def test_ineligible_model_keeps_the_banner_even_if_validated_is_true(report):
    edited = clean_report(report)
    edited['models']['earth']['eligibility']['eligible'] = False
    assert app.parse_report(edited)['provisional'] is True


def test_content_safe_numbers_keep_the_banner_even_in_a_validated_report(report):
    edited = clean_report(report)
    edited['models']['earth']['conditions']['default']['en']['content_safe'] = {'rate': 0.5, 'ci': [0.4, 0.6], 'n': 9}
    view = app.parse_report(edited)
    assert view['provisional'] is True and 'Content-safe numbers are outside' in app.banners_html(view)


def test_model_without_results_is_listed_not_dropped(report):
    report['models']['earth-extra-nf4'] = {'status': 'missing_judgments'}
    view = app.parse_report(report)
    assert view['missing_models'] == [{'tag': 'earth-extra-nf4', 'status': 'missing_judgments'}]
    assert 'earth-extra-nf4' in app.banners_html(view)


def test_banner_text_sets_its_own_colour_on_every_element(view, prompts):
    """Gradio's prose styles otherwise recolour it, which left the amber banner unreadable in dark mode."""
    for page in (app.banners_html(view), app.translation_notice(prompts, 'jv')):
        tags = re.findall(r'<(?:div|strong)\b[^>]*>', page)
        assert tags and all(re.search(r'color:#[0-9a-f]{6} !important', tag) for tag in tags)


def test_report_text_is_escaped(report):
    report['profile'] = '<script>alert(1)</script>'
    report['models']['<b>x</b>'] = {'status': '<i>y</i>'}
    page = app.banners_html(app.parse_report(report))
    assert '<script>' not in page and '<b>x' not in page and '<i>y' not in page


# Filtering the full table --------------------------------------------------------------------------------------------

def test_all_filters_keep_everything(view):
    assert len(app.filter_rows(view['rows'])) == len(view['rows']) > 0


@pytest.mark.parametrize('key, value, argument', [('lang', 'jv', 'language'), ('preamble', 'none', 'preamble'),
                                                  ('precision', 'q4', 'precision')])
def test_each_filter_selects_only_its_value(view, key, value, argument):
    selected = app.filter_rows(view['rows'], **{argument: value})
    assert selected and len(selected) < len(view['rows']) and all(row[key] == value for row in selected)


def test_filters_combine(view):
    selected = app.filter_rows(view['rows'], 'en', 'default', 'q4', 'main')
    assert {row['tag'] for row in selected} == {'global-q4', 'earth-q4', 'earth-circuit_merge-0p5-q4'}


@pytest.mark.parametrize('role, tags', [('control', {f'earth-circuit_merge-0p5-rand{seed}' for seed in (1, 2, 3)}),
                                        ('arm', {'earth-resta-0p1', 'earth-circuit-0p1', 'earth-merge_back-0p5',
                                                 'earth-steer-1p0'})])
def test_model_filter_selects_its_group(view, role, tags):
    assert {row['tag'] for row in app.filter_rows(view['rows'], role=role)} == tags


def test_empty_selection_says_so(view):
    selected = app.filter_rows(view['rows'], language='id', role='control', preamble='none', precision='q4')
    assert selected == [] and app.summary_text(selected, view) == 'No measurements for this selection.'


def test_table_shows_every_metric(view):
    line = app.table_rows(app.filter_rows(view['rows'], 'en', 'default', 'fp16', 'main'))[0]
    assert len(line) == len(app.TABLE_HEADERS) and line[0] == 'Global (reference)'
    assert all(re.search(r'\d+\.\d% \[\d+\.\d–\d+\.\d\], n=\d+', line[index]) for index in (5, 7, 8, 9, 11, 13))


# The prompt list -----------------------------------------------------------------------------------------------------

def test_prompt_languages_are_earths_seven(prompts):
    assert list(prompts['languages']) == EARTH
    assert all('families' not in meta for meta in prompts['languages'].values())     # Earth is the only pair now.


def test_every_prompt_exists_in_every_language(prompts):
    for item in prompts['prompts']:
        assert set(item['text']) == set(prompts['languages'])
        assert all(text.strip() for text in item['text'].values())


def test_translations_are_written_in_their_own_script(prompts):
    scripts = {'bn': 'ঀ-৿', 'ar': '؀-ۿ', 'th': '฀-๿', 'ko': '가-힣'}
    for item in prompts['prompts']:
        for language, block in scripts.items():
            assert re.search(f'[{block}]', item['text'][language]), (item['id'], language)


def test_latin_script_translations_are_not_english_left_in_place(prompts):
    for item in prompts['prompts']:
        for language in ('sw', 'jv'):
            assert item['text'][language] != item['text']['en'], (item['id'], language)


def test_prompt_ids_are_unique(prompts):
    ids = [item['id'] for item in prompts['prompts']]
    assert len(ids) == len(set(ids)) == 6


def test_only_everyday_benign_prompts_remain(prompts):
    """The 2026-11-11 review removed every look-alike: they borrow violent verbs on purpose."""
    assert {item['category'] for item in prompts['prompts']} == {'benign'}
    assert all(item['why_safe'].strip() and item['id'].startswith('benign-') for item in prompts['prompts'])


def test_no_violent_or_harmful_wording_slipped_in(prompts):
    """A tripwire for later edits. The real control is a person reading the list."""
    banned = (r'\b(kill\w*|gut|blow\w*|shoot\w*|crush\w*|hit|attack\w*|bomb\w*|explosive\w*|poison\w*|suicide|'
              r'self-harm|weapon\w*|gun\w*|knife|hack\w*|malware|ransomware|drug\w*|meth|nude|steal\w*|launder\w*|'
              r'murder\w*|die|dead|blood)\b')
    for item in prompts['prompts']:
        assert not re.search(banned, item['text']['en'], re.I), item['id']


def test_non_english_languages_are_flagged_until_a_named_person_reviews_them(prompts):
    for code, meta in prompts['languages'].items():
        if code == 'en':
            assert meta['review_status'] == 'source'
            continue
        assert meta['review_status'] in {'unreviewed_machine_translation', 'reviewed'}
        assert (meta['review_status'] == 'reviewed') == bool(meta['reviewed_by'])
        if meta['review_status'] != 'reviewed':
            assert 'UNREVIEWED TRANSLATION' in app.translation_notice(prompts, code)
    assert 'NOT been checked by a native reader' in prompts['translation_notice']


def test_notice_is_on_the_page_for_translations_and_not_for_english(prompts):
    assert 'UNREVIEWED' not in app.prompt_html(prompts, 'en', 'benign-tea')
    assert 'UNREVIEWED' in app.prompt_html(prompts, 'jv', 'benign-tea')


def test_marking_reviewed_without_a_name_keeps_the_notice(prompts):
    edited = copy.deepcopy(prompts)
    edited['languages']['jv']['review_status'] = 'reviewed'
    assert 'UNREVIEWED' in app.translation_notice(edited, 'jv')
    edited['languages']['jv']['reviewed_by'] = 'A Reader'
    assert app.translation_notice(edited, 'jv') == ''


# The public page: results only, no model, no free text ---------------------------------------------------------------

PUBLIC_COMPONENTS = (gr.Markdown, gr.HTML, gr.Dropdown, gr.Dataframe)
LIVE_COMPONENTS = (*PUBLIC_COMPONENTS, gr.Radio, gr.Button)


class ExplodingFactory:
    """Stands in for model loading in tests where reaching a model would be the bug."""

    def __call__(self, path):
        raise AssertionError('a model was reached')


@pytest.fixture(scope='module')
def demo(view):
    return app.build_demo(view=view, live=False)


@pytest.fixture
def live_demo(view, prompts):
    return app.build_demo(view=view, prompts=prompts, pool=app.ModelPool(factory=ExplodingFactory(), env={}), live=True)


def components(demo):
    return [block for block in demo.blocks.values() if isinstance(block, gr.components.Component)]


def test_live_comparison_is_off_unless_explicitly_enabled():
    assert app.live_enabled({}) is False and app.live_enabled({app.LIVE_ENV: 'true'}) is False
    assert app.live_enabled({app.LIVE_ENV: '1'}) is True


def test_space_requirements_leave_out_the_model_runtime():
    space = app.RESULTS_PATH.parent
    assert 'llama' not in (space / 'requirements.txt').read_text(encoding='utf-8').replace('# llama-cpp-python is', '')
    assert 'llama-cpp-python==' in (space / 'requirements-live.txt').read_text(encoding='utf-8')


def test_default_page_has_only_display_and_selection_components(monkeypatch, view):
    monkeypatch.delenv(app.LIVE_ENV, raising=False)
    page = app.build_demo(view=view)
    unexpected = [type(block).__name__ for block in components(page) if not isinstance(block, PUBLIC_COMPONENTS)]
    assert unexpected == []
    assert not any(fn.name == 'side_by_side' for fn in page.fns.values())


def test_live_page_adds_only_a_choice_and_a_button(live_demo):
    unexpected = [type(block).__name__ for block in components(live_demo) if not isinstance(block, LIVE_COMPONENTS)]
    assert unexpected == []


def test_no_dropdown_takes_custom_values_and_the_table_is_read_only(demo):
    dropdowns = [block for block in components(demo) if isinstance(block, gr.Dropdown)]
    assert len(dropdowns) == 10
    assert all(block.allow_custom_value is False and block.filterable is False for block in dropdowns)
    assert all(block.interactive is False for block in components(demo) if isinstance(block, gr.Dataframe))


def test_the_model_column_stays_visible_when_the_table_scrolls():
    assert 'position: sticky' in app.LAUNCH_OPTIONS['css'] and '.body-cell.first-column' in app.LAUNCH_OPTIONS['css']


def test_mcp_and_uploads_are_off():
    assert app.LAUNCH_OPTIONS['mcp_server'] is False
    assert gr.utils._parse_file_size(app.LAUNCH_OPTIONS['max_file_size']) == 1


@pytest.mark.parametrize('live', [False, True])
def test_no_event_is_published_as_an_api(view, prompts, live):
    page = app.build_demo(view=view, prompts=prompts, pool=app.ModelPool(factory=ExplodingFactory(), env={}), live=live)
    events = [fn for fn in page.fns.values() if fn.fn is not None]
    assert events and all(fn.api_visibility == 'private' for fn in events)


def test_page_shows_the_banner_findings_and_results(demo):
    page = ' '.join(str(block.value) for block in components(demo) if isinstance(block, gr.HTML))
    assert 'PROVISIONAL' in page and 'tests confirmed' in page and '<svg' in page and 'Release' in page
    table = next(block for block in components(demo) if isinstance(block, gr.Dataframe))
    assert len(table.value['data']) == len(app.load_report({})['rows'])


def test_links_point_at_the_published_artifacts(view):
    page = app.links_html(view)
    for url in (app.DATASET_URL, app.CODE_URL, *app.MODEL_REPOS.values(), *app.BASE_MODEL_URLS.values()):
        assert f'href="{url}"' in page
    anchors = re.findall(r'<a\b[^>]*>', page)
    assert anchors and all('target="_blank"' in tag and 'noopener' in tag for tag in anchors)


def test_prose_states_no_result_of_its_own():
    """Results come from the report. The fixed text may name method constants (95% intervals, top 20% of layers) only."""
    for text in (app.INTRO, app.LANGUAGE_NOTES, app.REPAIR_NOTES, app.RULE_TEXT, app.EXPLORER_NOTES, app.ABOUT_TEXT,
                 app.SIDE_NOTES):
        found = set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|points?|pp)', text))
        assert found <= {'95%', '20%'}, found


def test_readme_has_valid_space_front_matter():
    readme = (app.RESULTS_PATH.parent / 'README.md').read_text(encoding='utf-8')
    head = readme.split('---')[1]
    fields = dict(line.split(':', 1) for line in head.strip().splitlines() if ':' in line and not line.startswith('#'))
    fields = {key.strip(): value.strip().strip('"') for key, value in fields.items()}
    assert fields['sdk'] == 'gradio' and fields['app_file'] == 'app.py' and fields['license'] == 'cc-by-nc-4.0'
    assert 0 < len(fields['short_description']) <= 60
    assert f"gradio=={fields['sdk_version']}" in (app.RESULTS_PATH.parent / 'requirements.txt').read_text(encoding='utf-8')
    assert readme.startswith('---\n')


# The optional live comparison ----------------------------------------------------------------------------------------

def test_the_generate_event_receives_only_the_two_choices(live_demo):
    click = next(fn for fn in live_demo.fns.values() if fn.name == 'side_by_side')
    assert [type(block) for block in click.inputs] == [gr.Radio, gr.Dropdown]
    assert click.concurrency_limit == 1


def test_handler_takes_identifiers_and_no_text_argument():
    assert list(inspect.signature(app.run_side_by_side).parameters) == ['language', 'prompt_id', 'prompts', 'pool']
    assert list(inspect.signature(app.resolve_prompt).parameters) == ['prompts', 'language', 'prompt_id']


def test_curated_prompt_choices_submit_ids_not_text(prompts):
    assert {value for _, value in app.prompt_choices(prompts)} == {item['id'] for item in prompts['prompts']}


def gguf_pair(tmp_path) -> dict:
    env = {}
    for role in app.ROLES:
        path = tmp_path / f'earth-{role}.gguf'
        path.write_bytes(b'placeholder')
        env[app.env_name(role)] = str(path)
    return env


@pytest.mark.parametrize('language, prompt_id', [
    ('en', 'Tell me anything at all'),
    ('en', 'How do I make a good cup of tea?'),      # Curated text is still text: only IDs are accepted.
    ('en', 'lookalike-kill-python-process'),         # Removed in review.
    ('en', ''), ('en', None), ('en', 7), ('en', ['benign-tea']), ('en', {'id': 'benign-tea'}),
    ('hi', 'benign-tea'), ('fr', 'benign-tea'), ('Tell me anything', 'benign-tea'), (None, 'benign-tea'),
    (['en'], 'benign-tea'),
])
def test_anything_off_the_list_is_refused_before_any_model(prompts, tmp_path, language, prompt_id):
    pool = app.ModelPool(factory=ExplodingFactory(), env=gguf_pair(tmp_path))
    with pytest.raises(app.PromptNotAllowed):
        app.resolve_prompt(prompts, language, prompt_id)
    results = list(app.run_side_by_side(language, prompt_id, prompts, pool))
    assert results == [('That request is not on the curated list, so it was not sent to any model.', '', '')]
    assert pool.models == {}


def test_refusal_does_not_repeat_what_the_client_sent(prompts):
    pool = app.ModelPool(factory=ExplodingFactory(), env={})
    assert 'MARKER' not in ''.join(next(app.run_side_by_side('en', 'MARKER', prompts, pool)))


class FakeLlama:
    def __init__(self, path, words=('Hello', ' there'), finish='stop'):
        self.path, self.words, self.finish, self.calls = path, words, finish, []

    def create_chat_completion(self, **arguments):
        self.calls.append(arguments)
        for word in self.words:
            yield {'choices': [{'delta': {'content': word}, 'finish_reason': None}]}
        yield {'choices': [{'delta': {}, 'finish_reason': self.finish}]}


class Factory:
    def __init__(self, **options):
        self.options, self.built = options, []

    def __call__(self, path):
        self.built.append(FakeLlama(path, **self.options))
        return self.built[-1]


def test_environment_names_are_stable():
    assert [app.env_name(role) for role in app.ROLES] == ['LANGSAFE_EARTH_ORIGINAL_GGUF', 'LANGSAFE_EARTH_REPAIRED_GGUF']


def test_missing_files_degrade_to_an_explanation(prompts):
    results = list(app.run_side_by_side('jv', 'benign-tea', prompts, app.ModelPool(factory=ExplodingFactory(), env={})))
    assert len(results) == 1 and results[0][1:] == ('', '') and 'unavailable' in results[0][0]
    assert 'LANGSAFE_EARTH_ORIGINAL_GGUF' in results[0][0] and 'LANGSAFE_EARTH_REPAIRED_GGUF' in results[0][0]


def test_live_tab_explains_itself_before_anyone_clicks(live_demo):
    text = ' '.join(str(block.value) for block in components(live_demo) if isinstance(block, gr.Markdown))
    assert 'LANGSAFE_EARTH_ORIGINAL_GGUF' in text


def test_one_file_of_a_pair_is_not_enough(prompts, tmp_path):
    env = gguf_pair(tmp_path)
    env.pop(app.env_name('repaired'))
    pool = app.ModelPool(factory=ExplodingFactory(), env=env)
    assert 'LANGSAFE_EARTH_REPAIRED_GGUF' in next(app.run_side_by_side('en', 'benign-tea', prompts, pool))[0]


def test_paths_that_do_not_exist_are_reported_without_showing_the_path(prompts):
    env = {app.env_name(role): f'/private/secret/{role}.gguf' for role in app.ROLES}
    message = next(app.run_side_by_side('en', 'benign-tea', prompts, app.ModelPool(factory=ExplodingFactory(), env=env)))[0]
    assert 'does not point to a file' in message and 'secret' not in message


def test_same_file_twice_is_refused(prompts, tmp_path):
    path = tmp_path / 'one.gguf'
    path.write_bytes(b'placeholder')
    env = {app.env_name(role): str(path) for role in app.ROLES}
    pool = app.ModelPool(factory=ExplodingFactory(), env=env)
    assert 'nothing to compare' in next(app.run_side_by_side('en', 'benign-tea', prompts, pool))[0]


def test_load_failure_is_reported_without_details(prompts, tmp_path):
    def broken(path):
        raise RuntimeError('unknown pre-tokenizer at /private/secret/model.gguf')

    final = list(app.run_side_by_side('en', 'benign-tea', prompts, app.ModelPool(factory=broken, env=gguf_pair(tmp_path))))[-1]
    assert 'could not be run' in final[0] and 'secret' not in ''.join(final)


def test_missing_llama_cpp_degrades_rather_than_crashing(prompts, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'llama_cpp', None)
    pool = app.ModelPool(env=gguf_pair(tmp_path))
    assert 'could not be run' in list(app.run_side_by_side('en', 'benign-tea', prompts, pool))[-1][0]


@pytest.mark.parametrize('language', EARTH)
def test_both_models_answer_the_curated_text_with_greedy_decoding(prompts, tmp_path, language):
    factory = Factory()
    env = gguf_pair(tmp_path)
    final = list(app.run_side_by_side(language, 'benign-tea', prompts, app.ModelPool(factory=factory, env=env)))[-1]
    assert 'Hello there' in final[1] and 'Hello there' in final[2] and final[0].startswith('Done.')
    assert [model.path for model in factory.built] == [env['LANGSAFE_EARTH_ORIGINAL_GGUF'], env['LANGSAFE_EARTH_REPAIRED_GGUF']]
    expected = next(item for item in prompts['prompts'] if item['id'] == 'benign-tea')['text'][language]
    for model in factory.built:
        call = model.calls[0]
        assert call['messages'] == [{'role': 'user', 'content': expected}]
        assert (call['temperature'], call['top_k'], call['repeat_penalty']) == (0.0, 1, 1.0)
        assert call['max_tokens'] == app.DEFAULT_MAX_NEW_TOKENS


def test_answers_stream_as_they_arrive(prompts, tmp_path):
    pool = app.ModelPool(factory=Factory(), env=gguf_pair(tmp_path))
    originals = [result[1] for result in app.run_side_by_side('en', 'benign-tea', prompts, pool)]
    assert any('Hello<' in text for text in originals) and 'Hello there' in originals[-1]


def test_repeat_clicks_reuse_finished_answers(prompts, tmp_path):
    factory = Factory()
    pool = app.ModelPool(factory=factory, env=gguf_pair(tmp_path))
    first = list(app.run_side_by_side('en', 'benign-tea', prompts, pool))[-1]
    second = list(app.run_side_by_side('en', 'benign-tea', prompts, pool))[-1]
    assert first == second and all(len(model.calls) == 1 for model in factory.built) and len(factory.built) == 2


def test_each_model_is_loaded_once_across_prompts(prompts, tmp_path):
    factory = Factory()
    pool = app.ModelPool(factory=factory, env=gguf_pair(tmp_path))
    for prompt_id in ('benign-tea', 'benign-monsoon'):
        list(app.run_side_by_side('en', prompt_id, prompts, pool))
    assert len(factory.built) == 2 and all(len(model.calls) == 2 for model in factory.built)


def test_model_output_cannot_inject_markup(prompts, tmp_path):
    pool = app.ModelPool(factory=Factory(words=('<script>alert(1)</script>',)), env=gguf_pair(tmp_path))
    final = list(app.run_side_by_side('en', 'benign-tea', prompts, pool))[-1]
    assert '<script>' not in final[1] and '&lt;script&gt;' in final[1]


def test_truncated_answers_say_so(prompts, tmp_path):
    pool = app.ModelPool(factory=Factory(finish='length'), env=gguf_pair(tmp_path))
    assert 'Cut off at 200 new tokens' in list(app.run_side_by_side('en', 'benign-tea', prompts, pool))[-1][1]


def test_arabic_answers_set_their_direction_automatically():
    assert 'dir="auto"' in app.answer_html('مرحبا', 'ar') and 'lang="ar"' in app.answer_html('مرحبا', 'ar')


def test_models_stay_on_the_cpu(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, **arguments):
            captured.update(arguments)

    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=Recorder))
    app.load_llama('model.gguf', {app.THREADS_ENV: '3'})
    assert captured['n_gpu_layers'] == 0 and captured['n_threads'] == 3 and captured['model_path'] == 'model.gguf'
    assert captured['kv_overrides'] == {'tokenizer.ggml.eos_token_id': 3}     # Otherwise a bad export never stops.
    assert captured['n_batch'] == 128


@pytest.mark.parametrize('value, expected', [('64', 64), ('9999', 256), ('1', 16), ('many', 200)])
def test_token_limit_setting_stays_within_the_spec(value, expected):
    assert app.ModelPool(factory=ExplodingFactory(), env={app.MAX_TOKENS_ENV: value}).max_tokens == expected
