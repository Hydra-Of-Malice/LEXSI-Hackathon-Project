"""Aggregate-only experiment reports; measurement eligibility stays explicit."""
from __future__ import annotations

import json
from pathlib import Path

from .config import capability_languages, load_spec, spec_digest
from .eval.axes import compare, safety_table
from .eval.report import check_thresholds, markdown_table, reporting_eligibility
from .eval.stats import mcnemar_exact, paired_diff_ci, rate_ci
from .experiment import atomic_json, capability_path, file_sha256


def primary_metric(spec: dict) -> str:
    axis = spec.get('experiment', {}).get('primary_safety_axis', 'refusal')
    return {'content': 'content_safe', 'refusal': 'harmful_refusal'}[axis]


def _sampled(cell: dict | None) -> str:
    return f"{cell['sampled']} of {cell['total']}" if cell else 'not recorded'


def _accuracy(value: float | None, n: int | None) -> str:
    return f'{100 * value:.1f}% (n={n})' if value is not None else 'not labelled'


def _read(path: Path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def report_dependencies(runs: Path, tags: list[str], judge_set: str, conditions: list[str]) -> list[Path]:
    paths = [runs / 'data' / 'manifest.json', runs / 'selection.json']
    for tag in tags:
        judged = runs / 'judged' / (f'{tag}.jsonl' if judge_set == 'router' else f'{tag}.{judge_set}.jsonl')
        paths.extend([judged, judged.with_suffix('.summary.json'), runs / 'gens' / f'{tag}.jsonl',
                      runs / 'calibration' / f'{tag}.{judge_set}.json',
                      runs / 'calibration' / f'{tag}.router-comparison.json'])
        paths.extend(capability_path(runs, tag, pre) for pre in conditions)
    return paths


def build_report(tags: list[str], family: str, judge_set: str, runs: Path, output: Path | None = None) -> Path:
    spec = load_spec()
    languages = spec['languages'][family]
    cap_languages = capability_languages(spec, family)
    models = {}
    artifacts = {}
    tables = {}
    validation_gaps = []
    evaluation_keys = {}
    judged_rows = {}
    resamples = spec['stats']['bootstrap_resamples']
    for tag in tags:
        judged_path = runs / 'judged' / (f'{tag}.jsonl' if judge_set == 'router' else f'{tag}.{judge_set}.jsonl')
        if not judged_path.exists():
            validation_gaps.append(f'{tag}: missing judgments')
            models[tag] = {'status': 'missing_judgments', 'eligibility': {'eligible': False}, 'conditions': {}, 'capability': {}}
            continue
        rows = [json.loads(line) for line in judged_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        judged_rows[tag] = rows
        keys = sorted((r['id'], r['lang'], r['kind'], r['preamble']) for r in rows)
        evaluation_keys[tag] = keys
        if len(keys) != len(set(keys)):
            validation_gaps.append(f'{tag}: duplicate evaluation identities')
        for pre in spec['preamble_conditions']:
            for lang in languages:
                for kind in ('harmful', 'benign'):
                    if not any(r['preamble'] == pre and r['lang'] == lang and r['kind'] == kind for r in rows):
                        validation_gaps.append(f'{tag}: missing {pre}/{lang}/{kind}')
        if any(row.get('split', 'heldout') != 'heldout' for row in rows):
            raise ValueError('Headline reports accept held-out generations only')
        summary = _read(judged_path.with_suffix('.summary.json'), {})
        calibration = _read(runs / 'calibration' / f'{tag}.{judge_set}.json')
        router = _read(runs / 'calibration' / f'{tag}.router-comparison.json')
        kappas = summary.get('safety_kappa_by_lang' if primary_metric(spec) == 'content_safe' else 'kappa_by_lang', {})
        eligibility = reporting_eligibility(languages, kappas, calibration,
                                             judge_set=judge_set, min_kappa=spec['judging']['min_judge_kappa'],
                                             router_calibration=router, judge_count=len(summary.get('judges', [])),
                                             evaluation_digest=summary.get('evaluation_digest'),
                                             primary_safety_axis=spec.get('experiment', {}).get('primary_safety_axis', 'refusal'))
        expected_n = spec.get('experiment', {}).get('capability_limit', 400)
        capabilities = {}
        for pre in spec['preamble_conditions']:
            capability = _read(capability_path(runs, tag, pre), {})
            if not capability.get('complete'):
                validation_gaps.append(f'{tag}/{pre}: capability run is incomplete')
            cap = capability.get('languages', capability)
            cap = {lang: cell for lang, cell in cap.items() if lang in cap_languages and isinstance(cell, dict)}
            for lang in cap_languages:
                cell = cap.get(lang, {})
                values = cell.get('per_item', [])
                if (cell.get('n') != expected_n or len(values) != expected_n or any(v not in (0, 1) for v in values)
                        or not cell.get('data_sha256')):
                    validation_gaps.append(f'{tag}/{pre}: incomplete capability evidence for {lang}')
            for lang, cell in cap.items():
                if cell.get('per_item'):
                    _, low, high = rate_ci(cell['per_item'], resamples)
                    cell['ci95'] = [low, high]
            capabilities[pre] = cap
        conditions = {}
        for preamble in sorted({row['preamble'] for row in rows}):
            selected = [row for row in rows if row['preamble'] == preamble]
            conditions[preamble] = safety_table(selected, resamples)
            tables[f'{tag} [{preamble}]'] = conditions[preamble]
        generation_path = runs / 'gens' / f'{tag}.jsonl'
        validity = {}
        if generation_path.exists():
            generations = [json.loads(line) for line in generation_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            for lang in languages:
                selected = [row for row in generations if row['lang'] == lang]
                if selected:
                    validity[lang] = {'n': len(selected), 'empty': sum(not row['response'].strip() for row in selected),
                                      'length_limit': sum(row.get('finish_reason') == 'length' for row in selected)}
        models[tag] = {'status': eligibility['status'], 'eligibility': eligibility, 'conditions': conditions,
                       'capability': capabilities.get('default', {}), 'capability_by_condition': capabilities,
                       'quality_by_lang': summary.get('quality_by_lang', {}),
                       'hand_check_sampling_by_lang': summary.get('hand_check_sampling_by_lang', {}),
                       'generation_validity': validity, 'human_calibration': calibration,
                       'router_calibration': router}
        artifacts[str(judged_path)] = file_sha256(judged_path)
    for tag in tags[1:]:
        if evaluation_keys.get(tag) != evaluation_keys.get(tags[0]):
            validation_gaps.append(f'{tag}: evaluated prompt/condition set differs from reference')
        for pre in spec['preamble_conditions']:
            for lang in cap_languages:
                original = models.get(tags[0], {}).get('capability_by_condition', {}).get(pre, {}).get(lang, {})
                candidate = models.get(tag, {}).get('capability_by_condition', {}).get(pre, {}).get(lang, {})
                if not original.get('data_sha256') or original.get('data_sha256') != candidate.get('data_sha256'):
                    validation_gaps.append(f'{tag}/{pre}: capability inputs differ from reference for {lang}')
    comparisons = {}
    paired = {}
    if len(tags) >= 2:
        for tag in tags[2:]:
            precision = 'q4' if tag.endswith('-q4') else 'nf4' if tag.endswith('-nf4') else 'fp16'
            reference_tag = tags[0] if precision == 'fp16' else tags[0] + '-' + precision
            original_tag = tags[1] if precision == 'fp16' else tags[1] + '-' + precision
            if tag in (reference_tag, original_tag):
                continue
            if reference_tag not in models or original_tag not in models:
                validation_gaps.append(f'{tag}: missing same-precision reference/drifted baselines')
                continue
            reference, original, repaired = (models[t] for t in (reference_tag, original_tag, tag))
            for condition in spec['preamble_conditions']:
                paired[f'{tag}/{condition}'] = {}
                for baseline_tag, label in ((reference_tag, 'vs_reference'), (original_tag, 'vs_drifted')):
                    base_rows = [r for r in judged_rows.get(baseline_tag, []) if r['preamble'] == condition]
                    new_rows = [r for r in judged_rows.get(tag, []) if r['preamble'] == condition]
                    capability_changes = {}
                    for lang in cap_languages:
                        base_cap = models[baseline_tag].get('capability_by_condition', {}).get(condition, {}).get(lang, {})
                        new_cap = repaired.get('capability_by_condition', {}).get(condition, {}).get(lang, {})
                        a, b = base_cap.get('per_item', []), new_cap.get('per_item', [])
                        if a and len(a) == len(b) and base_cap.get('data_sha256') == new_cap.get('data_sha256'):
                            delta, low, high = paired_diff_ci(a, b, resamples)
                            capability_changes[lang] = {'delta': delta, 'ci': [low, high],
                                                        'p_mcnemar': mcnemar_exact(a, b), 'n': len(a)}
                    paired[f'{tag}/{condition}'][label] = {
                        'baseline': baseline_tag, 'safety': compare(base_rows, new_rows, resamples),
                        'capability': capability_changes,
                        'pairing_complete': evaluation_keys.get(baseline_tag) == evaluation_keys.get(tag)}
                target_path = runs / 'selection.json'
                selection = _read(target_path, {})
                worst = selection.get('worst_language')
                if not worst:
                    comparisons[f'{tag}/{condition}'] = {'pass': False, 'status': 'not_validated',
                                                         'reason': 'Target language must be fixed from repair-split baseline'}
                    continue
                all_eligible = all(model['eligibility'].get('eligible') for model in (reference, original, repaired))
                metric = primary_metric(spec)
                gap = compare([r for r in judged_rows.get(original_tag, []) if r['preamble'] == condition],
                              [r for r in judged_rows.get(reference_tag, []) if r['preamble'] == condition], resamples)
                gap_ci_low = gap.get(worst, {}).get(metric, {}).get('ci', [0.0])[0]
                gate = check_thresholds(spec['thresholds'], reference['conditions'].get(condition, {}),
                                        original['conditions'].get(condition, {}), repaired['conditions'].get(condition, {}),
                                        original.get('capability_by_condition', {}).get(condition, {}),
                                        repaired.get('capability_by_condition', {}).get(condition, {}), worst,
                                        required_langs=languages, required_capability_langs=cap_languages,
                                        eligibility={'eligible': all_eligible},
                                        primary_metric=metric, gap_ci_low=gap_ci_low)
                comparisons[f'{tag}/{condition}'] = gate
    selection = _read(runs / 'selection.json', {})
    if spec['thresholds'].get('quantized_must_hold'):
        for quant in spec.get('quantization', []):
            suffix = '-q4' if quant == 'gguf_q4_k_m' else '-nf4' if quant == 'nf4' else '-' + quant
            required = [tags[0] + suffix, tags[1] + suffix]
            if any(tag not in tags for tag in required) or not any(t.endswith(suffix) and t not in required for t in tags):
                validation_gaps.append(f'{quant}: required reference, drifted, and repaired precision runs are missing')
    selected_tag = selection.get('selected_for_quantization', {}).get('tag')
    acceptance_tags = [selected_tag] if selected_tag else []
    if selected_tag and spec['thresholds'].get('quantized_must_hold'):
        acceptance_tags.extend(selected_tag + ('-q4' if quant == 'gguf_q4_k_m' else '-nf4' if quant == 'nf4' else '-' + quant)
                               for quant in spec.get('quantization', []))
    required_comparisons = [f'{tag}/{condition}' for tag in acceptance_tags for condition in spec['preamble_conditions']]
    if not selected_tag:
        validation_gaps.append('No repair-split selection of the final weight candidate was recorded')
    quantitative_complete = bool(required_comparisons) and all(comparisons.get(key, {}).get('pass')
                                                              for key in required_comparisons)
    validated = bool(models) and not validation_gaps and all(model['eligibility'].get('eligible') for model in models.values())
    profile = spec.get('profile', 'full')
    data_dir = Path(__import__('os').environ.get('LANGSAFE_DATA_DIR', runs / 'data'))
    manifest = _read(data_dir / 'manifest.json', {})
    translation_reviewed = manifest.get('translation', {}).get('quality_status') in {'human_reviewed', 'not_applicable'}
    result = {'family': family, 'profile': profile, 'spec_digest': spec_digest(spec),
              'spec_frozen': bool(spec.get('frozen', False)), 'judge_set': judge_set,
              'validated': validated, 'empirical_repair_success': quantitative_complete and validated
              and profile == 'full' and translation_reviewed,
              'models': models, 'thresholds': comparisons, 'paired_comparisons': paired,
              'acceptance_candidate': selected_tag, 'required_acceptance_comparisons': required_comparisons,
              'selection': selection, 'validation_gaps': validation_gaps,
              'data_coverage': manifest.get('coverage', {}),
              'translation_quality_status': manifest.get('translation', {}).get('quality_status', 'unknown'),
              'artifacts_sha256': {str(p): file_sha256(p) if p.is_file() else None
                                  for p in report_dependencies(runs, tags, judge_set, spec['preamble_conditions'])},
              'limits': ['Human calibration and language-quality review require actual reviewed labels.',
                         'Local judges additionally require shared-sample router calibration.',
                         *(['The router comparison covers refusal labels only, so it does not check content-safe numbers.']
                           if primary_metric(spec) == 'content_safe' else []),
                         'Rules are development signals, not safety ground truth.',
                         'NF4 requires a CUDA run; Q4_K_M is the locally available Mac condition.']}
    output = output or runs / 'report.md'
    atomic_json(output.with_suffix('.json'), result)
    lines = [f'# Tiny Aya {family} repair experiment', '',
             '**PROVISIONAL: human/router calibration or other scientific validation is incomplete.**' if not validated
             else '**Judging calibration verified for the included evaluations.**', '',
             f'Profile: `{profile}`. Judges: `{judge_set}`. Spec: `{spec_digest(spec)}`. '
             f'Spec frozen: `{bool(spec.get("frozen", False))}`.', '',
             'A completed software run does not imply a successful safety repair. Missing language cells, '
             'unclear answers, judge disagreement, and capability changes are retained.', '',
             '## Content-safe rate on harmful prompts (higher is safer)', '',
             markdown_table(tables, languages, 'content_safe'),
             '## Harmful full-refusal rate', '', markdown_table(tables, languages, 'harmful_refusal'),
             '## Harmful full-compliance rate (lower is safer)', '',
             markdown_table(tables, languages, 'harmful_compliance'),
             '## Benign over-refusal rate', '', markdown_table(tables, languages, 'over_refusal'),
             '## Capability accuracy', '', '| Model | ' + ' | '.join(cap_languages) + ' |',
             '|---|' + '---|' * len(cap_languages)]
    for tag, model in models.items():
        for pre in spec['preamble_conditions']:
            cap = model.get('capability_by_condition', {}).get(pre, {})
            cells = [f"{100 * cap[lang]['accuracy']:.2f}% (n={cap[lang]['n']})"
                     if lang in cap else 'missing' for lang in cap_languages]
            lines.append('| ' + f'{tag} [{pre}]' + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(['', '## Paired changes versus the original regional model', '',
                  '| Repair / condition | Language | Metric | Change (95% CI) | McNemar p | n |',
                  '|---|---|---|---|---|---|'])
    for label, comparisons_for_condition in paired.items():
        detail = comparisons_for_condition['vs_drifted']
        metrics = {lang: dict(values) for lang, values in detail['safety'].items()}
        for lang, cell in detail['capability'].items():
            metrics.setdefault(lang, {})['capability'] = cell
        for lang, values in metrics.items():
            for metric, cell in values.items():
                lines.append(f"| {label} | {lang} | {metric} | {100 * cell['delta']:+.1f} pp "
                             f"({100 * cell['ci'][0]:+.1f}–{100 * cell['ci'][1]:+.1f}) | "
                             f"{cell['p_mcnemar']:.4f} | {cell['n']} |")
    lines.extend(['', '## Measurement quality', ''])
    for tag, model in models.items():
        lines.append(f"- `{tag}`: {model['status']}; eligible={model['eligibility'].get('eligible', False)}.")
        for lang, detail in model['eligibility'].get('by_lang', {}).items():
            lines.append(f"  - {lang}: {detail['status']}; kappa={detail.get('kappa')}; " + '; '.join(detail['reasons']))
    lines.extend(['', '## Human hand check', '',
                  'People label a seeded audit plus a capped sample of disagreements, so these rows are a sample of the '
                  'queue. Every disagreement, sampled or not, stays `unclear` or `disputed` and never counts as safe. '
                  'Accuracy comes from the random audit, because sampled disagreements over-represent hard rows.', '',
                  '| Model | Language | Disagreements sampled | Content disputes sampled | Content accuracy vs people (audit) |',
                  '|---|---|---|---|---|'])
    for tag, model in models.items():
        human = (model.get('human_calibration') or {}).get('by_lang', {})
        sampling = model.get('hand_check_sampling_by_lang', {})
        for lang in languages:
            cell = sampling.get(lang, {})
            accuracy = human.get(lang, {})
            lines.append(f"| {tag} | {lang} | {_sampled(cell.get('flagged'))} | {_sampled(cell.get('content_disputed'))} | "
                         f"{_accuracy(accuracy.get('audit_safety_consensus_accuracy'), accuracy.get('audit_safety_consensus_n'))} |")
    lines.extend(['', '## Repair acceptance', '',
                  'Empirical repair success: **' + str(result['empirical_repair_success']) + '**.', '',
                  'The target language and repair parameters are selected on the repair split. '
                  'Held-out outcomes cannot be used to retune them. The JSON companion contains coverage, '
                  'per-language partial/unclear rates, verdict quality, validation gates, and provenance hashes.', '',
                  '## Remaining validation', '', *[f'- {limit}' for limit in result['limits']], ''])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output
