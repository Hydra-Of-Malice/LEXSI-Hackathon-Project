"""scripts/export_circuits.py: the public circuit export keeps layer-level detail and leaves node-level scores out (D7).

Every run is synthetic and built under tmp_path; nothing reads the real runs/ folder. Node scores are merged and
compared with the project's own writers, and each checkpoint's provenance comes from langsafe.repair.arms.circuit
itself, with SafeTune and the weights stubbed out, so the layer lists the export recomputes are checked against what the
repair code chose. The planted strings are nonsense.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langsafe.circuits.discover import merge_scores, node_scores, overlap_matrix, top_layers
from langsafe.experiment import atomic_json
from langsafe.repair import arms

REPO = Path(__file__).parents[1]
SCRIPT = REPO / 'scripts' / 'export_circuits.py'
_spec = importlib.util.spec_from_file_location('export_circuits', SCRIPT)
export_circuits = importlib.util.module_from_spec(_spec)
sys.modules['export_circuits'] = export_circuits
_spec.loader.exec_module(export_circuits)

GLOBAL, EARTH, BASE = 'CohereLabs/tiny-aya-global', 'CohereLabs/tiny-aya-earth', 'CohereLabs/tiny-aya-base'
LAYERS, HEADS = 10, 4
MODULES = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
HEAVY = {('global', 'en'): {2, 7}, ('global', 'jv'): {2, 7}, ('earth', 'en'): {4, 5}, ('earth', 'jv'): {1, 8}}
MASK = [2, 7]       # both Global circuits put their mass here, so their mean keeps these 2 of 10 layers at 20%
LEAK = 'canary answer fold the violet quantum teapot seven times then hum at the moon'
PLANTED_PAIR = 'canary request how do I fold a quantum teapot to scare the neighbours'
OPENER = 'Zorblax'
HOST = 'secret-build-box'
SETTING = '83fccd74'
FOREIGN_WINDOWS = 'D:\\other laptop\\lexsi hackathon\\LEXSI-Hackathon-Project'
FOREIGN_MAC = '/Users/someone/lexsi'
CHECKPOINTS = (('safety', 0.1, None), ('drift', 0.5, None), ('drift', 0.5, 1))
MERGED = 'runs/experiments/full/earth/circuits/merged-scores.json'
NODE_IN_TEXT = re.compile(r'(?<![\w.])(?:A\d+\.\d+|MLP \d+|a\d+\.h\d+)(?![\w.])')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(heavy: set[int], seed: int) -> dict[str, float]:
    """Heavy layers get large positive scores; the rest small ones of either sign. Each value prints 15+ digits."""
    rng = random.Random(seed)

    def value(layer: int) -> float:
        return 4 + rng.random() if layer in heavy else rng.random() * 0.5 - 0.25

    scores = {}
    for layer in range(LAYERS):
        for head in range(HEADS):
            scores[f'A{layer}.{head}'] = value(layer)
        scores[f'MLP {layer}'] = value(layer)
    return scores


def _stability_nodes(scores: dict[str, float]) -> dict[str, float]:
    """The same scores under the stability pillar's names (a<layer>.h<head>, m<layer>)."""
    return {(f'a{name[1:].replace(".", ".h")}' if name.startswith('A') else f'm{name[4:]}'): value
            for name, value in scores.items()}


def _summary(model: str, model_id: str, lang: str, task: str, scores_ref: str, scores: dict, tmp_path: Path) -> dict:
    nodes = _stability_nodes(scores)
    return {
        'model': model_id, 'resolved_model': str(tmp_path / 'hf' / f'models--{model_id.replace("/", "--")}'),
        'model_tag': f'{model}-circuit', 'lang': lang, 'task': task, 'scores_json': scores_ref,
        'faithfulness': {
            'patching_score': 0.9375, 'ablation_score': 0.5,
            'stability': {'mean_jaccard': 1.0, 'std_jaccard': 0.0, 'mean_dice': 1.0, 'std_dice': 0.0,
                          'mean_spearman': 1.0, 'std_spearman': 0.0, 'jaccard_matrix': [[1.0, 1.0], [1.0, 1.0]],
                          'dice_matrix': [[1.0, 1.0], [1.0, 1.0]], 'spearman_matrix': [[1.0, 1.0], [1.0, 1.0]],
                          'n_stable_nodes': 35, 'overlap_per_layer': {str(layer): 1.0 for layer in range(LAYERS)},
                          'circuits': [nodes, nodes], 'raw_scores': [nodes, nodes], 'n_runs': 2},
            'robustness': None,
            'baseline_comparison': {
                'circuit_score': 3.0, 'best_baseline_score': 0.75, 'circuit_advantage': 2.25,
                'baselines': {'random': {'score': 0.5, 'percentage': None, 'score_std': 0.25, 'n_draws': 3,
                                         'random_scores': [0.25, 0.5, 0.75], 'z_score': 10.0},
                              'magnitude': {'score': 0.75}},
                'summary': LEAK, 'circuit_sparsity': 0.7, 'baseline_types': ['random', 'magnitude']},
            'metadata': {'patching_raw_ratio': 0.9375, 'ablation_raw_ratio': 0.5, 'algorithm': 'eap-ig',
                         'model': 'tiny-aya', 'task': task, 'level': 'node', 'scope': 'heads', 'sparsity': 0.3,
                         'pillars_computed': ['patching', 'ablation', 'baselines', 'stability'],
                         'timestamp': 1794115734.0, 'total_duration_seconds': 60.0}},
        'parity': {'model': model_id, 'passed': True, 'dtype': 'float32', 'device': 'cpu', 'kl_tolerance': 1e-4,
                   'max_logprob_tolerance': 5e-3, 'circuitkit_commit': 'e71d5e1',
                   'per_prompt': [{'prompt_sha256': 'ab' * 32, 'kl': 1e-8, 'max_logprob_deviation': 1e-5,
                                   'argmax_agrees': True}]},
        'settings': {'precision': 'float32', 'n_examples': 4, 'ig_steps': 3, 'sparsity': 0.3, 'seed': 0,
                     'data_max_records': 4, 'pillars': ['patching', 'ablation', 'baselines', 'stability'],
                     'n_stability_runs': 2},
        'circuitkit_commit': 'e71d5e1', 'task_examples': 4, 'openers': [OPENER, 'Sure']}


def _tensor_names() -> list[str]:
    names = ['model.embed_tokens.weight', 'model.norm.weight']
    for layer in range(LAYERS):
        names.append(f'model.layers.{layer}.input_layernorm.weight')
        names += [f'model.layers.{layer}.mlp.{module}.weight' for module in ('gate_proj', 'up_proj', 'down_proj')]
        names += [f'model.layers.{layer}.self_attn.{module}.weight' for module in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    return names


def _write_safetensors_header(path: Path, names: list[str]) -> None:
    header = {'__metadata__': {'format': 'pt'}}
    for index, name in enumerate(names):
        header[name] = {'dtype': 'F16', 'shape': [1], 'data_offsets': [2 * index, 2 * index + 2]}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(blob)) + blob + bytes(2 * len(names)))


def _stub(monkeypatch, name: str, **exports) -> None:
    module = ModuleType(name)
    module.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, module)


def _checkpoints(run: Path, merged: Path, monkeypatch) -> list[tuple]:
    """Each checkpoint's provenance params come from langsafe.repair.arms.circuit, with SafeTune and weights stubbed."""
    names = _tensor_names()
    state = {name: SimpleNamespace(shape=(2, 2)) for name in names}
    # SafeTune's own node-level layer subset is broad on purpose: the repair code must replace it with top_layers.
    info = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=list(range(LAYERS)), target_modules=MODULES))
    _stub(monkeypatch, 'safetune.core.circuit_kit.adapter', circuit_info_from_node_scores=lambda *a, **k: info,
          load_circuit_info_from_file=lambda *a, **k: info)
    _stub(monkeypatch, 'safetune.recover', apply_ctheta=lambda *a, **k: None)
    monkeypatch.setattr(arms, 'load_spec', lambda: {'models': {'base': BASE, 'reference': GLOBAL}})
    monkeypatch.setattr(arms, '_load_cpu', lambda name: SimpleNamespace(state_dict=lambda: state))
    saved = []
    monkeypatch.setattr(arms, '_save', lambda *args: saved.append(args) or args[2])
    for vector, strength, seed in CHECKPOINTS:
        arms.circuit(EARTH, str(merged), strength, 'earth', top_fraction=0.2, vector=vector, random_seed=seed)
    for _model, _drifted, out, method, inputs, params in saved:
        folder = run / 'checkpoints' / out.name
        folder.mkdir(parents=True)
        params = {**params, 'edit_l2': 1.25, 'relative_edit_l2': 0.0125, 'edited_tensors': params['matched_parameters'],
                  'edited_layers': params['layer_subset'], 'safetune_commit': arms.SAFETUNE_COMMIT,
                  'runtime': {'torch': '2.11.0+cu128', 'safetune': '0.1.12'}}
        if params['vector'] == 'drift' and params['random_seed'] is None:     # written on another machine
            params['scores'] = FOREIGN_WINDOWS + '\\runs\\experiments\\full\\earth\\circuits\\merged-scores.json'
        provenance = {'schema': 'lexsi.provenance/1', 'library': 'safetune', 'version': '0.1.12', 'git_sha': None,
                      'created_at': '2026-11-08T07:00:53Z', 'base_model': inputs[0], 'method': f'langsafe.{method}',
                      'inputs': [{'kind': 'model', 'ref': ref, 'provenance': None} for ref in inputs], 'params': params}
        (folder / 'lexsi_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        _write_safetensors_header(folder / 'model.safetensors', names)
    return saved


def build_run(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    """A small Earth run: four circuits, merged Global scores, overlap, logs, selection, three circuit checkpoints."""
    run = tmp_path / 'runs' / 'experiments' / 'full' / 'earth'
    (run / 'gens').mkdir(parents=True)
    (run / 'gens' / 'earth.jsonl').write_text(
        json.dumps({'id': 'g1::en', 'response': LEAK}) + '\n'
        + json.dumps({'id': 'g2::en', 'response': 'I am sorry, but I cannot help with that request.'}) + '\n',
        encoding='utf-8')
    (run / 'logs').mkdir()
    paths = {}
    for seed, (model, model_id) in enumerate((('global', GLOBAL), ('earth', EARTH))):
        for lang in ('en', 'jv'):
            tag = f'{model}-circuit'
            task_csv = run / 'circuit_tasks' / f'refusal_{lang}__{tag}__none.csv'
            task_csv.parent.mkdir(parents=True, exist_ok=True)
            rows = [f'{PLANTED_PAIR} {index},harmless twin {index},{OPENER},Sure' for index in range(4)]
            task_csv.write_text('harmful,harmless,refuse,comply\n' + '\n'.join(rows) + '\n', encoding='utf-8')
            digest = _sha(task_csv)[:8]
            task = f'refusal_{lang}__{tag}__{digest}_{SETTING}'
            folder = run / 'circuits' / f'{tag}__{lang}'
            inner = folder / f'{digest}-{SETTING}'
            inner.mkdir(parents=True)
            stem = f'eap-ig_{model_id.replace("/", "_")}_{task}_node'
            scores = _scores(HEAVY[model, lang], seed * 10 + len(lang) + (lang == 'jv'))
            scores_json = inner / f'{stem}_scores.json'
            scores_json.write_text(json.dumps({'model': model_id, 'algorithm': 'eap-ig', 'level': 'node',
                                               'node_scores': scores,
                                               'safety_units': {'unit_ids': list(scores)[:5]}}), encoding='utf-8')
            (inner / f'{stem}_scores.pt').write_bytes(b'node scores, private')
            (inner / f'{stem}.pt').write_bytes(b'pruning mask, private')
            relative = scores_json.relative_to(tmp_path)
            # The recorded path takes each form a real run has produced.
            scores_ref = {('global', 'en'): str(scores_json),
                          ('global', 'jv'): str(relative).replace('/', '\\'),
                          ('earth', 'en'): FOREIGN_MAC + '/' + relative.as_posix(),
                          ('earth', 'jv'): FOREIGN_WINDOWS + '\\' + str(relative).replace('/', '\\')}[model, lang]
            summary = _summary(model, model_id, lang, task, scores_ref, scores, tmp_path)
            (inner / 'summary.json').write_text(json.dumps({key: value for key, value in summary.items()
                                                            if key not in ('openers', 'task_examples')}),
                                                encoding='utf-8')
            (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            (folder / 'fair_baseline.json').write_text(json.dumps({
                'method': 'random heads, every MLP block and the input kept; CircuitKIT patching pillar',
                'kept_heads': 25, 'kept_mlps': 10, 'seeds': [0, 1, 2], 'patching_ratios': [0.25, 0.375, 0.5],
                'mean': 0.375, 'circuit_patching_ratio': 0.9375, 'check_ratio': 0.9375, 'n_examples': 4,
                'circuitkit_commit': 'e71d5e1',
                'hardware': {'device': 'cuda', 'gpu': 'NVIDIA L40S', 'torch': '2.11.0+cu128', 'cuda': '12.8',
                             'tf32': False, 'float32_matmul_precision': 'highest', 'hostname': HOST}}),
                encoding='utf-8')
            (run / 'logs' / f'{model}-circuit-{lang}.log').write_text(
                f'loading pairs: {PLANTED_PAIR}\n12:00:00 \u2022 Pillar 1 Score (Causal Patching faithfulness ratio): '
                '0.9375 (raw metric: 3.0000, clean: 3.2000, corrupt: 0.2000)\n', encoding='utf-8')
            paths[model, lang] = scores_json
    merged = merge_scores([paths['global', 'en'], paths['global', 'jv']], run / 'circuits' / 'merged-scores.json',
                          GLOBAL)
    atomic_json(run / 'circuits' / 'overlap.json',
                overlap_matrix({f'{model}/{lang}': path for (model, lang), path in paths.items()}, k=5))
    saved = _checkpoints(run, merged, monkeypatch)

    def checkpoint(name: str) -> str:
        return str(run / 'checkpoints' / name)

    merge_selected = {'arm': 'circuit_merge', 'knob': 0.5, 'tag': 'earth-circuit_merge-0p5',
                      'model': checkpoint('earth-circuit_merge-s0.5-top0.2')}
    atomic_json(run / 'selection.json', {
        'arms': {'circuit': {'knob': 0.1, 'selected': {'arm': 'circuit', 'knob': 0.1, 'tag': 'earth-circuit-0p1',
                                                       'model': checkpoint('earth-circuit-s0.1-top0.2')}},
                 'circuit_merge': {'knob': 0.5, 'selected': merge_selected}},
        'controls': [{'arm': 'circuit_merge', 'knob': 0.5, 'random_seed': 1, 'tag': 'earth-circuit_merge-0p5-rand1',
                      'model': checkpoint('earth-circuit_merge-s0.5-top0.2-rand1')}],
        'selected_for_quantization': merge_selected})
    return run, {'scores': paths, 'merged': merged, 'saved': saved}


def export(run: Path, tmp_path: Path, **options) -> tuple[int, Path]:
    out = tmp_path / 'public' / 'circuits'
    return export_circuits.export_circuits(run, out, repo_root=tmp_path, **options), out


def output_text(out: Path) -> str:
    return '\n'.join(path.read_text(encoding='utf-8') for path in sorted(out.rglob('*')) if path.is_file())


def read(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text(encoding='utf-8'))


def test_node_level_scores_are_left_out(tmp_path, monkeypatch):
    run, built = build_run(tmp_path, monkeypatch)
    code, out = export(run, tmp_path)
    assert code == 0
    files = {path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file()}
    assert files == {'README.md', 'EXPORT_MANIFEST.json', 'editing_mask.json', 'repair_arms.json', 'overlap.json',
                     *(f'circuits/{model}-circuit__{lang}.json' for model, lang in HEAVY)}
    text = output_text(out)
    for path in [*built['scores'].values(), built['merged']]:
        for value in node_scores(path).values():
            assert repr(value) not in text
    assert not NODE_IN_TEXT.search(text)
    for planted in (LEAK, PLANTED_PAIR, OPENER, HOST):
        assert planted not in text

    record = read(out, 'circuits/global-circuit__en.json')
    scores_json = built['scores']['global', 'en']
    assert record['scores_file'] == {'path': scores_json.relative_to(tmp_path).as_posix(),
                                     'sha256': _sha(scores_json), 'public': False}
    assert record['nodes'] == {'total': 50, 'attention_heads': 40, 'mlp_blocks': 10, 'neurons': 0, 'other': 0,
                               'kept_at_sparsity': int(50 * (1 - 0.3))}
    assert set(record['faithfulness']['stability']) == {
        'n_runs', 'mean_jaccard', 'std_jaccard', 'mean_dice', 'std_dice', 'mean_spearman', 'std_spearman',
        'n_stable_nodes', 'jaccard_matrix', 'overlap_per_layer'}
    assert record['task']['pairs_in_file'] == 4 and record['task']['preamble'] == 'none'
    assert record['refusal_signal']['signal'] == pytest.approx(3.0)
    assert 'hostname' not in record['fair_baseline']['hardware']

    sources = {entry['file']: entry for entry in read(out, 'EXPORT_MANIFEST.json')['sources']}
    private = [built['merged'], *built['scores'].values()]
    private += [path for path in run.glob('circuits/*/*/*.pt')] + list(run.glob('circuit_tasks/*.csv'))
    assert len(private) == 17
    for path in private:
        entry = sources[path.relative_to(tmp_path).as_posix()]
        assert entry['status'] == 'withheld' and entry['sha256'] == _sha(path)
    left_out = {item['field'] for item in read(out, 'EXPORT_MANIFEST.json')['summary_fields_left_out']}
    assert {'faithfulness.stability.raw_scores', 'faithfulness.stability.circuits', 'openers'} <= left_out


def test_local_paths_are_stripped(tmp_path, monkeypatch):
    run, built = build_run(tmp_path, monkeypatch)
    code, out = export(run, tmp_path)
    assert code == 0
    text = output_text(out)
    for marker in (str(tmp_path), tmp_path.as_posix(), 'D:', 'C:\\', 'Users', '\\', 'lexsi hackathon', 'other laptop',
                   'someone'):
        assert marker not in text, marker
    mask = read(out, 'editing_mask.json')['masks'][0]
    assert mask['source']['scores_file'] == MERGED
    assert [item['scores_file'] for item in mask['source']['merged_from']] == [
        built['scores'][key].relative_to(tmp_path).as_posix() for key in (('global', 'en'), ('global', 'jv'))]
    assert [item['circuit'] for item in mask['source']['merged_from']] == ['global-circuit__en', 'global-circuit__jv']
    arms_out = read(out, 'repair_arms.json')['arms']
    assert {arm['path'] for arm in arms_out} == {f'runs/experiments/full/earth/checkpoints/{out_dir.name}'
                                                 for *_, out_dir, _, _, _ in built['saved']}
    assert {arm['scores_file'] for arm in arms_out} == {MERGED}       # absolute here, and from another machine
    assert all(arm['checks']['scores_sha256_matches'] for arm in arms_out)
    for model, lang in HEAVY:         # recorded as absolute, relative, macOS and foreign Windows paths
        assert read(out, f'circuits/{model}-circuit__{lang}.json')['scores_file']['path'] == \
            built['scores'][model, lang].relative_to(tmp_path).as_posix()


@pytest.mark.parametrize('value,expected', [
    (FOREIGN_WINDOWS + '\\runs\\experiments\\full\\earth\\circuits\\merged-scores.json', MERGED),
    (FOREIGN_MAC + '/runs/experiments/full/earth/circuits/merged-scores.json', MERGED),
    ('runs\\experiments\\full\\earth\\circuits\\merged-scores.json', MERGED),
    ('C:\\Users\\someone\\.cache\\huggingface\\hub\\models--CohereLabs--tiny-aya-global\\snapshots\\0123abc', GLOBAL),
    ('C:\\Users\\someone\\Desktop\\scores.json', 'external/scores.json'),
    ('/home/someone/notes/../scores.json', 'external/scores.json'),
    ('runs/../../escape.json', None),
])
def test_recorded_paths_become_repo_relative(tmp_path, value, expected):
    assert export_circuits.repo_relative(value, tmp_path) == expected


def test_paths_inside_the_repo_keep_their_place_under_it(tmp_path):
    root = tmp_path / 'LEXSI-Hackathon-Project'
    assert export_circuits.repo_relative(str(root / 'runs' / 'x.json'), root) == 'runs/x.json'
    assert export_circuits.repo_relative('E:\\copy\\LEXSI-Hackathon-Project\\configs\\spec.yaml', root) == \
        'configs/spec.yaml'


def test_layer_lists_match_what_the_repair_code_selects(tmp_path, monkeypatch):
    run, built = build_run(tmp_path, monkeypatch)
    code, out = export(run, tmp_path)
    assert code == 0
    expected = top_layers(node_scores(built['merged']), 0.2)
    assert expected == MASK

    mask = read(out, 'editing_mask.json')['masks'][0]
    assert mask['layers'] == expected and mask['top_fraction'] == 0.2 and mask['target_modules'] == MODULES
    assert mask['source']['layers']['selected'] == [{'top_fraction': 0.2, 'layers': expected, 'layer_count': 2}]
    ranking = mask['source']['layers']['ranking']
    assert [row['layer'] for row in ranking] == list(range(LAYERS))
    assert {row['layer'] for row in ranking if row['rank'] <= 2} == set(MASK)
    assert sum(row['share'] for row in ranking) == pytest.approx(1.0)
    assert mask['parameters'] == sorted((name for name in _tensor_names() if name.split('.')[2] in map(str, MASK)
                                         and any(module in name for module in MODULES)),
                                        key=lambda name: (int(name.split('.')[2]), name))
    assert mask['source']['mean_of_sources_verified'] is True
    assert mask['used_by'] == ['earth-circuit-s0.1-top0.2', 'earth-circuit_merge-s0.5-top0.2']
    assert mask['random_layer_controls'] == ['earth-circuit_merge-s0.5-top0.2-rand1']

    for (model, lang), path in built['scores'].items():
        layers = read(out, f'circuits/{model}-circuit__{lang}.json')['layers']
        assert layers['selected'][0]['layers'] == top_layers(node_scores(path), 0.2) == sorted(HEAVY[model, lang])
        assert read(out, f'circuits/{model}-circuit__{lang}.json')['used_for_editing_mask'] is (model == 'global')

    by_name = {arm['checkpoint']: arm for arm in read(out, 'repair_arms.json')['arms']}
    assert len(by_name) == len(CHECKPOINTS)
    for *_, out_dir, _, inputs, params in built['saved']:
        arm = by_name[out_dir.name]
        assert arm['circuit_layers'] == params['circuit_layers'] == expected
        assert arm['edited_layers'] == params['layer_subset']
        assert arm['parameter_count'] == params['matched_parameters'] == 7 * len(params['layer_subset'])
        assert arm['models'] == dict(zip(('drifted', 'positive', 'negative'), inputs))
        assert set(arm['checks']) == {'checkpoint_name_matches', 'edited_layers_match_layer_subset',
                                      'scores_sha256_matches', 'circuit_layers_recomputed', 'edited_layers_recomputed',
                                      'parameter_count_matches'}
        assert all(arm['checks'].values())
    control = by_name['earth-circuit_merge-s0.5-top0.2-rand1']
    assert len(control['edited_layers']) == 2 and not set(control['edited_layers']) & set(MASK)
    assert control['role'] == 'random-layer control' and control['random_seed'] == 1
    assert by_name['earth-circuit-s0.1-top0.2']['role'] == 'selected on the repair split'
    assert by_name['earth-circuit-s0.1-top0.2']['models']['negative'] == BASE
    assert by_name['earth-circuit_merge-s0.5-top0.2']['selected_for_quantization'] is True
    assert by_name['earth-circuit_merge-s0.5-top0.2']['vector'] == 'drift'


def test_a_mask_that_disagrees_with_its_scores_blocks_the_export(tmp_path, monkeypatch):
    run, _ = build_run(tmp_path, monkeypatch)
    path = run / 'checkpoints' / 'earth-circuit-s0.1-top0.2' / 'lexsi_provenance.json'
    provenance = json.loads(path.read_text(encoding='utf-8'))
    provenance['params'].update(circuit_layers=[3, 7], layer_subset=[3, 7], edited_layers=[3, 7])
    path.write_text(json.dumps(provenance), encoding='utf-8')
    code, out = export(run, tmp_path)
    assert code == 1 and not out.exists()
    failed = json.loads(out.with_name('circuits.scan-failed.json').read_text(encoding='utf-8'))
    assert {'circuit_layers_recomputed', 'edited_layers_recomputed'} <= {problem['check']
                                                                         for problem in failed['problems']}
    assert not list(out.parent.glob('.circuits.staging-*'))


def test_model_text_in_the_output_blocks_the_export(tmp_path, monkeypatch):
    run, _ = build_run(tmp_path, monkeypatch)
    path = run / 'circuits' / 'earth-circuit__en' / 'fair_baseline.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps({**document, 'method': LEAK}), encoding='utf-8')
    code, out = export(run, tmp_path)
    assert code == 1 and not out.exists()
    failed = out.with_name('circuits.scan-failed.json').read_text(encoding='utf-8')
    assert 'response_text' in failed and LEAK not in failed


def test_an_earlier_export_is_replaced_only_on_request(tmp_path, monkeypatch):
    run, _ = build_run(tmp_path, monkeypatch)
    code, out = export(run, tmp_path)
    assert code == 0
    with pytest.raises(export_circuits.ExportError):
        export(run, tmp_path)
    assert export(run, tmp_path, replace=True)[0] == 0
    other = tmp_path / 'public' / 'other'
    other.mkdir()
    (other / 'keep.txt').write_text('not an export', encoding='utf-8')
    with pytest.raises(export_circuits.ExportError):
        export_circuits.export_circuits(run, other, repo_root=tmp_path, replace=True)
    assert (other / 'keep.txt').is_file()
    with pytest.raises(export_circuits.ExportError):
        export_circuits.export_circuits(run, run / 'public', repo_root=tmp_path)
