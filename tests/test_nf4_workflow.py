"""Offline contracts for the CUDA handoff; no model, GPU, or network is used."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'check_nf4_environment.py'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def handoff(tmp_path):
    source = tmp_path / 'source'
    models = tmp_path / 'models'
    for name in ('global', 'fire'):
        checkpoint = models / name
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{"model_type":"cohere2"}')
        (checkpoint / 'model.safetensors').write_bytes(b'fixture weights, never loaded')
        (checkpoint / 'tokenizer.json').write_text('{}')
        (checkpoint / 'tokenizer_config.json').write_text('{}')
    checkpoint = source / 'checkpoints' / 'fire-resta-a0.1'
    checkpoint.mkdir(parents=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        (checkpoint / name).write_bytes((models / 'fire' / name).read_bytes())
    spec = {
        'spec_version': 1, 'frozen': False,
        'models': {'reference': 'Fixture/global', 'drifted': {'fire': 'Fixture/fire'}},
        'languages': {'fire': ['en']}, 'capability_languages': ['en'],
        'preamble_conditions': ['default', 'none'],
        'decoding': {'dtype': 'float16', 'do_sample': False, 'max_new_tokens': 256, 'batch_size': 2},
        'data': {'seed': 1}, 'experiment': {'local_batch_size': 2},
        'judging': {'local_judges': [
            {'name': 'first', 'model': 'Fixture/global', 'base_url': 'local'},
            {'name': 'second', 'model': 'Fixture/fire', 'base_url': 'local'}]},
        'quantization': ['nf4', 'gguf_q4_k_m'], 'thresholds': {'quantized_must_hold': True},
    }
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(spec))
    data = source / 'data'
    heldout = [dict(id=f'heldout-{kind}::en', group=f'heldout-{kind}', lang='en', kind=kind,
                    text=f'{kind} held-out fixture', source='fixture', source_id=kind, split='heldout')
               for kind in ('harmful', 'benign')]
    repair = [dict(row, id=row['id'].replace('heldout', 'repair'),
                   group=row['group'].replace('heldout', 'repair'), split='repair') for row in heldout]
    _write_rows(data / 'heldout.jsonl', heldout)
    _write_rows(data / 'repair.jsonl', repair)
    manifest = {'spec_digest': _digest(spec), 'family': 'fire', 'languages': ['en'],
                'coverage': {'complete': True}, 'leakage_check': {'ok': True},
                'files': {split: {'sha256': hashlib.sha256((data / f'{split}.jsonl').read_bytes()).hexdigest(),
                                 'rows': 2} for split in ('heldout', 'repair')}}
    (data / 'manifest.json').write_text(json.dumps(manifest))
    selected = {'tag': 'fire-resta-0p1', 'arm': 'resta', 'knob': 0.1,
                'model': '/original-machine/run/checkpoints/fire-resta-a0.1'}
    selection = {'split': 'repair', 'preamble': 'default', 'worst_language': 'en',
                 'arms': {'resta': {'selected': selected}}, 'selected_for_quantization': selected}
    (source / 'selection.json').write_text(json.dumps(selection))
    for tag in ('global', 'fire', 'fire-resta-0p1'):
        gens = [dict(row, preamble=pre, model_tag=tag, fingerprint=tag, response='fixture',
                     rendered_prompt_sha256=f"{row['id']}-{pre}") for row in heldout for pre in ('default', 'none')]
        _write_rows(source / 'gens' / f'{tag}.jsonl', gens)
        _write_rows(source / 'judged' / f'{tag}.local.jsonl', [dict(row, label='unclear') for row in gens])
        (source / 'judged' / f'{tag}.local.summary.json').write_text(json.dumps({'judges': ['first', 'second']}))
        for pre in ('default', 'none'):
            suffix = '' if pre == 'default' else '.none'
            path = source / 'capability' / f'{tag}{suffix}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'complete': True, 'metadata': {'quant': None, 'preamble': pre},
                                       'languages': {'en': {'data_sha256': 'same-items', 'n': 2, 'per_item': [1, 0]}}}))
    return {'source': source, 'models': models, 'spec': spec_path, 'output': tmp_path / 'remote-nf4'}


def run_handoff(handoff, *args):
    env = dict(os.environ, LANGSAFE_MODELS=str(handoff['models']), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return subprocess.run([sys.executable, str(SCRIPT), '--source-run', str(handoff['source']),
                           '--spec', str(handoff['spec']), '--output', str(handoff['output']), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)


def test_offline_plan_preserves_full_prompts_and_precision_matched_trio(handoff):
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['status'] == 'planned_not_executed'
    assert plan['profile'] == 'full'
    assert plan['hardware_checked'] is False
    commands = [stage['args'] for stage in plan['stages']]
    generations = [args for args in commands if args[0] == 'generate']
    assert [args[args.index('--tag') + 1] for args in generations] == ['global-nf4', 'fire-nf4', 'fire-resta-0p1-nf4']
    assert all(args[args.index('--quant') + 1] == 'nf4' and '--limit' not in args and '--preamble' not in args
               for args in generations)
    capabilities = [args for args in commands if args[0] == 'capability']
    assert len(capabilities) == 6
    assert {args[args.index('--preamble') + 1] for args in capabilities} == {'default', 'none'}
    assert all('--quant' in args and '--limit' not in args for args in capabilities)
    assert len([args for args in commands if args[0] == 'judge' and '--judges' in args]) == 3
    assert len([args for args in commands if args[0] == 'audit']) == 3
    report = next(args for args in commands if args[0] == 'report')
    assert report[report.index('--tags') + 1] == 'global,fire,fire-resta-0p1,global-nf4,fire-nf4,fire-resta-0p1-nf4'
    assert not handoff['output'].exists()
    assert plan['resolved_models']['fire-resta-0p1'] == str(handoff['source'] / 'checkpoints' / 'fire-resta-a0.1')
    assert plan['model_refs']['global'] == 'Fixture/global'
    assert plan['source_evidence']['selected_checkpoint']['tag'] == 'fire-resta-0p1'
    assert plan['source_evidence']['selected_checkpoint']['files']
    selected_generation = next(args for args in generations if args[args.index('--tag') + 1] == 'fire-resta-0p1-nf4')
    assert selected_generation[selected_generation.index('--model') + 1] == str(handoff['source'] / 'checkpoints' / 'fire-resta-a0.1')
    assert plan['code']['content_sha256'] and plan['source_evidence']['files']
    assert plan['selected_checkpoint_target'].endswith('/checkpoints/fire-resta-0p1')


@pytest.mark.parametrize('mutation,expected', [
    ('data', 'hash'), ('spec', 'spec'), ('selection', 'repair'), ('preambles', 'preamble'),
    ('precision', 'precision'), ('prompts', 'prompt'), ('occupied', 'empty'), ('nested', 'overlap')])
def test_handoff_fails_closed_before_gpu_work(handoff, mutation, expected):
    if mutation == 'data':
        with (handoff['source'] / 'data' / 'heldout.jsonl').open('a') as stream:
            stream.write('\n')
    elif mutation in {'spec', 'preambles'}:
        spec = yaml.safe_load(handoff['spec'].read_text())
        if mutation == 'spec':
            spec['data']['seed'] = 9
        else:
            spec['preamble_conditions'] = ['default']
        handoff['spec'].write_text(yaml.safe_dump(spec))
    elif mutation == 'selection':
        path = handoff['source'] / 'selection.json'
        value = json.loads(path.read_text()); value['split'] = 'heldout'; path.write_text(json.dumps(value))
    elif mutation == 'precision':
        path = handoff['source'] / 'capability' / 'global.json'
        value = json.loads(path.read_text()); value['metadata']['quant'] = 'nf4'; path.write_text(json.dumps(value))
    elif mutation == 'prompts':
        path = handoff['source'] / 'gens' / 'fire.jsonl'
        path.write_text(path.read_text().splitlines()[0] + '\n')
    elif mutation == 'occupied':
        handoff['output'].mkdir(); (handoff['output'] / 'unrelated').write_text('keep me')
    elif mutation == 'nested':
        handoff['output'] = handoff['source'] / 'nf4'
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0
    assert expected in result.stderr.lower()
    assert not (handoff['output'] / 'gens').exists()


def test_source_generation_quantization_metadata_cannot_be_mixed(handoff):
    path = handoff['source'] / 'gens' / 'global.jsonl'
    value = json.loads(path.read_text().splitlines()[0]); value['variant'] = {'quant': 'nf4'}
    path.write_text(json.dumps(value) + '\n' + '\n'.join(path.read_text().splitlines()[1:]) + '\n')
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0 and 'full precision' in result.stderr.lower()


def test_development_inputs_require_explicit_label(handoff):
    spec = yaml.safe_load(handoff['spec'].read_text()); spec['profile'] = 'development'
    handoff['spec'].write_text(yaml.safe_dump(spec))
    path = handoff['source'] / 'data' / 'manifest.json'
    manifest = json.loads(path.read_text()); manifest['spec_digest'] = _digest(spec); path.write_text(json.dumps(manifest))
    denied = run_handoff(handoff, '--dry-run')
    assert denied.returncode != 0 and 'development' in denied.stderr.lower()
    allowed = run_handoff(handoff, '--dry-run', '--allow-development')
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)['profile'] == 'development'


def test_missing_cached_checkpoint_is_an_explicit_offline_blocker(handoff):
    (handoff['models'] / 'global' / 'model.safetensors').unlink()
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0
    assert 'checkpoint' in result.stderr.lower()


def test_nf4_model_validation_rejects_cpu_or_disk_offload():
    from langsafe.gen.model import validate_nf4_model

    class Offloaded:
        is_loaded_in_4bit = True
        hf_device_map = {'': 'cuda', 'lm_head': 'cpu'}

    with pytest.raises(RuntimeError, match='CUDA'):
        validate_nf4_model(Offloaded())


def test_nf4_validation_rejects_unquantized_or_unplaced_models():
    from types import SimpleNamespace
    from langsafe.gen.model import validate_nf4_model

    with pytest.raises(RuntimeError, match='4-bit'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=False, hf_device_map={'': 'cuda'}))
    with pytest.raises(RuntimeError, match='CUDA'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, hf_device_map={'lm_head': 'cpu'}))
    with pytest.raises(RuntimeError, match='device map'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, hf_device_map=None))
    quantizer = SimpleNamespace(quantization_config=SimpleNamespace(_load_in_4bit=True, bnb_4bit_quant_type='nf4'))
    validate_nf4_model(SimpleNamespace(hf_quantizer=quantizer, hf_device_map={'': 0}))
    with pytest.raises(RuntimeError, match='quantizer type'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True,
                                           hf_quantizer=SimpleNamespace(quantization_config=SimpleNamespace(
                                               _load_in_4bit=True, bnb_4bit_quant_type='fp4')),
                                           hf_device_map={'': 0}))


def test_nf4_load_requires_cuda_before_tokenizer_or_checkpoint_loading(monkeypatch):
    from types import SimpleNamespace
    from langsafe.gen import model

    def forbidden(*args, **kwargs):
        pytest.fail('A CUDA preflight failure must precede all checkpoint/tokenizer loading')

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=forbidden),
        AutoTokenizer=SimpleNamespace(from_pretrained=forbidden)))
    monkeypatch.setenv('LANGSAFE_DEVICE', 'cuda')
    with pytest.raises(RuntimeError, match='CUDA'):
        model.load('must-not-load', quant='nf4')


def test_nf4_cuda_map_and_quantization_settings_are_explicit(monkeypatch):
    from types import SimpleNamespace
    from langsafe.gen import model

    loaded = SimpleNamespace(is_loaded_in_4bit=True, hf_device_map={'': 'cuda:0'},
                             named_parameters=lambda: iter([('weight', SimpleNamespace(device=SimpleNamespace(type='cuda')))]),
                             eval=lambda: None)
    capture = {}
    def from_pretrained(path, **kwargs):
        capture.update(kwargs)
        return loaded
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        float16='fake-fp16', cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args: 'fake-tokenizer'),
        BitsAndBytesConfig=lambda **kwargs: kwargs))
    monkeypatch.setenv('LANGSAFE_DEVICE', 'cuda')
    actual, tokenizer = model.load('offline-fixture', quant='nf4')
    assert actual is loaded and tokenizer == 'fake-tokenizer'
    assert capture['device_map'] == {'': 0}
    assert capture['quantization_config'] == {
        'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'fake-fp16',
        'bnb_4bit_use_double_quant': False}
    with pytest.raises(RuntimeError, match='strict single-CUDA'):
        model.load('offline-fixture', quant='nf4', device_map='auto')


def test_plan_commands_are_accepted_by_existing_cli(handoff, monkeypatch):
    from langsafe import cli

    result = run_handoff(handoff, '--dry-run')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    called = []
    for name in ('generate', 'capability', 'judge', 'audit', 'report', 'ledger_verify'):
        monkeypatch.setattr(cli, 'cmd_' + name, lambda args: called.append(args))
    monkeypatch.setenv('LANGSAFE_SPEC', str(handoff['spec']))
    for stage in plan['stages']:
        cli.main(['--spec', str(handoff['spec']), *stage['args']])
    assert len(called) == len(plan['stages'])


def test_router_execution_requires_explicit_network_authorization_before_work(handoff):
    result = run_handoff(handoff, '--execute', '--judges', 'router')
    assert result.returncode != 0
    assert '--allow-router' in result.stderr
    assert not handoff['output'].exists()
