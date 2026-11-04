"""Sequential, resumable execution of the existing repair design.

Each model job is a separate process, releasing accelerator memory before the
next stage. The state file verifies artifact hashes before skipping work.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from .config import ROOT, load_spec, spec_digest
from .experiment import ExperimentState, atomic_json, capability_path, checkpoint_identity, file_sha256
from .repair.select import pick, quick_rates


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


class StageRunner:
    def __init__(self, directory: Path, identity: dict, python: str | None = None):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state = ExperimentState(self.directory / 'experiment.json', identity)
        self.python = python or sys.executable
        self.env = dict(os.environ, LANGSAFE_RUNS_DIR=str(self.directory),
                        LANGSAFE_DATA_DIR=str(self.directory / 'data'), PYTORCH_ENABLE_MPS_FALLBACK='1')

    def settings(self, args: list[str], inputs: list[Path] = (), optional_inputs: list[Path] = ()) -> dict:
        inputs = list(inputs)
        spec = load_spec()
        model_ids = []
        if '--model' in args:
            model_ids.append(args[args.index('--model') + 1])
        if args[0] == 'repair':
            family = args[args.index('--family') + 1]
            arm = args[args.index('--arm') + 1]
            model_ids.append(spec['models']['drifted'][family])
            if arm in ('resta', 'circuit', 'merge_back'):
                model_ids.append(spec['models']['reference'])
            if arm in ('resta', 'circuit'):
                model_ids.append(spec['models']['base'])
        if '--steering' in args:
            inputs.append(Path(args[args.index('--steering') + 1]) / 'refusal_direction.pt')
        cache_args = list(args)
        if '--server' in cache_args:
            cache_args[cache_args.index('--server') + 1] = '<local-model-server>'
        settings = {'args': cache_args, 'inputs_sha256': {str(p): file_sha256(p) for p in inputs},
                    'optional_inputs_sha256': {str(p): file_sha256(p) if p.is_file() else None for p in optional_inputs},
                    'models': {model: checkpoint_identity(model) for model in dict.fromkeys(model_ids)},
                    'spec': spec_digest(spec)}
        if args[0] == 'convert' or (args[0] == 'quantize' and '--model' in args):
            from .quant.gguf import CONVERSION_PROTOCOL
            settings['conversion_protocol'] = CONVERSION_PROTOCOL
        return settings

    def run(self, name: str, args: list[str], outputs: list[Path], inputs: list[Path] = (),
            python: str | None = None, optional_inputs: list[Path] = ()) -> None:
        settings = self.settings(args, inputs, optional_inputs)
        if self.state.completed(name, settings):
            print(f'[cached] {name}', flush=True)
            return
        print(f'[running] {name}', flush=True)
        atomic_json(self.directory / 'current-stage.json', {'stage': name, 'status': 'running', 'pid': os.getpid()})
        log = self.directory / 'logs' / f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log.open('a', encoding='utf-8') as stream:
            process = subprocess.Popen([python or self.python, '-u', '-m', 'langsafe.cli', *args],
                                       cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    stream.write(line); stream.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            failure = {'stage': name, 'exit_code': code, 'log': str(log), 'status': 'failed',
                       'elapsed_seconds': time.perf_counter() - started}
            atomic_json(self.directory / 'last-failure.json', failure)
            atomic_json(self.directory / 'current-stage.json', {**failure, 'pid': os.getpid()})
            raise RuntimeError(f'Stage {name} failed (exit {code}); inspect {log}. Rerun resumes verified stages.')
        self.state.record(name, settings, outputs, {'elapsed_seconds': time.perf_counter() - started})
        atomic_json(self.directory / 'current-stage.json', {'stage': name, 'status': 'complete', 'pid': os.getpid()})
        failure_path = self.directory / 'last-failure.json'
        if failure_path.exists() and json.loads(failure_path.read_text()).get('stage') == name:
            failure_path.unlink()

    def retire_checkpoint(self, producer_stage: str, checkpoint: Path, consumer_stage: str,
                          reason: str = 'unselected generated checkpoint after verified repair evidence') -> None:
        """Delete a generated candidate only after recording its verified consumer stage."""
        checkpoint = Path(checkpoint).resolve()
        checkpoints_root = (self.directory / 'checkpoints').resolve()
        if not checkpoint.is_relative_to(checkpoints_root):
            raise ValueError('Only checkpoints inside this experiment may be retired')
        consumer = self.state.data.get('stages', {}).get(consumer_stage)
        if consumer is None or not self.state.completed(consumer_stage, consumer['settings']):
            raise ValueError('Cannot retire a checkpoint before its consumer evidence is verified')
        self.state.retire(producer_stage, [checkpoint], reason,
                          consumer={'stage': consumer_stage, 'record': consumer})

    def reuse_retired_checkpoint(self, producer_stage: str, producer_args: list[str],
                                 checkpoint: Path, consumer_stage: str, consumer_args: list[str],
                                 inputs: list[Path] = (), optional_inputs: list[Path] = ()) -> bool:
        """Return true only when a missing checkpoint has an exact, verified cached consumer.

        This deliberately does not treat a missing weight file as a cache hit. The
        producer recipe/spec/model identities and the consumer's saved args,
        input hashes, and artifact hashes must all match a retirement receipt.
        """
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.exists():
            return False
        producer_settings = self.settings(producer_args)
        consumer_args = list(consumer_args)
        normalized = list(consumer_args)
        if '--server' in normalized:
            normalized[normalized.index('--server') + 1] = '<local-model-server>'
        for receipt in self.state.data.get('retired_stages', {}).get(producer_stage, []):
            record = receipt.get('record', {})
            consumer = receipt.get('consumer') or {}
            consumer_record = consumer.get('record') or {}
            if record.get('settings') != producer_settings:
                continue
            if receipt.get('paths') != [str(checkpoint)]:
                continue
            if consumer.get('stage') != consumer_stage:
                continue
            saved = consumer_record.get('settings', {})
            if saved.get('args') != normalized or saved.get('spec') != producer_settings.get('spec'):
                continue
            if saved.get('inputs_sha256') != {str(p): file_sha256(p) for p in inputs}:
                continue
            if saved.get('optional_inputs_sha256') != {
                    str(p): file_sha256(p) if p.is_file() else None for p in optional_inputs}:
                continue
            if not self.state.completed(consumer_stage, saved):
                continue
            return True
        return False


def _model_args(model: str, descriptor: dict | None = None) -> list[str]:
    args = ['--model', model]
    if descriptor and descriptor.get('steering'):
        args += ['--steering', descriptor['steering'], '--strength', str(descriptor['knob'])]
    return args


def _capability_stages(runner: StageRunner, tag: str, model: str, family: str, spec: dict,
                       descriptor: dict | None = None, extra_args: list[str] = (), inputs: list[Path] = ()) -> None:
    for pre in spec['preamble_conditions']:
        args = ['capability', '--tag', tag, '--family', family, '--preamble', pre,
                *_model_args(model, descriptor), *extra_args]
        if spec.get('experiment', {}).get('capability_limit'):
            args += ['--limit', str(spec['experiment']['capability_limit'])]
        runner.run(f'{tag}-capability-{pre}', args, [capability_path(runner.directory, tag, pre)], inputs)


def run_experiment(family: str, profile: str, *, stop_after: str | None = None,
                   judges: str = 'local', skip_circuit: bool = False, skip_quantization: bool = False) -> None:
    configured = os.environ.get('LANGSAFE_SPEC')
    if not configured:
        os.environ['LANGSAFE_SPEC'] = str(ROOT / 'configs' / ('development.yaml' if profile == 'development' else 'spec.yaml'))
    spec = load_spec()
    if profile == 'development' and spec.get('profile') != 'development':
        raise ValueError('development profile requires its separately marked development spec')
    if profile == 'full' and spec.get('profile') == 'development':
        raise ValueError('full run cannot use development settings')
    directory = Path(os.environ.get('LANGSAFE_RUNS_DIR', ROOT / 'runs' / 'experiments' / profile / family))
    runner = StageRunner(directory, {'family': family, 'profile': profile, 'spec': spec_digest(spec),
                                     'judges': judges, 'skip_circuit': skip_circuit, 'skip_quantization': skip_quantization})
    os.environ.update({k: runner.env[k] for k in ('LANGSAFE_RUNS_DIR', 'LANGSAFE_DATA_DIR')})
    experiments = spec.get('experiment', {})
    data_manifest = runner.directory / 'data' / 'manifest.json'
    data_outputs = [data_manifest, runner.directory / 'data' / 'heldout.jsonl', runner.directory / 'data' / 'repair.jsonl']
    runner.run('data', ['build-data', '--family', family, '--translate', '--translator', 'local'], data_outputs)
    if stop_after == 'data':
        return
    repair_limit = int(experiments.get('repair_per_cell', 20))
    batch = int(experiments.get('local_batch_size', 2))
    baseline_models = {'global': spec['models']['reference'], family: spec['models']['drifted'][family]}
    for tag, model in baseline_models.items():
        runner.run(tag + '-repair', ['generate', '--family', family, '--split', 'repair', '--model', model,
                                     '--tag', tag + '-repair', '--limit', str(repair_limit), '--preamble', 'default',
                                     '--max-new-tokens', str(experiments.get('selection_max_new_tokens', 64)),
                                     '--batch-size', str(batch)],
                   [runner.directory / 'gens' / f'{tag}-repair.jsonl'], [data_manifest])
        runner.run(tag + '-heldout', ['generate', '--family', family, '--model', model, '--tag', tag,
                                      '--batch-size', str(batch)], [runner.directory / 'gens' / f'{tag}.jsonl'], [data_manifest])
        _capability_stages(runner, tag, model, family, spec)
    if stop_after == 'baseline':
        return
    circuit_summary = None
    if not skip_circuit:
        circuit_python = os.environ.get('LANGSAFE_CIRCUIT_PYTHON')
        if not circuit_python and (ROOT / '.venv-kits' / 'bin' / 'python').exists():
            circuit_python = str(ROOT / '.venv-kits' / 'bin' / 'python')
        summaries = []
        for lang in experiments.get('circuit_languages', ['en']):
            tag = 'global-circuit'
            output = runner.directory / 'circuits' / f'{tag}__{lang}' / 'summary.json'
            runner.run('circuit-' + lang, ['circuit', '--model', baseline_models['global'], '--tag', tag,
                                          '--lang', lang, '--pairs', str(experiments.get('circuit_pairs', 32)),
                                          '--examples', str(experiments.get('circuit_examples', 16)),
                                          '--ig-steps', str(experiments.get('circuit_ig_steps', 3))],
                       [output.parent, runner.directory / 'circuit_tasks' / f'refusal_{lang}__{tag}__default.csv'],
                       python=circuit_python)
            summaries.append(json.loads(output.read_text()))
        from .circuits.discover import merge_scores
        scores = [Path(row['scores_json']) for row in summaries]
        scores = [p if p.is_absolute() else ROOT / p for p in scores]
        circuit_summary = runner.directory / 'circuits' / 'merged-scores.json'
        merge_scores(scores, circuit_summary, baseline_models['global'])
    if stop_after == 'circuit':
        return
    reference_rates = quick_rates(_rows(runner.directory / 'gens' / 'global-repair.jsonl'))
    original_rates = quick_rates(_rows(runner.directory / 'gens' / f'{family}-repair.jsonl'))
    available = {lang: values['harmful'] for lang, values in original_rates.items() if 'harmful' in values}
    if not available:
        raise ValueError('No harmful repair-split baseline available')
    worst_language = min(available, key=available.get)
    selection = {'split': 'repair', 'preamble': 'default', 'scorer': 'multilingual phrase signal (development selection only)',
                 'worst_language': worst_language, 'reference_rates': reference_rates,
                 'drifted_rates': original_rates, 'arms': {}, 'omitted': []}
    selected_descriptors = {}
    for arm in spec['repair_arms']:
        identifier = arm['id']
        if identifier == 'circuit' and skip_circuit:
            selection['omitted'].append({'arm': identifier, 'reason': 'explicit --skip-circuit'})
            continue
        candidates = {}
        descriptors = {}
        direction = runner.directory / 'checkpoints' / f'{family}-steer-default'
        if identifier == 'steer':
            runner.run('steer-direction', ['repair', '--family', family, '--arm', 'steer', '--limit', str(repair_limit)],
                       [direction / 'refusal_direction.pt', direction / 'meta.json'], [data_manifest])
        for knob in arm['values']:
            suffix = str(knob).replace('.', 'p')
            tag = f'{family}-{identifier}-{suffix}'
            descriptor = {'arm': identifier, 'knob': knob}
            if identifier == 'resta':
                checkpoint = runner.directory / 'checkpoints' / f'{family}-resta-a{knob}'
            elif identifier == 'circuit':
                checkpoint = runner.directory / 'checkpoints' / f'{family}-circuit-s{knob}-top{arm.get("top_fraction", 0.2)}'
            elif identifier == 'merge_back':
                checkpoint = runner.directory / 'checkpoints' / f'{family}-merge-l{knob}'
            else:
                checkpoint = Path(baseline_models[family])
                descriptor['steering'] = str(direction)
            if identifier != 'steer':
                args = ['repair', '--family', family, '--arm', identifier, '--knob', str(knob)]
                inputs = []
                if identifier == 'circuit':
                    args += ['--scores', str(circuit_summary), '--top-fraction', str(arm.get('top_fraction', 0.2))]
                    inputs = [circuit_summary]
                runner.run(tag + '-weights', args, [checkpoint], inputs)
            model_id = baseline_models[family] if identifier == 'steer' else str(checkpoint)
            descriptor.update(model=model_id, tag=tag)
            descriptors[knob] = descriptor
            args = ['generate', '--family', family, '--split', 'repair', '--tag', tag + '-repair',
                    '--limit', str(repair_limit), '--preamble', 'default',
                    '--max-new-tokens', str(experiments.get('selection_max_new_tokens', 64)), '--batch-size', str(batch),
                    *_model_args(model_id, descriptor)]
            artifact = runner.directory / 'gens' / f'{tag}-repair.jsonl'
            runner.run(tag + '-repair', args, [artifact], [data_manifest])
            candidates[knob] = quick_rates(_rows(artifact))
        chosen, reason = pick(candidates, reference_rates, original_rates, spec['thresholds']['over_refusal_increase_max'])
        selection['arms'][identifier] = {'knob': chosen, 'reason': reason,
                                        'candidates': {str(k): v for k, v in candidates.items()},
                                        'selected': descriptors[chosen]}
        selected_descriptors[identifier] = descriptors[chosen]
        atomic_json(runner.directory / 'selection.json', selection)
    # This comparison uses repair data only; held-out evaluations never choose a candidate.
    def score(descriptor):
        rates = selection['arms'][descriptor['arm']]['candidates'][str(descriptor['knob'])]
        harmful = min(v.get('harmful', 0) for v in rates.values())
        penalty = max([max(0, rates[l].get('benign', 0) - original_rates.get(l, {}).get('benign', 0)) for l in rates] or [0])
        return harmful - 2 * penalty
    if not selected_descriptors:
        raise ValueError('No repair arms were evaluated')
    # A steering vector cannot be represented by a plain quantized checkpoint.
    weight_candidates = [d for d in selected_descriptors.values() if not d.get('steering')]
    winner = max(weight_candidates or list(selected_descriptors.values()), key=score)
    selection['selected_for_quantization'] = winner
    atomic_json(runner.directory / 'selection.json', selection)
    for descriptor in selected_descriptors.values():
        tag = descriptor['tag']
        runner.run(tag + '-heldout', ['generate', '--family', family, '--tag', tag, '--batch-size', str(batch),
                                     *_model_args(descriptor['model'], descriptor)],
                   [runner.directory / 'gens' / f'{tag}.jsonl'], [data_manifest])
        _capability_stages(runner, tag, descriptor['model'], family, spec, descriptor)
    evaluation_tags = ['global', family] + [d['tag'] for d in selected_descriptors.values()]
    for tag in evaluation_tags:
        path = runner.directory / 'judged' / (f'{tag}.jsonl' if judges == 'router' else f'{tag}.{judges}.jsonl')
        runner.run(tag + '-judge-' + judges, ['judge', '--tag', tag, '--judges', judges],
                   [path, path.with_suffix('.summary.json')], [runner.directory / 'gens' / f'{tag}.jsonl'])
        runner.run(tag + '-audit-' + judges, ['audit', '--tag', tag, '--judges', judges],
                   [runner.directory / 'auditkit' / f'{tag}.{judges}'],
                   [path, runner.directory / 'gens' / f'{tag}.jsonl'])
    if stop_after == 'repairs':
        return
    if not skip_quantization:
        if winner.get('steering'):
            raise ValueError('Only steering was available; no compatible weight checkpoint for Q4 export')
        server_binary = os.environ.get('LANGSAFE_LLAMA_SERVER') or shutil.which('llama-server')
        if not server_binary:
            server_binary = str(ROOT / 'cache' / 'llama.cpp' / 'build' / 'bin' / 'llama-server')
        # All three precision-matched checkpoints are evaluated on the same inputs.
        quantization_models = [*baseline_models.items(), (winner['tag'], winner['model'])]
        for source_tag, source_model in quantization_models:
            fp16 = runner.directory / 'quantized' / (source_tag + '-f16.gguf')
            quantized = runner.directory / 'quantized' / (source_tag + '-q4_k_m.gguf')
            runner.run(source_tag + '-convert', ['convert', '--model', source_model,
                                                '--output', str(fp16), '--overwrite'], [fp16])
            runner.run(source_tag + '-quantize', ['quantize', '--input', str(fp16),
                                                 '--output', str(quantized), '--overwrite'], [quantized], [fp16])
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            endpoint = f'http://127.0.0.1:{port}'
            logfile = (runner.directory / 'logs' / f'llama-server-{source_tag}.log').open('a')
            server = subprocess.Popen([server_binary, '-m', str(quantized), '--port', str(port),
                                       '--host', '127.0.0.1', '-c', '8192', '-ngl', '99'],
                                      stdout=logfile, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if server.poll() is not None:
                        raise RuntimeError(f'llama-server exited; inspect its {source_tag} experiment log')
                    try:
                        with urlopen(endpoint + '/health', timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, ValueError):
                        time.sleep(1)
                else:
                    raise RuntimeError('llama-server did not become ready')
                tag = source_tag + '-q4'
                extra = ['--gguf', str(quantized), '--server', endpoint]
                runner.run(tag + '-heldout', ['generate', '--family', family, '--tag', tag,
                                             '--model', source_model, *extra],
                           [runner.directory / 'gens' / f'{tag}.jsonl'], [data_manifest, quantized])
                _capability_stages(runner, tag, source_model, family, spec, extra_args=extra, inputs=[quantized])
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                logfile.close()
            path = runner.directory / 'judged' / (f'{tag}.jsonl' if judges == 'router' else f'{tag}.{judges}.jsonl')
            runner.run(tag + '-judge', ['judge', '--tag', tag, '--judges', judges], [path, path.with_suffix('.summary.json')],
                       [runner.directory / 'gens' / f'{tag}.jsonl'])
            runner.run(tag + '-audit', ['audit', '--tag', tag, '--judges', judges],
                       [runner.directory / 'auditkit' / f'{tag}.{judges}'],
                       [path, runner.directory / 'gens' / f'{tag}.jsonl'])
            evaluation_tags.append(tag)
    if stop_after == 'quantization':
        return
    from .experiment_report import report_dependencies
    runner.run('report', ['report', '--tags', ','.join(evaluation_tags), '--family', family, '--judges', judges],
               [runner.directory / 'report.md', runner.directory / 'report.json'],
               optional_inputs=report_dependencies(runner.directory, evaluation_tags, judges, spec['preamble_conditions']))
    runner.run('ledger-verify', ['ledger-verify'], [runner.directory / 'ledger.jsonl'])
    report = json.loads((runner.directory / 'report.json').read_text())
    status = {'software_run_complete': True, 'profile': profile, 'family': family,
              'evaluated_tags': evaluation_tags, 'scientific_validation': 'provisional pending real human/router calibration',
              'all_four_repair_arms_evaluated': len(selected_descriptors) == 4,
              'empirical_repair_success': report['empirical_repair_success'],
              'validation_gaps': report['validation_gaps'], 'spec_frozen': spec.get('frozen', False),
              'nf4_cuda': 'not_executed', 'q4_evaluated': not skip_quantization,
              'report': str(runner.directory / 'report.md')}
    atomic_json(runner.directory / 'status.json', status)
    print(json.dumps(status, indent=2), flush=True)
