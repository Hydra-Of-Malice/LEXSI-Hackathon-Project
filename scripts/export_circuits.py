#!/usr/bin/env python3
"""Write the public, layer-level copy of a run's refusal circuits, then prove it holds no model text.

Usage: .venv/Scripts/python.exe scripts/export_circuits.py [RUN_DIR] [OUT_DIR] [--replace]
Defaults: runs/experiments/full/earth -> runs/experiments/public/circuits

Team decision D7 (docs/submission/decisions.md): publish the circuits' layer lists, summary metrics and the editing
mask, which are enough to reproduce every repair. Keep the node-level score files private and share them with
researchers on request. So the export holds layer-level numbers and aggregates only:

  circuits/<model>-circuit__<lang>.json   one per circuit: settings, node counts, per-layer score mass and rank, the
                                          layers the repair rule picks, faithfulness pillars, the fair random-head
                                          baseline, the HF/TransformerLens parity check, the private score file's sha256
  editing_mask.json                       the mask the circuit repairs used: its score file, layers, modules and the
                                          exact parameter names
  repair_arms.json                        every circuit-restricted checkpoint: vector, strength, edited layers and
                                          parameters, edit size, and the checks below
  overlap.json                            top-50 node Jaccard and full-ranking Spearman between circuits
  README.md, EXPORT_MANIFEST.json         what is and is not here; every source file with its sha256 and what became of it

Nothing is copied wholesale: every published value is picked by name and type-checked. Node-level scores, CircuitKIT's
pruning artifacts, the merged score file, the stability pillar's node sets, the circuit task tables (minimal pairs of
harmful and harmless requests) and the answer openers stay private.

The layer lists come from the repair code, not a re-implementation. For each checkpoint, the score file its provenance
names is checked against the recorded sha256, `langsafe.circuits.discover.top_layers` recomputes the circuit layers,
the random-layer control's draw is repeated as `langsafe.repair.arms.circuit` makes it, and
`langsafe.repair.arms._circuit_keys` lists the edited parameters from the checkpoint's own tensor names. Any
disagreement with the provenance blocks the export.

As in scripts/export_public.py, the files are written to a staging folder and re-read by its final scan (`scan_output`:
model answers from the run's private generations, forbidden keys, local paths), plus checks for node names, long
numeric arrays and this machine's paths. OUT_DIR appears only if everything passes. Nothing here prints model text or
prompts, and the run folder is only read.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

from langsafe.circuits.discover import layer_mass, node_scores, top_layers
from langsafe.repair.arms import _circuit_keys, circuit_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
OUT = ROOT / 'runs' / 'experiments' / 'public' / 'circuits'
MANIFEST_NAME = 'EXPORT_MANIFEST.json'
SCHEMA = 'langsafe.public-circuits/1'
DEFAULT_TOP_FRACTION = 0.2      # configs/spec.yaml, used when no circuit arm was built
MAX_NUMBERS = 100               # a per-layer table fits; a per-node vector (612 values in Tiny Aya) does not
MAX_HEADER = 100_000_000        # a safetensors header larger than this is not a checkpoint we wrote


def _script(name: str):
    """Load a sibling script as a module: scripts/ is not a package."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f'{name}.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


export_public = _script('export_public')
fair_circuit_baseline = _script('fair_circuit_baseline')
fill_submission = _script('fill_submission')
publish_hf = _script('publish_hf')
file_sha256 = export_public.file_sha256
ExportError = export_public.ExportError

# CircuitKIT's node names: A<layer>.<head>, MLP <layer>, L<layer>.<neuron>; its stability pillar writes a<l>.h<h>, m<l>.
NODE_NAME = re.compile(r'A\d+\.\d+|MLP \d+|L\d+\.\d+|a\d+\.h\d+|m\d+')
NODE_IN_TEXT = re.compile(r'(?<![\w.])(?:A\d+\.\d+|MLP \d+|a\d+\.h\d+)(?![\w.])')
_SHORT_TEXT = re.compile(r'[\w .:+\-/#@(),;=]+')
_MATRIX_KEY = re.compile(r'[\w\-]+/[\w\-]+')
_CIRCUIT = re.compile(r'([\w\-]+-circuit)__([\w\-]+)')
_TASK_PREAMBLE = re.compile(r'refusal_[\w\-]+__[\w.\-]+__(\w+)\.csv')

# Why each kind of source file stays private. Every file under circuits/ and circuit_tasks/ is listed in the manifest.
WITHHELD = (
    (re.compile(r'.*_node_scores\.(?:json|pt)'), 'node-level scores: private under D7, shared with researchers on request'),
    (re.compile(r'.*_node\.pt'), "CircuitKIT's pruning artifact, a node-level mask: private under D7"),
    (re.compile(r'(?:.*/)?merged-scores\.json'), 'node-level scores averaged over the circuits the mask came from: private '
                                                 'under D7'),
    (re.compile(r'.*\.csv'), 'circuit task table: minimal pairs of harmful and harmless requests, and the answer openers '
                             'taken from model answers'),
    (re.compile(r'(?:.*/)?summary\.json'), "holds the stability pillar's node sets and per-node scores, and the answer "
                                           'openers'),
    (re.compile(r'(?:.*/)?fair_baseline\.json'), 'copied field by field'),
    (re.compile(r'(?:.*/)?overlap\.json'), 'copied field by field'),
)
# Fields of a circuit's summary.json that are deliberately not published.
SUMMARY_FIELDS_LEFT_OUT = {
    'faithfulness.stability.circuits': 'node sets found by each stability run (node-level)',
    'faithfulness.stability.raw_scores': 'per-node scores from each stability run (node-level)',
    'faithfulness.stability.dice_matrix, spearman_matrix': 'summarised by their means and standard deviations',
    'faithfulness.baseline_comparison.summary': "CircuitKIT's free-text verdict; the numbers are kept",
    'faithfulness.metadata.timestamp, per_pillar_duration_seconds': 'not needed to check the circuit',
    'parity.per_prompt': 'reduced to the largest divergence over the parity sentences',
    'openers': "the refusal and compliance opener tokens, taken from Global's answers on the repair split",
    'resolved_model, scores_json': 'replaced by the model id and a repo-relative path',
}
VECTORS = {
    'safety': ('circuit', 'drifted + strength x (reference - base): the ReSta safety vector, only on the listed '
                          'parameters'),
    'drift': ('circuit_merge', 'drifted + strength x (reference - drifted): the merge-back vector, only on the listed '
                               'parameters'),
}
ARITHMETIC = ('SafeTune apply_ctheta: for each listed weight, W = W_drifted + strength x (W_positive - W_negative), '
              'computed in float16 as the models were loaded; every other weight is the drifted model\'s.')


# ---- value checks: each returns the clean value or None ---------------------------------------------------------

def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _flag(value):
    return value if isinstance(value, bool) else None


def _name(value):
    return value if export_public._is_name(value) else None


def _digest(value):
    return value if isinstance(value, str) and export_public._HEX.fullmatch(value) else None


def _timestamp(value):
    return value if isinstance(value, str) and export_public._TIMESTAMP.fullmatch(value) else None


def _stem(value) -> str:
    """The file name of a recorded path without its suffix, whichever OS wrote it."""
    return str(value).replace('\\', '/').rsplit('/', 1)[-1].removesuffix('.json')


def _text(value, limit: int = 160):
    """A short one-line string written by our own tools or CircuitKIT, never model text."""
    ok = isinstance(value, str) and len(value) <= limit and _SHORT_TEXT.fullmatch(value) \
        and not export_public._LOCAL_PATH.search(value)
    return value if ok else None


def _numbers(value, limit: int = MAX_NUMBERS):
    if not isinstance(value, list) or len(value) > limit:
        return None
    clean = [_number(item) for item in value]
    return None if any(item is None for item in clean) else clean


def _names(value):
    return [item for item in value if _name(item)] if isinstance(value, list) else None


def _pick(source, rules: dict) -> dict:
    """{key: rule(source[key])} for the named keys only."""
    source = source if isinstance(source, dict) else {}
    return {key: rule(source.get(key)) for key, rule in rules.items()}


# ---- paths --------------------------------------------------------------------------------------------------------

def repo_relative(value, repo_root: Path) -> str | None:
    """A path relative to the repo root, or None when it cannot safely be made one.

    Paths written on another machine keep what follows the last `runs` folder, Hugging Face cache paths become the repo
    id, and anything else outside the repo keeps only its file name under `external/`."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace('\\', '/').rstrip('/')
    root = str(repo_root).replace('\\', '/').rstrip('/')
    if text.casefold().startswith(root.casefold() + '/'):
        text = text[len(root) + 1:]
    elif export_public._ABSOLUTE.match(text):
        parts = [part for part in text.split('/') if part]
        lowered = [part.casefold() for part in parts]
        anchor = Path(root).name.casefold()
        hub = next((part for part in parts if part.startswith('models--')), None)
        if hub:
            text = hub[len('models--'):].replace('--', '/')
        elif anchor in lowered:
            text = '/'.join(parts[len(lowered) - lowered[::-1].index(anchor):])
        elif 'runs' in lowered:
            text = '/'.join(parts[len(lowered) - 1 - lowered[::-1].index('runs'):])
        else:
            text = f'external/{parts[-1]}' if parts else ''
    safe = export_public._RELATIVE_PATH.fullmatch(text) and '..' not in text.split('/') \
        and not export_public._LOCAL_PATH.search(text)
    return text if safe else None


def _locate(value, repo_root: Path) -> Path | None:
    """The file a recorded path points to on this machine, if it is here."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    relative = repo_relative(value, repo_root)
    return repo_root / relative if relative and (repo_root / relative).is_file() else None


def _model_ref(value, repo_root: Path) -> str | None:
    """A model reference: a hub id as it is, a local folder made repo-relative."""
    if isinstance(value, str) and not export_public._ABSOLUTE.match(value.replace('\\', '/')) and _name(value):
        return value
    return repo_relative(value, repo_root)


# ---- what the repair code computes ----------------------------------------------------------------------------------

def layer_table(values: dict[str, float], fractions: list[float]) -> dict:
    """Per-layer mass and rank, and the layers `top_layers` picks at each fraction."""
    mass = layer_mass(values)
    total = sum(mass.values())
    order = sorted(mass, key=lambda layer: (-mass[layer], layer))       # top_layers' own ordering
    rank = {layer: position + 1 for position, layer in enumerate(order)}
    selections = []
    for fraction in fractions:
        layers = top_layers(values, fraction)
        if set(layers) != {layer for layer in mass if rank[layer] <= len(layers)}:
            raise ValueError('layer ranking disagrees with top_layers')
        selections.append({'top_fraction': fraction, 'layers': layers, 'layer_count': len(layers)})
    return {'layers_in_model': max(mass) + 1,
            'mass_rule': 'sum of absolute node scores per layer (langsafe.circuits.discover.layer_mass)',
            'selection_rule': 'top round(layers x top_fraction) layers by mass, ties to the lower layer '
                              '(langsafe.circuits.discover.top_layers)',
            'selected': selections,
            'ranking': [{'layer': layer, 'mass': mass[layer], 'share': mass[layer] / total if total else None,
                         'rank': rank[layer]} for layer in sorted(mass)]}


def node_counts(values: dict[str, float], sparsity) -> dict:
    kinds = {'attention_heads': r'A\d+\.\d+', 'mlp_blocks': r'MLP \d+', 'neurons': r'L\d+\.\d+'}
    counts = {kind: sum(bool(re.fullmatch(pattern, name)) for name in values) for kind, pattern in kinds.items()}
    sparsity = _number(sparsity)
    return {'total': len(values), **counts, 'other': len(values) - sum(counts.values()),
            'kept_at_sparsity': None if sparsity is None else int(len(values) * (1 - sparsity))}


def control_layers(values: dict[str, float], circuit: list[int], seed: int) -> list[int]:
    """The random-layer control's draw, exactly as langsafe.repair.arms.circuit makes it."""
    outside = sorted(set(layer_mass(values)) - set(circuit))
    return sorted(random.Random(seed).sample(outside, len(circuit)))


def _by_layer(name: str) -> tuple:
    """Sort weight names by layer number (the first number in the name, as `_circuit_keys` reads it), then by name."""
    return next((int(part) for part in name.split('.') if part.isdigit()), -1), name


class _TensorNames:
    """Enough of a model for `_circuit_keys`: its state-dict names, without any weights."""

    def __init__(self, names: list[str]):
        self.names = names

    def state_dict(self) -> dict:
        return dict.fromkeys(self.names)


def tensor_names(checkpoint: Path) -> list[str] | None:
    """Tensor names from the checkpoint's safetensors header or index. No weights are read."""
    index = checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        return sorted(json.loads(index.read_text(encoding='utf-8')).get('weight_map') or {}) or None
    single = checkpoint / 'model.safetensors'
    if not single.is_file():
        return None
    with single.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if size > MAX_HEADER:
            raise ValueError('safetensors header is too large')
        header = json.loads(stream.read(size))
    return sorted(name for name in header if name != '__metadata__') or None


# ---- the export -------------------------------------------------------------------------------------------------

class _Export:
    def __init__(self, run: Path, staging: Path, repo_root: Path):
        self.run, self.staging, self.repo_root = run, staging, repo_root
        self.exported: list[dict] = []
        self.read: dict[str, dict] = {}      # source file -> what became of it
        self.failures: list[dict] = []
        self.unverified: list[dict] = []
        self.scores_cache: dict[Path, dict] = {}

    def rel(self, path: Path) -> str:
        return repo_relative(str(path), self.repo_root) or f'external/{path.name}'

    def note(self, path: Path, status: str, detail: str, into: str | None = None, digest: bool = True) -> None:
        entry = self.read.setdefault(self.rel(path), {'sha256': file_sha256(path) if digest else None,
                                                      'status': status, 'detail': detail, 'into': []})
        if into and into not in entry['into']:
            entry['into'].append(into)

    def fail(self, check: str, where: str, detail: str) -> None:
        self.failures.append({'check': check, 'where': where, 'detail': detail})

    def scores(self, path: Path) -> dict[str, float]:
        if path not in self.scores_cache:
            self.scores_cache[path] = node_scores(path)
        return self.scores_cache[path]

    def write(self, target: str, document, sources: list[str]) -> None:
        path = self.staging / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(document, str):
            path.write_text(document, encoding='utf-8', newline='\n')
        else:
            with path.open('w', encoding='utf-8', newline='\n') as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        self.exported.append({'file': target, 'sha256': file_sha256(path), 'bytes': path.stat().st_size,
                              'from': sorted(set(sources))})

    # -- selection.json: which checkpoint each arm chose --

    def roles(self) -> tuple[dict[str, dict], str | None]:
        path = self.run / 'selection.json'
        if not path.is_file():
            return {}, None
        selection = json.loads(path.read_text(encoding='utf-8-sig'))
        self.note(path, 'read', 'which candidate each arm selected on the repair split, and the controls',
                  'repair_arms.json')
        roles = {}
        for entry in (selection.get('arms') or {}).values():
            chosen = (entry or {}).get('selected') or {}
            if isinstance(chosen.get('model'), str):
                roles[publish_hf.checkpoint_folder(chosen['model'])] = {'role': 'selected on the repair split',
                                                                        'tag': _name(chosen.get('tag'))}
        for control in selection.get('controls') or []:
            if isinstance(control, dict) and isinstance(control.get('model'), str):
                roles[publish_hf.checkpoint_folder(control['model'])] = {'role': 'random-layer control',
                                                                         'tag': _name(control.get('tag'))}
        quantized = (selection.get('selected_for_quantization') or {}).get('model')
        return roles, publish_hf.checkpoint_folder(quantized) if isinstance(quantized, str) else None

    # -- repair arms: one entry per circuit-restricted checkpoint --

    def arms(self) -> list[dict]:
        roles, quantized = self.roles()
        entries = []
        for provenance in sorted((self.run / 'checkpoints').glob('*/lexsi_provenance.json')):
            document = json.loads(provenance.read_text(encoding='utf-8-sig'))
            if document.get('method') != 'langsafe.ctheta':
                continue
            self.note(provenance, 'read', 'repair provenance; local paths made repo-relative', 'repair_arms.json')
            entries.append(self.arm(provenance.parent, document, roles, quantized))
        return entries

    def arm(self, folder: Path, document: dict, roles: dict, quantized: str | None) -> dict:
        params = document.get('params') or {}
        where = folder.name
        strength, top_fraction = _number(params.get('strength')), _number(params.get('top_fraction'))
        vector, seed = params.get('vector', 'safety'), params.get('random_seed')
        layer_subset, modules = _numbers(params.get('layer_subset')), _names(params.get('target_modules'))
        recorded = _numbers(params.get('circuit_layers')) or (layer_subset if seed is None else None)
        if vector not in VECTORS or strength is None or top_fraction is None or not layer_subset or not modules \
                or (seed is not None and type(seed) is not int):
            self.fail('provenance', where, 'missing or invalid strength, top_fraction, vector, layers or modules')
            return {'checkpoint': where}
        arm_id, rule = VECTORS[vector]
        family = folder.name.split('-circuit')[0]
        knob = str(strength).replace('.', 'p')
        role = roles.get(folder.name) or {}
        inputs = [_model_ref((item or {}).get('ref'), self.repo_root) for item in document.get('inputs') or []]
        drifted, reference, subtracted = (inputs + [None, None, None])[:3]
        edited = params.get('edited_layers')        # what edit_stats saw change, recorded by the repair itself
        checks = {'checkpoint_name_matches': circuit_checkpoint(family, strength, top_fraction, vector, seed).name
                  == folder.name,
                  'edited_layers_match_layer_subset': None if edited is None else edited == layer_subset}
        scores_path = _locate(params.get('scores'), self.repo_root)
        if scores_path is None and isinstance(params.get('scores'), str):
            fallback = self.run / 'circuits' / params['scores'].replace('\\', '/').rsplit('/', 1)[-1]
            scores_path = fallback if fallback.is_file() else None
        recomputed = None
        if scores_path is None:
            self.unverified.append({'where': where, 'detail': 'the score file named in the provenance is not here'})
        else:
            checks['scores_sha256_matches'] = file_sha256(scores_path) == params.get('scores_sha256')
            if checks['scores_sha256_matches']:
                values = self.scores(scores_path)
                recomputed = top_layers(values, top_fraction)
                checks['circuit_layers_recomputed'] = recomputed == recorded
                expected = recomputed if seed is None else control_layers(values, recomputed, seed)
                checks['edited_layers_recomputed'] = expected == layer_subset
        parameters = None
        names = tensor_names(folder)
        if names is None:
            self.unverified.append({'where': where, 'detail': 'no safetensors header to list the parameters from'})
        else:
            self.note(folder / ('model.safetensors.index.json' if (folder / 'model.safetensors.index.json').is_file()
                                else 'model.safetensors'), 'read',
                      'tensor names only, from the safetensors header; weights are released separately',
                      'repair_arms.json', digest=False)
            try:
                parameters = sorted(_circuit_keys(_TensorNames(names), layer_subset, modules), key=_by_layer)
            except ValueError:
                checks['parameters_match_mask'] = False
            else:
                recorded_counts = [params.get(key) for key in ('matched_parameters', 'edited_tensors')
                                   if params.get(key) is not None]
                checks['parameter_count_matches'] = (all(len(parameters) == count for count in recorded_counts)
                                                     if recorded_counts else None)
        for check, ok in checks.items():
            if ok is False:
                self.fail(check, where, 'the recomputed value disagrees with the provenance')
        runtime = params.get('runtime') if isinstance(params.get('runtime'), dict) else {}
        return {
            'checkpoint': folder.name, 'path': self.rel(folder), 'arm': arm_id,
            'tag': role.get('tag') or f'{family}-{arm_id}-{knob}' + ('' if seed is None else f'-rand{seed}'),
            'role': role.get('role') or ('random-layer control' if seed is not None else 'candidate'),
            'selected_for_quantization': folder.name == quantized,
            'method': _name(document.get('method')), 'vector': vector, 'rule': rule,
            'edit': f'{drifted} + {strength} x ({reference} - {subtracted})',
            'models': {'drifted': drifted, 'positive': reference, 'negative': subtracted},
            'strength': strength, 'top_fraction': top_fraction, 'random_seed': seed,
            'circuit_mask': f'{_stem(params.get("scores"))}-top{top_fraction}',
            'circuit_layers': recorded, 'edited_layers': layer_subset, 'target_modules': modules,
            'parameter_count': len(parameters) if parameters is not None else _number(params.get('matched_parameters')),
            'parameters': parameters,
            'edit_l2': _number(params.get('edit_l2')), 'relative_edit_l2': _number(params.get('relative_edit_l2')),
            'edited_tensors': _number(params.get('edited_tensors')),
            'scores_file': repo_relative(params.get('scores'), self.repo_root),
            'scores_sha256': _digest(params.get('scores_sha256')),
            'safetune_commit': _name(params.get('safetune_commit')), 'library_version': _name(document.get('version')),
            'runtime': {key: _name(value) for key, value in runtime.items() if _name(key)},
            'created_at': _timestamp(document.get('created_at')),
            'checks': checks}

    # -- the editing masks: one per (score file, top fraction) the arms used --

    def masks(self, arms: list[dict]) -> list[dict]:
        groups: dict[tuple, list[dict]] = {}
        for arm in arms:
            if arm.get('scores_sha256') and arm.get('top_fraction') is not None:
                groups.setdefault((arm['scores_file'], arm['scores_sha256'], arm['top_fraction']), []).append(arm)
        masks = []
        for (scores_file, digest, fraction), members in sorted(groups.items(), key=lambda item: str(item[0])):
            circuit = [arm for arm in members if arm.get('random_seed') is None] or members
            first = circuit[0]
            if any(arm['circuit_layers'] != first['circuit_layers'] or arm['target_modules'] != first['target_modules']
                   for arm in members):
                self.fail('mask', first['circuit_mask'], 'arms on one score file and fraction disagree on the mask')
            path = _locate(scores_file, self.repo_root)
            parameters = next((arm['parameters'] for arm in circuit if arm.get('parameters')), None)
            masks.append({'id': first['circuit_mask'], 'source': self.mask_source(path, scores_file, digest, fraction),
                          'top_fraction': fraction, 'layers': first['circuit_layers'],
                          'layer_count': len(first['circuit_layers'] or []), 'target_modules': first['target_modules'],
                          'parameter_count': len(parameters) if parameters else None, 'parameters': parameters,
                          'used_by': [arm['checkpoint'] for arm in circuit],
                          'random_layer_controls': [arm['checkpoint'] for arm in members
                                                    if arm.get('random_seed') is not None]})
        return masks

    def mask_source(self, path: Path | None, scores_file: str, digest: str, fraction: float) -> dict:
        source = {'scores_file': scores_file, 'scores_sha256': digest, 'public': False}
        if path is None or file_sha256(path) != digest:
            return source
        document = json.loads(path.read_text(encoding='utf-8'))
        values = self.scores(path)
        merged = (document.get('metadata') or {}).get('merged_from')
        self.note(path, 'withheld', self.withheld_reason(path.name), 'editing_mask.json')
        source.update(_pick(document, {'model': _name, 'algorithm': _name, 'level': _name}))
        if isinstance(merged, list):
            located = [_locate(item, self.repo_root) for item in merged]
            for found in filter(None, located):
                self.note(found, 'withheld', self.withheld_reason(found.name)
                          + '; its sha256, node counts and per-layer mass are published', 'editing_mask.json')
            source['combination'] = 'mean of the node scores of the circuits below (langsafe.circuits.discover.merge_scores)'
            source['merged_from'] = [{'circuit': self.circuit_of(found or Path(str(item).replace('\\', '/'))),
                                      'scores_file': repo_relative(item, self.repo_root),
                                      'scores_sha256': file_sha256(found) if found else None}
                                     for item, found in zip(merged, located)]
            source['mean_of_sources_verified'] = None
            if located and all(located):
                parts = [self.scores(found) for found in located]
                names = set().union(*parts)
                mean = {name: sum(part.get(name, 0.0) for part in parts) / len(parts) for name in names}
                source['mean_of_sources_verified'] = set(mean) == set(values) and all(
                    abs(mean[name] - values[name]) <= 1e-9 * max(1.0, abs(values[name])) for name in values)
                if not source['mean_of_sources_verified']:
                    self.fail('merged_scores', scores_file, 'the merged scores are not the mean of their sources')
            else:
                self.unverified.append({'where': scores_file, 'detail': 'a merged-from score file is not here'})
        source['nodes'] = node_counts(values, None)
        source['layers'] = layer_table(values, [fraction])
        return source

    @staticmethod
    def circuit_of(path: Path) -> str | None:
        for part in reversed(path.parts):
            if _CIRCUIT.fullmatch(part):
                return part
        return None

    # -- circuits: one file per (model, language) --

    def circuits(self, fractions: list[float], mask_sources: set[str]) -> list[dict]:
        records = []
        for folder in sorted(path for path in (self.run / 'circuits').glob('*__*') if path.is_dir()):
            match = _CIRCUIT.fullmatch(folder.name)
            if not match:
                continue
            summary_path = folder / 'summary.json'
            if not summary_path.is_file():
                summary_path = next(iter(sorted(folder.glob('*/summary.json'))), None)
            if summary_path is None:
                self.fail('circuit', folder.name, 'no summary.json')
                continue
            records.append(self.circuit(folder, summary_path, fractions, folder.name in mask_sources))
        return records

    def circuit(self, folder: Path, summary_path: Path, fractions: list[float], in_mask: bool) -> dict:
        name, target = folder.name, f'circuits/{folder.name}.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        self.note(summary_path, 'withheld', self.withheld_reason(summary_path.name) + '; its aggregates are published',
                  target)
        scores_path = _locate(summary.get('scores_json'), self.repo_root)
        if scores_path is None:
            found = sorted(folder.glob('*/*_node_scores.json'))
            scores_path = found[0] if len(found) == 1 else None
        if scores_path is None:
            self.fail('circuit', name, 'the node score file is not here')
            return {'circuit': name}
        values = self.scores(scores_path)
        self.note(scores_path, 'withheld', self.withheld_reason(scores_path.name)
                  + '; its sha256, node counts and per-layer mass are published', target)
        settings = summary.get('settings') or {}
        faithfulness = summary.get('faithfulness') or {}
        metadata = faithfulness.get('metadata') or {}
        baselines = faithfulness.get('baseline_comparison') or {}
        stability = faithfulness.get('stability') or {}
        random_baseline = (baselines.get('baselines') or {}).get('random') or {}
        magnitude = (baselines.get('baselines') or {}).get('magnitude') or {}
        record = {
            'circuit': name, 'model': _name(summary.get('model')), 'model_tag': _name(summary.get('model_tag')),
            'lang': _name(summary.get('lang')), 'used_for_editing_mask': in_mask,
            'algorithm': _name(metadata.get('algorithm')), 'level': _name(metadata.get('level')),
            'scope': _name(metadata.get('scope')), 'circuitkit_commit': _name(summary.get('circuitkit_commit')),
            'scores_file': {'path': self.rel(scores_path), 'sha256': file_sha256(scores_path), 'public': False},
            'task': self.task(summary, target),
            'settings': {**_pick(settings, {'precision': _name, 'n_examples': _number, 'ig_steps': _number,
                                            'sparsity': _number, 'seed': _number, 'data_max_records': _number,
                                            'n_stability_runs': _number}),
                         'pillars': _names(settings.get('pillars'))},
            'nodes': node_counts(values, settings.get('sparsity')),
            'layers': layer_table(values, fractions),
            'faithfulness': {
                'patching_score': _number(faithfulness.get('patching_score')),
                'patching_ratio_unclipped': _number(metadata.get('patching_raw_ratio')),
                'ablation_score': _number(faithfulness.get('ablation_score')),
                'ablation_ratio_unclipped': _number(metadata.get('ablation_raw_ratio')),
                'pillars_computed': _names(metadata.get('pillars_computed')),
                'duration_seconds': _number(metadata.get('total_duration_seconds')),
                'baselines': {**_pick(baselines, {'circuit_score': _number, 'best_baseline_score': _number,
                                                  'circuit_advantage': _number, 'circuit_sparsity': _number}),
                              'random': {**_pick(random_baseline, {'score': _number, 'score_std': _number,
                                                                   'n_draws': _number, 'z_score': _number}),
                                         'draw_scores': _numbers(random_baseline.get('random_scores'))},
                              'magnitude': _pick(magnitude, {'score': _number})},
                'stability': {**_pick(stability, {'n_runs': _number, 'mean_jaccard': _number, 'std_jaccard': _number,
                                                  'mean_dice': _number, 'std_dice': _number,
                                                  'mean_spearman': _number, 'std_spearman': _number,
                                                  'n_stable_nodes': _number}),
                              'jaccard_matrix': [_numbers(row) for row in stability.get('jaccard_matrix') or []
                                                 if _numbers(row) is not None][:MAX_NUMBERS],
                              'overlap_per_layer': {str(key): _number(value) for key, value
                                                    in (stability.get('overlap_per_layer') or {}).items()
                                                    if str(key).isdigit()}}},
            'refusal_signal': self.signal(summary, baselines, target),
            'fair_baseline': self.fair(folder, target),
            'parity': self.parity(summary.get('parity') or {})}
        return record

    def task(self, summary: dict, target: str) -> dict:
        task = {'name': _name(summary.get('task')), 'pairs_used': _number(summary.get('task_examples')),
                'file': None, 'file_sha256': None, 'pairs_in_file': None, 'preamble': None}
        try:
            path = fair_circuit_baseline.task_file(self.run, summary)
        except (FileNotFoundError, KeyError, IndexError, AttributeError, TypeError):
            self.unverified.append({'where': target, 'detail': 'no task file matches the hash in the task name'})
            return task
        with path.open(encoding='utf-8', newline='') as stream:
            pairs = sum(1 for _ in csv.DictReader(stream))      # counted, never read out
        preamble = _TASK_PREAMBLE.fullmatch(path.name)
        self.note(path, 'withheld', self.withheld_reason(path.name) + '; its row count and sha256 are published', target)
        task.update(file=self.rel(path), file_sha256=file_sha256(path), pairs_in_file=pairs,
                    preamble=preamble[1] if preamble else None)
        return task

    def signal(self, summary: dict, baselines: dict, target: str) -> dict | None:
        """The full model's and the harmless twin's mean logit difference, from the pillar-1 line of the circuit's log."""
        tag, lang = str(summary.get('model_tag', '')), str(summary.get('lang', ''))
        log = self.run / 'logs' / f'{tag.removesuffix("-circuit")}-circuit-{lang}.log'
        ends = fill_submission.clean_and_corrupt(log, _number(baselines.get('circuit_score')))
        if ends is None:
            return None
        self.note(log, 'withheld', 'free-text log; only the numbers on its pillar-1 line are read', target)
        harmful, harmless = ends
        best = _number(baselines.get('best_baseline_score'))
        return {'harmful_logit_difference': harmful, 'harmless_twin_logit_difference': harmless,
                'signal': harmful - harmless,
                'best_baseline_on_patching_scale': None if best is None else (best - harmless) / (harmful - harmless)}

    def fair(self, folder: Path, target: str) -> dict | None:
        path = folder / 'fair_baseline.json'
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding='utf-8-sig'))
        self.note(path, 'exported', 'copied field by field', target)
        return {**_pick(document, {'method': _text, 'kept_heads': _number, 'kept_mlps': _number,
                                   'seeds': _numbers, 'patching_ratios': _numbers, 'mean': _number,
                                   'circuit_patching_ratio': _number, 'check_ratio': _number,
                                   'n_examples': _number, 'circuitkit_commit': _name}),
                'hardware': _pick(document.get('hardware'), {'device': _name, 'gpu': _text, 'torch': _name,
                                                              'cuda': _name, 'tf32': _flag,
                                                              'float32_matmul_precision': _name})}

    @staticmethod
    def parity(parity: dict) -> dict:
        rows = [row for row in parity.get('per_prompt') or [] if isinstance(row, dict)]
        kl = [_number(row.get('kl')) for row in rows]
        deviation = [_number(row.get('max_logprob_deviation')) for row in rows]
        return {**_pick(parity, {'passed': _flag, 'dtype': _name, 'device': _name, 'kl_tolerance': _number,
                                 'max_logprob_tolerance': _number, 'circuitkit_commit': _name}),
                'sentences_checked': len(rows),
                'max_kl': max(kl) if rows and None not in kl else None,
                'max_logprob_deviation': max(deviation) if rows and None not in deviation else None,
                'argmax_agrees_on_all': all(row.get('argmax_agrees') is True for row in rows) if rows else None}

    # -- overlap.json --

    def overlap(self) -> dict | None:
        path = self.run / 'circuits' / 'overlap.json'
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding='utf-8-sig'))
        self.note(path, 'exported', 'copied field by field', 'overlap.json')

        def matrix(value) -> dict:
            return {row: {column: _number(cell) for column, cell in cells.items() if _MATRIX_KEY.fullmatch(column)}
                    for row, cells in (value or {}).items() if _MATRIX_KEY.fullmatch(row) and isinstance(cells, dict)}

        return {'schema': SCHEMA, 'k': _number(document.get('k')),
                'jaccard_rule': 'Jaccard overlap of the k highest-scoring nodes (signed score) of two circuits',
                'spearman_rule': 'Spearman rank correlation of all node scores of two circuits',
                'jaccard': matrix(document.get('jaccard')), 'spearman': matrix(document.get('spearman'))}

    # -- bookkeeping --

    @staticmethod
    def withheld_reason(name: str) -> str:
        return next((why for pattern, why in WITHHELD if pattern.fullmatch(name)), 'not on the allowlist')

    def sources(self) -> list[dict]:
        """Every file under circuits/ and circuit_tasks/, plus everything else that was read."""
        for folder in ('circuits', 'circuit_tasks'):
            for path in sorted((self.run / folder).rglob('*')) if (self.run / folder).is_dir() else []:
                if path.is_file() and not path.is_symlink() and self.rel(path) not in self.read:
                    self.note(path, 'withheld', self.withheld_reason(path.name))
        return [{'file': name, **entry} for name, entry in sorted(self.read.items())]

    def run_all(self) -> dict:
        arms = self.arms()
        fractions = sorted({arm['top_fraction'] for arm in arms if arm.get('top_fraction') is not None}) \
            or [DEFAULT_TOP_FRACTION]
        masks = self.masks(arms)
        in_mask = {item['circuit'] for mask in masks for item in mask['source'].get('merged_from') or []
                   if item.get('circuit')}
        in_mask |= {self.circuit_of(Path(mask['source']['scores_file'])) for mask in masks
                    if mask['source'].get('scores_file')} - {None}
        circuits = self.circuits(fractions, in_mask)
        overlap = self.overlap()
        provenance = sorted(name for name, entry in self.read.items() if name.endswith('lexsi_provenance.json')
                            or name.endswith('selection.json'))
        for record in circuits:
            self.write(f'circuits/{record["circuit"]}.json', {'schema': SCHEMA, **record},
                       [name for name, entry in self.read.items() if f'circuits/{record["circuit"]}.json'
                        in entry['into']])
        self.write('editing_mask.json', {'schema': SCHEMA, 'arithmetic': ARITHMETIC, 'masks': masks},
                   [name for name, entry in self.read.items() if 'editing_mask.json' in entry['into']] + provenance)
        self.write('repair_arms.json', {'schema': SCHEMA, 'arithmetic': ARITHMETIC,
                                        'vectors': {key: rule for key, (_, rule) in VECTORS.items()}, 'arms': arms},
                   [name for name, entry in self.read.items() if 'repair_arms.json' in entry['into']])
        if overlap:
            self.write('overlap.json', overlap,
                       [name for name, entry in self.read.items() if 'overlap.json' in entry['into']])
        self.write('README.md', readme(circuits, masks, arms, overlap is not None), [])
        return {'circuits': circuits, 'masks': masks, 'arms': arms}

    def manifest(self, run_name: str) -> dict:
        return {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
                'run': export_public.safe_label(run_name), 'exporter_sha256': file_sha256(Path(__file__)),
                'decision': 'D7: publish the layer lists, summary metrics and editing mask; keep node-level scores '
                            'private and share them with researchers on request',
                'policy': 'Built field by field in scripts/export_circuits.py. No source file is copied as it stands.',
                'exported': self.exported, 'sources': self.sources(),
                'summary_fields_left_out': [{'field': field, 'why': why}
                                            for field, why in SUMMARY_FIELDS_LEFT_OUT.items()],
                'unverified': self.unverified}


# ---- README -------------------------------------------------------------------------------------------------------

def _fmt(value, digits: int = 2) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def _layers(layers) -> str:
    return ', '.join(str(layer) for layer in layers) if layers else '-'


def readme(circuits: list[dict], masks: list[dict], arms: list[dict], has_overlap: bool) -> str:
    fraction = next((item['top_fraction'] for record in circuits for item in record.get('layers', {}).get('selected', [])),
                    DEFAULT_TOP_FRACTION)
    lines = [
        '# Refusal circuits: layer-level export',
        '',
        'Refusal circuits found with CircuitKIT (EAP-IG, node level) in Tiny Aya Global and Tiny Aya Earth, and the '
        'circuit-restricted repairs built from them with SafeTune. Written by `scripts/export_circuits.py` from '
        '`runs/experiments/full/earth`.',
        '',
        'Team decision D7: we publish the circuits\' layers, summary metrics and the editing mask, which are enough to '
        'reproduce every repair. The node-level score files stay private. Researchers can ask the authors for them, '
        'and the SHA-256 values here identify the exact files we used.',
        '',
        '## Files',
        '',
        '| File | Contents |',
        '|---|---|',
        '| `circuits/<model>-circuit__<lang>.json` | One circuit: settings, node counts, per-layer score mass and rank, '
        f'the top {fraction:.0%} of layers, faithfulness pillars, the fair random-head baseline, the parity check, '
        'and the SHA-256 of the private score file |',
        '| `editing_mask.json` | The mask the circuit repairs used: its score file and how it was built, its layers and '
        'modules, and the exact weight names |',
        '| `repair_arms.json` | Every circuit-restricted checkpoint: vector, strength, edited layers and weights, edit '
        'size, and the checks that tie it to the mask |',
    ]
    if has_overlap:
        lines.append('| `overlap.json` | Top-50 node Jaccard overlap and full-ranking Spearman correlation between '
                     'circuits |')
    lines += ['| `EXPORT_MANIFEST.json` | Every source file with its SHA-256 and what became of it, the fields left '
              'out, and the final scan |', '',
              '## Circuits', '',
              f'| Circuit | Pairs used | Patching ratio | Ablation ratio | Fair baseline, random heads (mean) '
              f'| Top {fraction:.0%} of layers | In the mask |',
              '|---|---:|---:|---:|---:|---|---|']
    for record in circuits:
        faithfulness = record.get('faithfulness') or {}
        selected = next(iter((record.get('layers') or {}).get('selected') or []), {})
        lines.append(f'| {record["circuit"]} | {(record.get("task") or {}).get("pairs_used") or "-"} '
                     f'| {_fmt(faithfulness.get("patching_ratio_unclipped"))} '
                     f'| {_fmt(faithfulness.get("ablation_ratio_unclipped"))} '
                     f'| {_fmt((record.get("fair_baseline") or {}).get("mean"))} '
                     f'| {_layers(selected.get("layers"))} | {"yes" if record.get("used_for_editing_mask") else "no"} |')
    lines += ['', 'Ratios are unclipped: 0 is no better than the harmless twin, 1 matches the full model.', '']
    for mask in masks:
        source = mask['source']
        merged = [item.get('circuit') or '?' for item in source.get('merged_from') or []]
        made = (f'the mean of the node scores of {", ".join(merged)}' if merged
                else f'the score file `{source.get("scores_file")}`')
        lines += [f'## Editing mask `{mask["id"]}`', '',
                  f'Built from {made}, by keeping the top {mask["top_fraction"]:.0%} of layers by total absolute node '
                  f'score: layers {_layers(mask["layers"])} ({mask["layer_count"]} of '
                  f'{(source.get("layers") or {}).get("layers_in_model", "?")}). Modules: '
                  f'{", ".join(mask["target_modules"] or [])}. That is {mask.get("parameter_count") or "?"} weight '
                  'matrices, named in `editing_mask.json`. Score file SHA-256: '
                  f'`{source.get("scores_sha256")}`.', '']
    lines += ['## Repairs', '',
              '| Checkpoint | Arm | Role | Strength | Edited layers | Relative edit size |', '|---|---|---|---:|---|---:|']
    for arm in arms:
        lines.append(f'| {arm["checkpoint"]} | {arm.get("arm", "-")} | {arm.get("role", "-")} '
                     f'| {arm.get("strength", "-")} | {_layers(arm.get("edited_layers"))} '
                     f'| {_fmt(arm.get("relative_edit_l2"), 4)} |')
    lines += ['',
              'To rebuild a repair without the score files, take its weight names from `repair_arms.json` and set each '
              'one to W_drifted + strength x (W_positive - W_negative), using the three models it names (SafeTune '
              '`apply_ctheta`, in float16). Every other weight stays as in the drifted model. The random-layer '
              'controls apply the same edit to as many randomly drawn layers outside the circuit; their weights are '
              'not released.',
              '',
              '## Not included',
              '',
              '- Node-level scores: the `*_node_scores.json` and `*_node_scores.pt` files, `merged-scores.json`, and the '
              'per-node scores and node sets of the stability pillar. Shared with researchers on request.',
              '- CircuitKIT\'s pruning artifacts (`*_node.pt`), which are node-level masks.',
              '- The circuit task tables: minimal pairs of harmful and harmless requests, and the opener tokens taken '
              'from the reference model\'s answers. Only their row counts and SHA-256 are here.',
              '- Logs. Only the pillar-1 numbers are read from them.',
              '- Model weights, which are released separately.',
              '',
              '## Checks',
              '',
              'For every checkpoint the exporter checks the score file against the SHA-256 in its provenance, recomputes '
              'the circuit layers with the repair code, repeats the random-layer draw, and lists the edited weights from '
              'the checkpoint\'s own tensor names. The results are under `checks` in `repair_arms.json`. Any mismatch '
              'blocks the export. The output was then scanned against every private model answer of the run.', '']
    return '\n'.join(lines)


# ---- final checks -------------------------------------------------------------------------------------------------

def _json_problems(value, where: str = 'root'):
    if isinstance(value, dict):
        if sum(_number(item) is not None for item in value.values()) > MAX_NUMBERS:
            yield where, 'long_numeric_table'
        for key, item in value.items():
            if NODE_NAME.fullmatch(str(key)):
                yield where, 'node_name_key'
            yield from _json_problems(item, f'{where}.{export_public.safe_label(str(key))}')
    elif isinstance(value, list):
        if sum(_number(item) is not None for item in value) > MAX_NUMBERS:
            yield where, 'long_numeric_list'
        for index, item in enumerate(value):
            yield from _json_problems(item, f'{where}[{index}]')
    elif isinstance(value, str) and NODE_NAME.fullmatch(value):
        yield where, 'node_name_value'


def guard(out_dir: Path, repo_root: Path) -> list[dict]:
    """What export_public's scan does not look for: node names, long numeric arrays, this machine's paths."""
    roots = {str(repo_root).casefold(), str(repo_root).replace('\\', '/').casefold()}
    violations = []
    for path in sorted(item for item in out_dir.rglob('*') if item.is_file()):
        relative = path.relative_to(out_dir).as_posix()
        text = path.read_text(encoding='utf-8')
        if publish_hf.LOCAL_PATH.search(text) or any(root in text.casefold() for root in roots):
            violations.append({'check': 'local_path', 'file': relative})
        if NODE_IN_TEXT.search(text):
            violations.append({'check': 'node_name', 'file': relative})
        if path.suffix == '.json':
            violations += [{'check': check, 'file': relative, 'where': where}
                           for where, check in _json_problems(json.loads(text))]
    return violations


def _report_failure(out: Path, problems: list[dict]) -> None:
    details = out.with_name(out.name + '.scan-failed.json')
    details.write_text(json.dumps({'status': 'failed', 'problem_count': len(problems), 'problems': problems},
                                  ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'EXPORT BLOCKED: {len(problems)} problem(s). Nothing was published and the staging folder was removed.',
          file=sys.stderr)
    for problem in problems[:20]:
        print('  ' + ' '.join(f'{key}={value}' for key, value in problem.items()), file=sys.stderr)
    print(f'Locations only, no text: {details}', file=sys.stderr)


def _check_request(run: Path, out: Path, replace: bool) -> None:
    if not run.is_dir():
        raise ExportError('RUN_DIR is not a folder')
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ExportError('OUT_DIR must be outside RUN_DIR, because the source run is never modified')
    if out.exists():
        manifest = out / MANIFEST_NAME
        ours = manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')).get('schema') == SCHEMA
        if not replace:
            raise ExportError('OUT_DIR already exists; pass --replace to rebuild an earlier circuit export')
        if not ours:
            raise ExportError('OUT_DIR exists and is not an earlier circuit export, so it is not replaced')


def export_circuits(run_dir: Path, out_dir: Path, *, repo_root: Path = ROOT, gens_dir: Path | None = None,
                    replace: bool = False) -> int:
    """Export `run_dir`'s circuits to `out_dir`. OUT_DIR appears only after every check passes. Returns the exit code."""
    run, out, repo_root = Path(run_dir).resolve(), Path(out_dir).resolve(), Path(repo_root).resolve()
    _check_request(run, out, replace)
    private = export_public.private_generation_files(run, Path(gens_dir).resolve() if gens_dir else None)
    if not private:
        raise ExportError('no private generations found (gens/*.jsonl), so the output cannot be proved clean; '
                          'pass --gens-dir if they live elsewhere')
    if not (run / 'circuits').is_dir():
        raise ExportError('RUN_DIR has no circuits folder')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f'.{out.name}.staging-{os.getpid()}')
    staging.mkdir()
    try:
        job = _Export(run, staging, repo_root)
        job.run_all()
        manifest = job.manifest(run.name)
        draft = staging / MANIFEST_NAME
        draft.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report = export_public.scan_output(staging, private)
        problems = [*job.failures, *report.violations, *guard(staging, repo_root)]
        if problems:
            _report_failure(out, problems)
            return 1
        # The draft manifest was scanned above; this adds only the scan's own counts.
        draft.write_text(json.dumps({**manifest, 'scan': {**report.summary(), 'node_and_path_guard': 'passed'}},
                                    ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if out.exists():
            shutil.rmtree(out)
        os.replace(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    out.with_name(out.name + '.scan-failed.json').unlink(missing_ok=True)
    print(f'Exported {len(job.exported) + 1} files from {len(job.read)} source files; '
          f'{len(job.unverified)} item(s) could not be verified (listed in the manifest).')
    print(f'Scan passed: {report.rows_compared} private rows from {len(private)} files. See {out / MANIFEST_NAME}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Write the public, layer-level copy of a run\'s refusal circuits.')
    parser.add_argument('run_dir', type=Path, nargs='?', default=RUN, help='run folder (only read)')
    parser.add_argument('out_dir', type=Path, nargs='?', default=OUT, help='folder to create, outside the run folder')
    parser.add_argument('--gens-dir', type=Path, help='extra folder of private generation JSONL files to scan against')
    parser.add_argument('--replace', action='store_true', help='rebuild an earlier circuit export in OUT_DIR')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):      # a Windows console may not encode every character in a path
            stream.reconfigure(errors='backslashreplace')
    try:
        return export_circuits(args.run_dir, args.out_dir, gens_dir=args.gens_dir, replace=args.replace)
    except ExportError as exc:
        print(f'EXPORT REFUSED: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        # Only the type: an exception's message can quote the data that caused it.
        print(f'EXPORT FAILED: {type(exc).__name__}. Nothing was published.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
