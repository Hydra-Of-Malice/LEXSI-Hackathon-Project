#!/usr/bin/env python3
"""Write the Hugging Face cards for the public release: one model card per published arm, a card for the selected arm's
GGUF, and the dataset card. They go to runs/experiments/public/cards/, where scripts/publish_hf.py picks them up.

    python scripts/release_cards.py --repo-url https://github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project --code-commit <commit> \
        --models-collection <url> --data-collection <url>

The cards start from the same templates and the same run files as the article (scripts/fill_submission.py), filled for
each arm in turn. Release details (URLs, repo ids, checkpoint hashes) are added here. Every card must come out with no
placeholder left, or nothing is written.

The template's acceptance section assumes the target language has a confirmed gap. When it has none (our Thai target),
the card says the arm is exploratory, as the article does.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
OUT = ROOT / 'runs' / 'experiments' / 'public' / 'cards'
TEMPLATES = ROOT / 'docs' / 'submission'
COMMENT = re.compile(r'\n<!--.*?-->\n', re.S)
NOT_RUN = 'not run: only the selected arm was quantised'
PRECISION_KEYS = ('Q4_E_SAFE', 'Q4_R_SAFE', 'Q4_OVER', 'Q4_CAP', 'NF4_E_SAFE', 'NF4_R_SAFE', 'NF4_OVER', 'NF4_CAP',
                  'Q4_RESULT', 'NF4_RESULT', 'PASS_Q4', 'PASS_NF4')


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fill_submission = load('fill_submission')
publish_hf = load('publish_hf')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def per_language(summary: dict, key: str) -> str:
    cells = summary.get(key) or {}
    return ', '.join(f'{lang} {value:.2f}' for lang, value in cells.items() if value is not None) or 'not measured'


def limitations(report: dict, tag: str, values: dict) -> str:
    """Where this arm missed the pre-set conditions, from the report's gates."""
    gate = (report.get('thresholds') or {}).get(f'{tag}/default') or {}
    missed = []
    closed = (gate.get('harmful_gap_closed') or {}).get('value')
    if closed is not None and not (gate.get('harmful_gap_closed') or {}).get('pass'):
        missed.append(f'it closed {100 * closed:.0f}% of the {values.get("TARGET_LANGUAGE", "target")} gap '
                      'against the 50% required')
    if not (gate.get('over_refusal_increase') or {}).get('pass') and values.get('MAX_OVER_REFUSAL_RISE'):
        missed.append(f'its largest benign over-refusal rise was {values["MAX_OVER_REFUSAL_RISE"]} points against a cap of 2')
    if not (gate.get('capability_drop') or {}).get('pass') and values.get('MAX_CAPABILITY_DROP'):
        missed.append(f'its largest capability drop was {values["MAX_CAPABILITY_DROP"]} points against a cap of 1')
    if not missed:
        return ''
    return ('- **This arm missed the pre-set conditions, with the preamble:** ' + '; '.join(missed) + '. The thresholds '
            'sit close to the noise of these measures, and the random-layer controls crossed them too.')


def card(template: str, values: dict, blocks: dict) -> str:
    text = COMMENT.sub('\n', template, count=1)
    filled, left = fill_submission.fill(text, values, blocks)
    if left:
        raise SystemExit(f'{values.get("MODEL_NAME") or values.get("DATASET_NAME")}: placeholders left: {sorted(left)}')
    return filled


def exploratory(text: str, values: dict) -> str:
    """The template's acceptance wording assumes a confirmed target gap. Say what is true when there is none."""
    sentence = values.get('TARGET_CONFIRMATION_SENTENCE', '')
    old = (f"These results apply to {values['TARGET_CONDITION']}, which has a confirmed gap for the target language. "
           'In a preamble condition without one, this checkpoint is exploratory.')
    new = (f"{values['TARGET_LANGUAGE']}, the target language, has no confirmed gap in either preamble condition, so this "
           'checkpoint is exploratory: it could not pass acceptance, and we make no repair claim.')
    if old in text:
        text = text.replace(old, new)
    why = f"for Earth against {values['TARGET_SAFE_GLOBAL']}% for Global."
    if sentence and why in text and sentence not in text:
        text = text.replace(why, f'{why} {sentence}', 1)
    return text


def steering(text: str, values: dict) -> str:
    text = text.replace('base_model_relation: merge\n', '').replace('library_name: transformers\n', '')
    text = text.replace('`lexsi_provenance.json` in this repository', '`meta.json` in this repository')
    text = re.sub(r'A research checkpoint derived from Tiny Aya Earth by weight arithmetic, built to test per-language '
                  r'safety repair\. Arm: \*\*Steering\*\*, knob strength = ([0-9.]+), precision FP16\.',
                  r'A refusal direction for Tiny Aya Earth, added at run time, built to test per-language safety repair. '
                  r'Arm: **Steering**, knob strength = \1. No weights are changed.', text, count=1)
    start, end = text.index('## Use the model'), text.index('## Reproducibility')
    use = ('## Use the direction\n\n'
           'This repository holds a refusal direction, not weights: `refusal_direction.pt` (the direction and its layer, 24) '
           'and `meta.json`. It is added to the residual stream of '
           '[CohereLabs/tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) at layer 24 with strength 1.0, '
           'at generation time, through SafeTune\'s `RefusalDirectionModel` in steer mode (commit `3901a9f`). '
           f"`langsafe.repair.arms.steered` in {values['REPO_URL']} wraps it:\n\n"
           '```python\nfrom pathlib import Path\nfrom huggingface_hub import snapshot_download\n'
           'from langsafe.gen.model import load\nfrom langsafe.repair.arms import steered\n\n'
           f'folder = Path(snapshot_download("{values["MODEL_REPO_ID"]}"))\n'
           'model, tokenizer = load("CohereLabs/tiny-aya-earth")\n'
           'with steered(model, folder, strength=1.0):\n'
           '    ...  # model.generate as usual\n```\n\n')
    return text[:start] + use + text[end:]


def gguf(text: str, values: dict, filename: str, full_id: str) -> str:
    text = text.replace('library_name: transformers', 'library_name: gguf').replace('base_model_relation: merge',
                                                                                   'base_model_relation: quantized')
    text = text.replace('base_model: CohereLabs/tiny-aya-earth\n', f'base_model: {full_id}\n', 1)
    text = text.replace(', precision FP16.', ', precision Q4_K_M (GGUF), quantised from FP16.', 1)
    text = text.replace('`lexsi_provenance.json` in this repository',
                        f'`lexsi_provenance.json` in [{full_id}](https://huggingface.co/{full_id})')
    text = text.replace(f"# {values['MODEL_NAME']}\n", f"# {values['MODEL_NAME']}\n\n> This repository holds `{filename}`, "
                        'the Q4_K_M GGUF of the full-precision checkpoint described below. The per-language table is the '
                        'full-precision one; the Q4_K_M results are in "After 4-bit quantisation".\n', 1)
    start, end = text.index('## Use the model'), text.index('## Reproducibility')
    use = ('## Use the model\n\n'
           f'Run it with llama.cpp: `llama-server -m {filename} -c 4096`. The file keeps the generation end token (3). '
           'Access to the Tiny Aya tokenizer and weights follows the same licence and policy.\n\n')
    return text[:start] + use + text[end:]


def build(args) -> list[Path]:
    run = args.run
    report, selection = read_json(run / 'report.json'), read_json(run / 'selection.json')
    blocks = fill_submission.kept_blocks(report, selection)
    written = fill_submission.written(report)
    namespace = args.namespace
    dataset_id = f'{namespace}/{publish_hf.PREFIX}-eval'
    release = {'ARTICLE_URL': args.article_url, 'COLLECTION_MODELS_URL': args.models_collection,
               'COLLECTION_DATA_URL': args.data_collection, 'CONTACT': f'Issues on {args.repo_url}/issues',
               'REPO_URL': args.repo_url, 'CODE_COMMIT': args.code_commit, 'DATASET_REPO_ID': dataset_id,
               'DATASET_NAME': f'{publish_hf.PREFIX}-eval',
               'RESULTS_URL': f'https://huggingface.co/datasets/{dataset_id}/tree/main/reports',
               'HAND_LABELLED_ITEMS': 'none yet: native readers have not labelled the sample',
               'JUDGE_ACCURACY': 'not yet measured, for want of hand labels'}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    template = (TEMPLATES / 'model_card_template.md').read_text(encoding='utf-8')
    selected_tag = selection['selected_for_quantization']['tag']
    for arm, entry in selection['arms'].items():
        chosen = entry.get('selected')
        if not chosen or publish_hf.release_rule(report, chosen['tag']):
            continue
        tag = chosen['tag']
        override = copy.deepcopy(selection)
        override['selected_for_quantization'] = chosen
        values = {**written, **fill_submission.measured(run, override)}
        name = f'{publish_hf.PREFIX}-{arm.replace("_", "-")}-{str(chosen["knob"]).replace(".", "p")}'
        if chosen.get('steering'):
            weights = run / 'checkpoints' / publish_hf.checkpoint_folder(chosen['steering']) / 'refusal_direction.pt'
            source = 'not used'
        else:
            folder = run / 'checkpoints' / publish_hf.checkpoint_folder(chosen['model'])
            weights = folder / 'model.safetensors'
            params = read_json(folder / 'lexsi_provenance.json').get('params', {})
            source = (f"Global's circuit scores, SHA-256 `{params['scores_sha256']}`, top {params['top_fraction']:.0%} "
                      f"of layers" if params.get('scores_sha256') else 'not used')
        summary = read_json(run / 'judged' / f'{tag}.local.summary.json')
        low = [lang for lang, value in (summary.get('safety_ac1_by_lang') or {}).items() if value is not None and value < 0.6]
        values.update(release, MODEL_NAME=name, MODEL_REPO_ID=f'{namespace}/{name}', CHECKPOINT_SHA256=sha256(weights),
                      CIRCUIT_SOURCE=source, KAPPA_BY_LANGUAGE=per_language(summary, 'kappa_by_lang'),
                      AC1_BY_LANGUAGE=per_language(summary, 'safety_ac1_by_lang'),   # this arm's, not Earth's
                      LANGUAGE_REPORT_NOTE=(f"Languages below the AC1 floor of 0.6, reported as not measurable: "
                                            f"{', '.join(low)}." if low else 'Every language cleared the AC1 floor of 0.6.'),
                      MODEL_SPECIFIC_LIMITATIONS=limitations(report, tag, values))
        if chosen.get('steering'):
            # Steering edits no weight: it adds a direction at run time.
            values.update(DONOR_MODELS="none: the direction comes from Earth's own activations",
                          SAFETUNE_FUNCTION='RefusalDirectionModel, steer mode', EDITED_LAYERS='none: added at layer 24 at run time',
                          EDITED_TENSORS='none', RELATIVE_EDIT_L2='not applicable')
        if tag != selected_tag:
            # Only the selected arm was quantised; Earth's own 4-bit rates stay, this arm's cells say why they are empty.
            for key in PRECISION_KEYS:
                if key in ('Q4_E_SAFE', 'NF4_E_SAFE') and key in values:
                    continue
                values[key] = NOT_RUN if key.endswith('_RESULT') else 'n/a' if key.startswith('PASS_') else 'not run'
        text = exploratory(card(template, values, blocks), values)
        if chosen.get('steering'):
            text = steering(text, values)
        path = out / f'model_card-{tag}.md'
        path.write_text(text, encoding='utf-8', newline='\n')
        paths.append(path)
        if tag == selected_tag:
            gguf_file = run / 'quantized' / f'{tag}-q4_k_m.gguf'
            gguf_values = {**values, 'MODEL_NAME': f'{name}-gguf', 'MODEL_REPO_ID': f'{namespace}/{name}-gguf',
                           'CHECKPOINT_SHA256': sha256(gguf_file)}
            gguf_text = gguf(exploratory(card(template, gguf_values, blocks), gguf_values), gguf_values, gguf_file.name,
                             f'{namespace}/{name}')
            path = out / f'model_card-{tag}-gguf.md'
            path.write_text(gguf_text, encoding='utf-8', newline='\n')
            paths.append(path)
    manifest = read_json(run / 'data' / 'manifest.json')
    shortfalls = (manifest.get('coverage') or {}).get('shortfalls') or {}
    harmful = [n for key, n in shortfalls.items() if key.endswith('/harmful')]
    benign = [n for key, n in shortfalls.items() if key.endswith('/benign')]
    failures = {}
    for row in manifest.get('translation_failures') or []:
        lang = str(row).split('::')[-1]
        failures[lang] = failures.get(lang, 0) + 1
    names = {'ar': 'Arabic', 'bn': 'Bengali', 'jv': 'Javanese', 'ko': 'Korean', 'sw': 'Swahili', 'th': 'Thai'}
    data_values = {**written, **fill_submission.measured(run), **release,
                   'SPLIT_COUNT_TABLE_NOTE': (f'Every language falls a little short of the caps, because rows that failed '
                                              f'translation or the checks were dropped: {min(harmful)} to {max(harmful)} '
                                              f'harmful and {min(benign)} to {max(benign)} benign prompts per language.'
                                              if harmful and benign else ''),
                   'TRANSLATION_FAILED_REASONS': ('by language, ' + ', '.join(
                       f'{names.get(lang, lang)} {count}' for lang, count in sorted(failures.items(), key=lambda kv: -kv[1]))
                       + '. The manifest records which rows failed, not why'),
                   'RELEASE_LICENCE_DECISION': 'We release this compilation under CC-BY-NC 4.0, as set out below.'}
    text = card((TEMPLATES / 'dataset_card_template.md').read_text(encoding='utf-8'), data_values, blocks)
    path = out / 'dataset_card.md'
    path.write_text(text, encoding='utf-8', newline='\n')
    paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--namespace', default='HydraOfMalice')
    parser.add_argument('--repo-url', required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--article-url', required=True)
    parser.add_argument('--models-collection', required=True)
    parser.add_argument('--data-collection', required=True)
    args = parser.parse_args(argv)
    for path in build(args):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
