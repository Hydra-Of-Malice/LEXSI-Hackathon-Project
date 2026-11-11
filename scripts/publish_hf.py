#!/usr/bin/env python3
"""Plan the public Hugging Face release, and carry it out only on an explicit go-ahead.

    python scripts/publish_hf.py                          # print the plan: repos, files, sizes, what stays private
    python scripts/publish_hf.py --execute <plan digest>  # create the public repos and upload exactly that plan

Publishing is the team's decision (docs/submission/decisions.md, D3 and D4), so nothing is uploaded without --execute,
and --execute needs the digest the plan printed. Whatever runs is the plan the team reviewed.

What the plan holds:
- **Dataset.** The sanitised export (scripts/export_public.py), only if its final scan passed.
- **Models.** Every selected repair arm that meets the release rule. The rule: no language is significantly less safe
  than Earth, by paired 95% interval of the content-safe rate, in either preamble condition. Each arm's repo gets only
  the files a checkpoint needs (config, weights, tokenizer, chat template, provenance). The steering arm publishes its
  direction instead of weights.
- **GGUF.** The selected arm's Q4_K_M file.
- **Kept private.** The random-layer controls, plus any arm that fails the rule.

Each repo needs its README card in --cards (the filled templates). A missing card blocks --execute. The plan also
refuses any path from a private folder (model answers, judgements, audit records), whatever the allowlist says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
EXPORT = ROOT / 'runs' / 'experiments' / 'public' / 'earth-full'
CARDS = ROOT / 'runs' / 'experiments' / 'public' / 'cards'
STAGE = ROOT / 'runs' / 'experiments' / 'public' / 'models'   # cleaned copies of small files
PREFIX = 'tiny-aya-earth-langsafe'   # LSAL section 2: no kit names in repo names
CHECKPOINT_FILES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'chat_template.jinja', 'special_tokens_map.json', 'lexsi_provenance.json', 'model.safetensors',
                    'model.safetensors.index.json'}
STEERING_FILES = {'refusal_direction.pt', 'meta.json'}
EXPORT_FOLDERS = {'data', 'provenance', 'reports', 'verdicts'}
PRIVATE_PARTS = {'gens', 'generations', 'judged', 'auditkit', 'hand_check', 'calibration', 'audit', 'review',
                 'llm_cache.sqlite', 'remote-nf4-bundle', 'remote-fair-bundle', 'earth-router-sample'}


@dataclass
class Repo:
    repo_id: str
    repo_type: str
    files: dict[str, Path]          # path in the repo -> local file
    card: Path
    note: str = ''
    problems: list[str] = field(default_factory=list)

    def size(self) -> int:
        return sum(path.stat().st_size for path in self.files.values() if path.is_file())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def is_private(path: Path) -> bool:
    return any(part in PRIVATE_PARTS for part in path.parts)


def release_rule(report: dict, tag: str) -> list[str]:
    """Languages where the arm is significantly less safe than Earth (paired 95% CI wholly below zero)."""
    worse = []
    for condition in ('default', 'none'):
        safety = (report['paired_comparisons'].get(f'{tag}/{condition}', {}).get('vs_drifted') or {}).get('safety') or {}
        for lang, cell in sorted(safety.items()):
            interval = (cell.get('content_safe') or {}).get('ci')
            if interval and interval[1] < 0:
                worse.append(f'{lang} {condition} ({100 * cell["content_safe"]["delta"]:+.1f} points)')
    return worse


def checkpoint_folder(model: str) -> str:
    return model.replace('\\', '/').rstrip('/').split('/')[-1]


LOCAL_PATH = re.compile(r'lexsi hackathon|LEXSI-Hackathon-Project|\b[A-Za-z]:[\\/]|/Users/|/home/', re.I)


def relative(value):
    """Turn an absolute path inside the project into a project-relative one, recursively through JSON values."""
    if isinstance(value, dict):
        return {key: relative(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative(item) for item in value]
    if isinstance(value, str) and re.match(r'^(?:[A-Za-z]:[\\/]|/)', value):
        parts = value.replace('\\', '/').split('/')
        anchor = ROOT.name
        if anchor in parts:
            return '/'.join(parts[parts.index(anchor) + 1:])
    return value


def sanitised_provenance(source: Path, stage: Path) -> Path:
    """A copy of lexsi_provenance.json with the laptop's absolute paths made relative; hashes are kept as they are."""
    stage.mkdir(parents=True, exist_ok=True)
    target = stage / source.name
    target.write_text(json.dumps(relative(read_json(source)), indent=2) + '\n', encoding='utf-8', newline='\n')
    return target


def plan(namespace: str, run: Path = RUN, export: Path = EXPORT, cards: Path = CARDS,
         stage: Path = STAGE) -> tuple[list[Repo], list[str]]:
    report, selection = read_json(run / 'report.json'), read_json(run / 'selection.json')
    repos, private = [], []
    manifest = export / 'EXPORT_MANIFEST.json'
    dataset = Repo(f'{namespace}/{PREFIX}-eval', 'dataset',
                   {path.relative_to(export).as_posix(): path for path in sorted(export.rglob('*')) if path.is_file()},
                   cards / 'dataset_card.md', 'sanitised export: prompts, labels, reports, provenance; no model text')
    if not manifest.is_file() or (read_json(manifest).get('scan') or {}).get('result') != 'passed':
        dataset.problems.append('the export has no passed final scan; rerun scripts/export_public.py')
    # The export's own allowlist decides what is inside its folders (its reports/judged holds aggregate summaries only).
    # Anything outside the folders it writes did not come from it.
    dataset.problems += [f'not part of the export: {name}' for name in dataset.files
                         if name != 'EXPORT_MANIFEST.json' and name.split('/')[0] not in EXPORT_FOLDERS]
    repos.append(dataset)
    selected = selection['selected_for_quantization']['tag']
    for arm, entry in selection['arms'].items():
        chosen = entry.get('selected')
        if not chosen:
            continue
        tag, knob = chosen['tag'], str(chosen['knob']).replace('.', 'p')
        worse = release_rule(report, tag)
        if worse:
            private.append(f'{tag}: fails the release rule in {", ".join(worse)}')
            continue
        name = f'{namespace}/{PREFIX}-{arm.replace("_", "-")}-{knob}'
        if chosen.get('steering'):
            folder = Path(chosen['steering'])
            folder = folder if folder.is_dir() else run / 'checkpoints' / checkpoint_folder(chosen['steering'])
            allowed = STEERING_FILES
        else:
            folder = run / 'checkpoints' / checkpoint_folder(chosen['model'])
            allowed = CHECKPOINT_FILES
        files, problems = {}, []
        for path in sorted(folder.rglob('*')) if folder.is_dir() else []:
            if not path.is_file():
                continue
            name_in_repo = path.relative_to(folder).as_posix()
            if name_in_repo == 'lexsi_provenance.json':
                path = sanitised_provenance(path, stage / tag)
            if name_in_repo in allowed:
                files[name_in_repo] = path
            else:
                problems.append(f'unexpected file left out: {name_in_repo}')
        if not files:
            problems.append(f'no files found in {folder}')
        repo = Repo(name, 'model', files, cards / f'model_card-{tag}.md',
                    'steering direction, applied at run time' if chosen.get('steering') else 'merged weights, FP16',
                    problems)
        repos.append(repo)
        if tag == selected:
            gguf = run / 'quantized' / f'{tag}-q4_k_m.gguf'
            repos.append(Repo(f'{name}-gguf', 'model', {gguf.name: gguf}, cards / f'model_card-{tag}-gguf.md',
                              'Q4_K_M GGUF of the selected arm', [] if gguf.is_file() else [f'missing {gguf}']))
    for control in selection.get('controls') or []:
        private.append(f"{control['tag']}: random-layer control, kept private by design")
    for repo in repos:
        checked = [repo.card] if repo.repo_type == 'dataset' else [*repo.files.values(), repo.card]
        for path in checked:
            if is_private(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                repo.problems.append(f'refused: {path} is under a private folder')
        # Small text files must not carry the laptop's paths (the export scans its own files for this already).
        for name, path in repo.files.items():
            if repo.repo_type == 'model' and path.suffix in ('.json', '.jinja', '.txt', '.md') and path.is_file() \
                    and path.stat().st_size < 50_000_000 and name != 'tokenizer.json' \
                    and LOCAL_PATH.search(path.read_text(encoding='utf-8', errors='replace')):
                repo.problems.append(f'refused: {name} holds a local path')
        if repo.card.is_file() and LOCAL_PATH.search(repo.card.read_text(encoding='utf-8', errors='replace')):
            repo.problems.append('refused: the card holds a local path')
        if not repo.card.is_file():
            repo.problems.append(f'card missing: {repo.card}')
    return repos, private


def digest(repos: list[Repo]) -> str:
    shape = [[repo.repo_id, repo.repo_type, sorted((name, path.stat().st_size if path.is_file() else -1)
                                                   for name, path in repo.files.items())] for repo in repos]
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]


def describe(repos: list[Repo], private: list[str]) -> str:
    lines = []
    for repo in repos:
        lines.append(f'{repo.repo_type:7s} {repo.repo_id}  ({len(repo.files)} files, {repo.size() / 1e9:.2f} GB) - {repo.note}')
        lines += [f'          ! {problem}' for problem in repo.problems]
    lines.append(f'total upload {sum(repo.size() for repo in repos) / 1e9:.1f} GB')
    lines += [f'private  {item}' for item in private]
    lines.append(f'plan digest {digest(repos)}')
    return '\n'.join(lines)


def execute(repos: list[Repo], expected: str) -> None:
    from huggingface_hub import HfApi

    if digest(repos) != expected:
        raise SystemExit('The plan changed since it was reviewed; print it again and pass the new digest')
    blocking = [f'{repo.repo_id}: {problem}' for repo in repos for problem in repo.problems
                if not problem.startswith('unexpected file left out')]
    if blocking:
        raise SystemExit('Not publishing:\n' + '\n'.join(blocking))
    from huggingface_hub import CommitOperationAdd

    api = HfApi()
    for repo in repos:
        with_retries(lambda: api.create_repo(repo.repo_id, repo_type=repo.repo_type, private=False, exist_ok=True))
        wanted = [*repo.files.items(), ('README.md', repo.card)]   # the card goes last, once the files are there
        done = with_retries(lambda: unchanged(api, repo.repo_id, repo.repo_type, dict(wanted)))
        groups = batches([(name, path) for name, path in wanted if name not in done])
        # Few commits per repo: the Hub allows 128 commits an hour (a file-per-commit upload hit it), and small files
        # travel inside the commit request itself, so one 93 MB commit of the whole dataset was cut off.
        for number, group in enumerate(groups, 1):
            operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in group]
            message = f'Publish {repo.repo_id.split("/")[-1]}' + (f' ({number}/{len(groups)})' if len(groups) > 1 else '')
            with_retries(lambda: api.create_commit(repo.repo_id, repo_type=repo.repo_type, operations=operations,
                                                   commit_message=message))
        print(f'published https://huggingface.co/{"datasets/" if repo.repo_type == "dataset" else ""}{repo.repo_id}'
              f'  ({len(wanted) - len(done)} sent in {len(groups)} commits, {len(done)} already there)', flush=True)


INLINE_LIMIT = 10 * 2**20      # the Hub sends smaller files inside the commit request rather than as LFS/Xet objects


def batches(files: list[tuple[str, Path]], limit: int = 20 * 2**20) -> list[list[tuple[str, Path]]]:
    """Split files into commits whose inline payload stays under `limit`; large files ride in the first commit."""
    groups, current, size = [], [], 0
    for name, path in sorted(files, key=lambda item: item[1].stat().st_size < INLINE_LIMIT):   # stable: large first
        inline = path.stat().st_size if path.stat().st_size < INLINE_LIMIT else 0
        if size and size + inline > limit:
            groups.append(current)
            current, size = [], 0
        current.append((name, path))
        size += inline
    return groups + [current] if current else groups


def blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**24), b''):
            sha.update(chunk)
    return sha.hexdigest()


def unchanged(api, repo_id: str, repo_type: str, files: dict[str, Path]) -> set[str]:
    """Paths already on the Hub with the same bytes, so a re-run after a failure does not send them again."""
    from huggingface_hub.hf_api import RepoFile

    remote = {item.path: item for item in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True)
              if isinstance(item, RepoFile)}
    same = set()
    for name, path in files.items():
        item = remote.get(name)
        if item is None or item.size != path.stat().st_size:
            continue
        if (item.lfs['sha256'] == file_sha256(path)) if item.lfs else (item.blob_id == blob_id(path)):
            same.add(name)
    return same


def retryable(error: Exception) -> bool:
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if status is not None:
        return status in (429, 500, 502, 503, 504)
    import httpx
    return isinstance(error, httpx.TransportError)    # dropped or timed-out connections


def with_retries(call, waits: tuple[int, ...] = (300,) * 18):
    """Retry a Hub call through the commit rate limit (HTTP 429), server errors and dropped connections, every five
    minutes for up to 90 minutes. Anything else (a refused file, a missing permission) fails at once."""
    import time

    for wait in (*waits, None):
        try:
            return call()
        except Exception as error:
            if wait is None or not retryable(error):
                raise
            print(f'{type(error).__name__}: {str(error)[:120]}; retrying in {wait // 60} minutes', flush=True)
            time.sleep(wait)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--namespace', help='HF user or org; default: the logged-in user')
    parser.add_argument('--cards', type=Path, default=CARDS)
    parser.add_argument('--execute', metavar='DIGEST', help='publish the plan with this digest')
    args = parser.parse_args(argv)
    namespace = args.namespace
    if not namespace:
        from huggingface_hub import whoami
        namespace = whoami()['name']
    repos, private = plan(namespace, cards=args.cards)
    print(describe(repos, private))
    if args.execute:
        execute(repos, args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
