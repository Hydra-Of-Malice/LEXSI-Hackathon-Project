"""Build a clean public copy of this repo for the team to review: the committed tree at HEAD, in a new folder with fresh
history, with the private notes left out.

    .venv/Scripts/python.exe scripts/make_public_copy.py "../LEXSI-Hackathon-Public" \
        --author "Name <id+login@users.noreply.github.com>" [--leave-out PATH ...] [--replace]

It never pushes and sets no remote: publishing stays a team decision. It starts from `git archive HEAD`, so uncommitted
edits and ignored files, which include every model output, cannot get in. Each edit below must match exactly once, so a
changed source file stops the build instead of leaving a stale sentence. The build also stops when a kept file still
names a left-out file, or when the copy holds anything that looks like a secret. It lists, without stopping, the places
that hold this machine's user name or hostname, for a person to judge.

A fresh history drops the authors' email addresses with the old commits. `--author` should be a GitHub noreply address.
"""
from __future__ import annotations

import argparse
import getpass
import io
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import secret_scan  # noqa: E402

MARKER = 'public-copy-source'      # kept inside .git, so it is never published


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str


LEAVE_OUT = {
    'docs/research/07_discord_intel.md': 'notes on other teams',
    'docs/research/08_kavach_crossover.md': 'notes on an earlier project of the team',
    'docs/research/raw': "the organisers' own text, slides and poster, copied word for word",
}
# Tools for the working repo. They name the left-out files by design, so they stay out and their names aren't checked.
TOOLING = ('scripts/make_public_copy.py', 'tests/test_make_public_copy.py')

EDITS = (
    Edit('README.md', 'Hackathon rules, judging, models, kit deep-dives, Discord clarifications, idea ranking',
         'Hackathon rules, judging, models, kit deep-dives, idea ranking'),
    Edit('docs/research/README.md',
         '| [07_discord_intel.md](07_discord_intel.md) | Discord scan: organiser clarifications, what other teams are '
         'building, re-ranked ideas, action items |\n'
         '| [08_kavach_crossover.md](08_kavach_crossover.md) | What our AgentVerse project Kavach can lend to the Lexsi '
         'build; options K1–K3 |\n'
         '| raw/ | Scraped slide text, site text, poster |\n', ''),
    Edit('docs/submission/checklist.md',
         'Sources: hackathon site and kickoff slides (`docs/research/01_official_brief.md`), Discord clarifications '
         '(`docs/research/07_discord_intel.md`), and a read of the code and repo state.',
         "Sources: hackathon site and kickoff slides (`docs/research/01_official_brief.md`), the organisers' answers "
         'on Discord, and a read of the code and repo state.'),
    Edit('docs/submission/checklist.md',
         '- [ ] `docs/research/07_discord_intel.md` names other participants and gives a competitive read of their '
         'projects. Remove it or move it out.\n'
         '- [ ] `docs/research/08_kavach_crossover.md` describes a private repo and a prior submission.\n'
         '- [ ] Removing a file does not remove it from history. Decide whether to rewrite history or start a clean '
         'public repo.\n',
         '- [x] Two private research notes and the organisers\' raw material are left out of the public repo, which '
         'starts with fresh history.\n'),
    Edit('docs/submission/checklist.md',
         "- [ ] Commit history shows the authors' email addresses: 76 of 77 commits carry a personal Gmail address, "
         'which goes public with the history. Decide whether that is acceptable, or start the public repo clean.\n',
         "- [x] The working repo's history carried the authors' email addresses. The public repo starts with fresh "
         'history under a GitHub noreply address.\n'),
)


class BuildError(RuntimeError):
    pass


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(['git', '-C', str(repo), *args], input=stdin, capture_output=True)
    if result.returncode:
        raise BuildError(f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def _writable_then_retry(function, path, _error) -> None:
    """Git stores its objects read-only, which stops a delete on Windows."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _prepare(out: Path, replace: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not (out / '.git' / MARKER).is_file():
            raise BuildError(f'{out} is not empty and was not made by this script; refusing to touch it')
        if not replace:
            raise BuildError(f'{out} already holds a public copy; pass --replace to rebuild it')
        shutil.rmtree(out, onexc=_writable_then_retry)
    out.mkdir(parents=True, exist_ok=True)


def _text_files(out: Path):
    for path in sorted(out.rglob('*')):
        if '.git' not in path.relative_to(out).parts and path.is_file():
            try:
                yield path, path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue


def build(source: Path, out: Path, author: str, *, leave_out: dict[str, str] = LEAVE_OUT,
          edits: tuple[Edit, ...] = EDITS, replace: bool = False) -> dict:
    match = re.fullmatch(r'(.+?) <([^<>@\s]+@[^<>\s]+)>', author.strip())
    if not match:
        raise BuildError('--author must look like "Name <email>"')
    if not match.group(2).endswith('@users.noreply.github.com'):
        raise BuildError('--author needs a GitHub noreply address, so no personal email goes public')
    commit = _git(source, 'rev-parse', '--short', 'HEAD').decode().strip()
    _prepare(out, replace)
    with tarfile.open(fileobj=io.BytesIO(_git(source, 'archive', '--format=tar', 'HEAD'))) as archive:
        archive.extractall(out, filter='data')
    for rel in leave_out:
        target = out / rel
        if not target.exists():
            raise BuildError(f'{rel} is not in the committed tree, so the leave-out list is stale')
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    for rel in TOOLING:
        (out / rel).unlink(missing_ok=True)
    for edit in edits:
        path = out / edit.path
        text = path.read_text(encoding='utf-8')
        if text.count(edit.old) != 1:
            raise BuildError(f'an edit for {edit.path} matched {text.count(edit.old)} times, not once: '
                             f'{edit.old[:60]!r}')
        path.write_text(text.replace(edit.old, edit.new), encoding='utf-8', newline='\n')
    names = {rel: Path(rel).name for rel in leave_out}
    dangling = [f'{path.relative_to(out).as_posix()} names {rel}' for path, text in _text_files(out)
                for rel, name in names.items() if rel in text or (Path(rel).suffix and name in text)]
    if dangling:
        raise BuildError('kept files still name left-out files: ' + '; '.join(dangling[:8]))
    secrets = secret_scan.scan_tree(out)
    if secrets:
        raise BuildError('possible secrets in the copy: ' + '; '.join(secret_scan.report(secrets)))
    local = [token for token in {getpass.getuser(), platform.node()} if len(token) >= 4]
    local_hits = [f'{path.relative_to(out).as_posix()}: {token}' for path, text in _text_files(out)
                  for token in local if token.lower() in text.lower()]
    _git(out, 'init', '-q', '-b', 'main')
    _git(out, 'add', '-A')
    name, email = match.groups()
    message = (f'Public copy of the project at {commit}\n\nBuilt by scripts/make_public_copy.py from the committed tree, '
               'with fresh history and the private notes left out.\n\n'
               '')
    _git(out, '-c', f'user.name={name}', '-c', f'user.email={email}', '-c', 'commit.gpgsign=false',
         'commit', '-q', '--author', f'{name} <{email}>', '-F', '-', stdin=message.encode())
    (out / '.git' / MARKER).write_text(f'{source.resolve()}\n{commit}\n', encoding='utf-8')
    files = _git(out, 'ls-files').decode('utf-8', 'replace').splitlines()
    return {'source_commit': commit, 'files': len(files), 'left_out': sorted(leave_out), 'tooling': list(TOOLING),
            'edits': len(edits), 'local_details': local_hits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='new folder for the copy, outside this repo')
    parser.add_argument('--author', required=True, help='"Name <email>" for the single commit, ideally a noreply address')
    parser.add_argument('--leave-out', action='append', default=[], metavar='PATH',
                        help='also leave out this committed file or folder, a team decision')
    parser.add_argument('--replace', action='store_true', help='rebuild a copy this script made earlier')
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        print('The copy must go outside this repo.', file=sys.stderr)
        return 2
    leave_out = {**LEAVE_OUT, **{path.strip('/'): 'a team decision' for path in args.leave_out}}
    try:
        result = build(ROOT, out, args.author, leave_out=leave_out, replace=args.replace)
    except BuildError as error:
        print(f'Not built: {error}', file=sys.stderr)
        return 1
    print(f"Built {out} from {result['source_commit']}: {result['files']} files, one commit, no remote.")
    for rel in result['left_out']:
        print(f'  left out: {rel} ({leave_out[rel]})')
    for hit in result['local_details']:
        print(f'  holds a local detail, check it: {hit}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
