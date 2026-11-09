from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_public_copy  # noqa: E402
import secret_scan  # noqa: E402

AUTHOR = 'Team Member <123+member@users.noreply.github.com>'
Edit = make_public_copy.Edit
TOKEN = 'hf_' + 'Q7w' * 11          # built at run time, so this file holds no token-shaped text


def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True, text=True).stdout


def toy_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    repo = tmp_path / 'repo'
    files = files or {'README.md': 'Project.\n', 'docs/index.md': 'Start here.\nSee private.md for notes.\n',
                      'docs/private.md': 'Private notes.\n', 'keep.txt': 'committed\n', '.gitignore': 'runs/\n'}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding='utf-8')
    git(repo.parent, 'init', '-q', str(repo))
    git(repo, 'add', '-A')
    git(repo, '-c', 'user.name=Someone', '-c', 'user.email=someone@personal.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-q', '-m', 'first')
    return repo


LEAVE = {'docs/private.md': 'private'}
EDITS = (Edit('docs/index.md', 'See private.md for notes.\n', ''),)


def test_the_copy_holds_only_the_committed_tree_minus_the_private_files(tmp_path):
    repo = toy_repo(tmp_path)
    (repo / 'keep.txt').write_text('uncommitted edit\n', encoding='utf-8')
    (repo / 'runs').mkdir()
    (repo / 'runs' / 'gens.jsonl').write_text('model output\n', encoding='utf-8')
    out = tmp_path / 'public'
    result = make_public_copy.build(repo, out, AUTHOR, leave_out=LEAVE, edits=EDITS)
    assert sorted(git(out, 'ls-files').split()) == ['.gitignore', 'README.md', 'docs/index.md', 'keep.txt']
    assert (out / 'keep.txt').read_text(encoding='utf-8') == 'committed\n'
    assert (out / 'docs' / 'index.md').read_text(encoding='utf-8') == 'Start here.\n'
    assert not (out / 'runs').exists() and result['files'] == 4
    assert git(out, 'rev-list', '--count', 'HEAD').strip() == '1'
    assert git(out, 'log', '--format=%an <%ae>|%cn <%ce>').strip() == f'{AUTHOR}|{AUTHOR}'
    assert 'personal.invalid' not in git(out, 'log', '--all', '--format=%ae %ce')
    assert git(out, 'remote').strip() == ''


def test_a_kept_file_that_still_names_a_left_out_file_stops_the_build(tmp_path):
    repo = toy_repo(tmp_path)
    with pytest.raises(make_public_copy.BuildError, match='docs/index.md names docs/private.md'):
        make_public_copy.build(repo, tmp_path / 'public', AUTHOR, leave_out=LEAVE, edits=())


def test_an_edit_that_no_longer_matches_stops_the_build(tmp_path):
    repo = toy_repo(tmp_path)
    stale = (Edit('docs/index.md', 'See notes.md instead.\n', ''),)
    with pytest.raises(make_public_copy.BuildError, match='matched 0 times'):
        make_public_copy.build(repo, tmp_path / 'public', AUTHOR, leave_out=LEAVE, edits=stale)


def test_a_left_out_path_missing_from_the_tree_stops_the_build(tmp_path):
    repo = toy_repo(tmp_path)
    with pytest.raises(make_public_copy.BuildError, match='stale'):
        make_public_copy.build(repo, tmp_path / 'public', AUTHOR, leave_out={'docs/gone.md': 'x'}, edits=())


def test_a_secret_in_a_committed_file_stops_the_build(tmp_path):
    repo = toy_repo(tmp_path, {'config.py': f'TOKEN = "{TOKEN}"\n'})
    with pytest.raises(make_public_copy.BuildError, match='huggingface token') as error:
        make_public_copy.build(repo, tmp_path / 'public', AUTHOR, leave_out={}, edits=())
    assert TOKEN not in str(error.value)


def test_only_a_noreply_author_is_accepted(tmp_path):
    repo = toy_repo(tmp_path)
    for author in ('Team Member <member@gmail.com>', 'no email at all'):
        with pytest.raises(make_public_copy.BuildError):
            make_public_copy.build(repo, tmp_path / 'public', author, leave_out=LEAVE, edits=EDITS)


def test_it_never_overwrites_a_folder_it_did_not_make(tmp_path):
    repo = toy_repo(tmp_path)
    out = tmp_path / 'public'
    out.mkdir()
    (out / 'someone_elses.txt').write_text('keep me\n', encoding='utf-8')
    with pytest.raises(make_public_copy.BuildError, match='not made by this script'):
        make_public_copy.build(repo, out, AUTHOR, leave_out=LEAVE, edits=EDITS, replace=True)
    assert (out / 'someone_elses.txt').exists()
    fresh = tmp_path / 'fresh'
    make_public_copy.build(repo, fresh, AUTHOR, leave_out=LEAVE, edits=EDITS)
    with pytest.raises(make_public_copy.BuildError, match='--replace'):
        make_public_copy.build(repo, fresh, AUTHOR, leave_out=LEAVE, edits=EDITS)
    make_public_copy.build(repo, fresh, AUTHOR, leave_out=LEAVE, edits=EDITS, replace=True)
    assert git(fresh, 'rev-list', '--count', 'HEAD').strip() == '1'


def test_the_real_repo_builds_with_its_private_notes_and_tooling_left_out(tmp_path):
    out = tmp_path / 'public'
    result = make_public_copy.build(ROOT, out, AUTHOR)
    kept = set(git(out, 'ls-files').splitlines())
    for rel in [*make_public_copy.LEAVE_OUT, *make_public_copy.TOOLING]:
        assert not any(path == rel or path.startswith(rel + '/') for path in kept), rel
    assert 'scripts/secret_scan.py' in kept or 'scripts/secret_scan.py' not in git(ROOT, 'ls-files', 'scripts')
    assert 'Discord clarifications' not in (out / 'README.md').read_text(encoding='utf-8')
    assert result['files'] == len(kept)


def test_the_scanner_finds_real_looking_tokens_and_skips_placeholders():
    found = secret_scan.scan_bytes(f'key = "{TOKEN}"\n'.encode())
    assert [kind for kind, _, _ in found] == ['huggingface token']
    assert TOKEN not in secret_scan.mask(found[0][1]) and secret_scan.mask(found[0][1]).startswith('hf_Q7w')
    for text in ("secret = 'PER-ITEM-DETAIL'", 'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"',
                 "password = '{{PASSWORD}}'", 'api_key = os.environ["HF_TOKEN"]'):
        assert secret_scan.scan_bytes(text.encode()) == [], text
    assert [k for k, _, _ in secret_scan.scan_bytes(b"api_key = 'a1b2c3d4e5f6g7h8i9'")] == ['assignment of a secret']
