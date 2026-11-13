## Summary

`output_split` shuffles sample by sample, so the translations or paraphrases of one item can land in both train and test, and a test score then measures memorisation.

This adds `output_split_group`, a metadata key whose value keeps a group in one split, and `curatorkit.splits` with `split_samples` and `leakage_report`. `Curator` and `curatorkit run` each had their own copy of the split code; both now call `split_samples`.

- **Grouped split.** A group's split is a hash of its id and `output_split_seed`, so an item keeps its split as the corpus grows, which a shuffle can't promise. The cost is that split sizes only approximate the fractions, more loosely with few groups. A sample with no value for the key is split on its own text, with a warning; a key that no sample has is an error.
- **Leakage check.** `leakage_report(splits, *, group_key=None, near_dup_threshold=0.8)` lists groups, exact prompts and near-duplicate prompts (MinHash, reusing the dedup helpers) found in more than one split, plus an `ok` flag. A grouped run warns when it finds any. It compares prompts, not prompt plus answer, because one prompt with two answers still leaks. Text matching can't see a translation, so only a group key catches those.
- **Unchanged default.** Without `output_split_group`, splitting is the same per-sample shuffle as before; a test pins it to the old code's output for the same seed.
- `output_split_group` is in `CuratorConfig`, the YAML `PipelineConfig` and the run provenance.
- Docs: `configuration.md`, `cli.md`, the splits section of `exporters.md`, and `architecture.md`. CHANGELOG entry under *Unreleased*.

Language ID is not part of this PR.

## Type of change

- [ ] Connector / data source
- [ ] Generation task
- [ ] Quality / hygiene gate
- [x] Exporter
- [ ] Adaptive recovery / diagnostics
- [x] CLI / YAML config
- [ ] Docs / tutorials
- [ ] Bug fix / refactor

## Checklist

- [x] `ruff check .` passes
- [x] `pytest tests/unit -q` passes
- [x] Tests added or updated for behavior changes
- [x] Docs updated (`docs/reference/configuration.md` if `CuratorConfig` changed, `docs/guides/exporters.md` if formats changed)
- [x] `CHANGELOG.md` entry added under *Unreleased*
- [x] No data files or pipeline outputs committed

## Tests

Windows 11, Python 3.12, `pip install -e . pytest pytest-asyncio pytest-cov` (the CI install), branch based on `main` at `1ceea1f`. Each command was run twice with the same result.

- `pytest tests/unit/test_splits.py -q`: 29 passed.
- `pytest tests/unit -q`: 626 passed, 18 skipped (597 passed, 18 skipped on `main`).
- `ruff check .`: all checks passed. The two new files are also `ruff format`-clean.
- `python scripts/check_version_consistency.py`: OK.

## Licence

This contribution is made under CuratorKIT's LSAL v1.1. As `LICENSE.md` section 6 says, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
<!-- Team lead: keep the next sentence only once both copyright holders have agreed to the grant; otherwise delete it. -->
The idea and parts of the code come from our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.

