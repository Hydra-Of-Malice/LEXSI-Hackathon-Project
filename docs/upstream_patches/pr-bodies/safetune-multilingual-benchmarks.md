## Summary

Every built-in safety benchmark is English, so there is no way to measure whether a repair holds in other languages. This adds two multilingual jailbreak benchmarks, `multijail` (315 prompts in each of 10 languages, translated) and `aya_redteaming` (7,419 prompts written by native speakers of 8 languages), to the dataset table, the `evaluate()` registry and `load_prompts()`.

## Pillar / method affected

Evaluate: `safetune.data.loaders.benchmarks`, `safetune.data.dataset_ids`, `safetune.core.eval.pipeline.loaders` and the `evaluate()` benchmark registry.

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] Faithfulness correction (method now matches its cited paper)
- [x] New method / benchmark
- [ ] Breaking change
- [ ] Documentation
- [ ] Other

## What changes

- `load_multijail(languages=None)`: MultiJail (Deng et al., arXiv:2310.06474, `DAMO-NLP-SG/MultiJail`) has a column per language, so its rows are unpivoted into one row per prompt and language, with `prompt`, `language`, `id` (shared by a prompt's translations) and `category`.
- `load_aya_redteaming(languages=None)`: Aya Red-teaming (Aakanksha et al., arXiv:2406.18682, `CohereLabs/aya_redteaming`) has one split per language and no `train` split, so its table entry names no split and the loader asks for each one. A split set in the table or passed as `split=` loads only that split, which suits a local copy.
- Both are in `DATASETS`, the `evaluate()` registry (category `jailbreak`, tag `multilingual`) and `load_prompts()`. Each row has a `language` column, and `languages=` (a code or a list) picks a subset; an unknown code is an error.
- Docs: the benchmark index (22 registered benchmarks, a table row for each, and a warning) and the references table. CHANGELOG entry under `[Unreleased]` / Added.

This adds data only. The refusal string lists are English and the default `wildguard` judge has not been validated on these languages. A refusal a judge can't read counts as an attack success, which inflates the ASR, so the docs and both registry descriptions say to check the judge first. No audit badge was added to `docs/trust/feature-map.md`; that verdict is yours.

## Audit badge change (if applicable)

None.

## Checklist

- [x] Code follows the repo style (black, isort, line-length 100). New functions and the new test file are black- and isort-clean. The one-line additions to existing tables (`DATASETS`, the test fixtures, the loader import list) follow the surrounding table style, which black and isort already change on `main`.
- [x] Tests added / updated for the change
- [ ] `pytest tests/` passes locally. Only the related files were run (below), not the full suite.
- [x] `docs/trust/feature-map.md` updated if a badge changed (no badge changed)
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] No secrets / credentials committed

## Tests

Windows 11, Python 3.12, `pip install -e ".[dev]"`, branch based on `main` at `263ea24`. Each command was run twice.

- `pytest tests/runtime/test_multilingual_benchmarks.py`: 18 passed, both runs. The Hub is faked, so these run offline.
- `pytest tests/runtime/test_multilingual_benchmarks.py tests/runtime/test_eval_reliability.py tests/runtime/test_runtime_config.py`: 159 passed and 1 skipped, then 160 passed. The skip is the existing tiny-Llama test in `test_eval_reliability.py`, which skips until `hf-internal-testing/tiny-random-LlamaForCausalLM` is fully in the local Hugging Face cache; it was cached by the second run. On `main` the last two files give 138 passed.

When this change was first written against `3901a9f`, both loaders were also run against the real datasets: Aya Red-teaming gave 7,419 rows across 8 languages, and MultiJail 3,150 rows. That run was not repeated on the current `main`.

Rebased on 13 November onto `main` at `cc3de5f`, with no conflicts. Re-run there on macOS, Python 3.12: `pytest tests/runtime/test_multilingual_benchmarks.py`: 18 passed.

## Licence

This contribution is made under SafeTune's LSAL v1.2. As `LICENSE.md` section 6 and `CONTRIBUTING.md` say, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
The code was first written for our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.

The benchmarks themselves are not copied into the repo; the loaders read them from the Hub under their own licences (MultiJail is MIT, Aya Red-teaming is Apache-2.0).
