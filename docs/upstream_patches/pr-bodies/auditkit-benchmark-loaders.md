## Problem

Evaluating safety per language needs samples tagged by language, and AuditKIT has no loader for the multilingual sets people use for it (MultiJail, XSTest, Global-MMLU).

## Change

Adds `src/auditkit/benchmark_loaders.py` with three loaders, exported as `ak.load_multijail`, `ak.load_xstest` and `ak.load_global_mmlu`:

| Loader | Dataset | Returns |
|---|---|---|
| `ak.load_multijail(languages=None)` | `DAMO-NLP-SG/MultiJail` | 315 harmful prompts in each of 10 languages; no reference answer. `metadata["translated"]` is False only for English. |
| `ak.load_xstest()` | `natolambert/xstest-v2-copy` | 250 safe and 200 unsafe English prompts; `metadata["label"]` and `tags` say which, and `metadata["type"]` keeps the prompt type. |
| `ak.load_global_mmlu(languages, split="test")` | `CohereLabs/Global-MMLU` (or `GLOBAL_MMLU_LITE`) | MCQ samples: four `choices` and the 0-based `target`. |

- Each sample's `task` is its language, so `RunComparison.per_task_deltas()` reports every metric per language. Translations of one item share `metadata["group"]`, for splits and paired tests that must keep them together.
- Each loader defaults to a pinned dataset revision and records it in `metadata["revision"]`, as `load_bfcl` does; `revision=` changes it and `repo=` also takes a local folder. `limit=` caps rows (per language where there are several).
- `load_xstest` reads only the prompt file, because the other five splits of that repo hold model completions.
- `datasets` stays a lazy import under the `interop` extra, so the core keeps zero required dependencies. Without it the loaders raise `ExtraNotInstalled`.
- Docs: a "Multilingual benchmarks" section in `scenarios.md`, plus `api.md`, `GUIDE.md` and `src/auditkit/README.md`. CHANGELOG entries in `CHANGELOG.md` and `docs/community/changelog.md`.

No dataset rows are added to the repo. The sources are MIT (MultiJail), CC-BY-4.0 (XSTest) and Apache-2.0 (Global-MMLU).

One choice for you: the XSTest default is `natolambert/xstest-v2-copy`, a third-party copy of the v2 prompts. SafeTune uses `walledai/XSTest`. Either works with `repo=`; tell me which you'd rather have as the default.

## Tests

Windows 11, Python 3.12, `pip install -e ".[dev,transformers]"` (the CI install), branch based on `main` at `1444ba5`. Each run was done twice.

- `pytest tests/test_benchmark_loaders.py`: 34 passed, both runs, and 34 passed on Python 3.10 with `.[dev]`. The Hub is faked, so these run offline.
- Full suite (`pytest`): 4354 passed and no failures, both runs: the 4320 tests that pass on `main` plus the 34 new ones. The skips are the same as on `main` (optional extras such as `datasets`, `lm_eval` and `bert-score` are not in this install). On `main`, `test_agent_endpoint.py::TestContract::test_default_round_trip` failed once in two runs (`latency_ms` was 0.0); it passed in both runs here.

When this change was first written against `71316da`, the loaders were also run against the real datasets at their pinned revisions: MultiJail 3,150 samples, XSTest 450 (250 safe, 200 unsafe), Global-MMLU-Lite 400 per language, and Global-MMLU `sw` dev 285. That run was not repeated on the current `main`.

## Licence

This contribution is made under AuditKIT's LSAL v1.2. As `LICENSE.md` section 5 says, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
<!-- Team lead: keep the next sentence only once both copyright holders have agreed to the grant; otherwise delete it. -->
The code was first written for our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.

