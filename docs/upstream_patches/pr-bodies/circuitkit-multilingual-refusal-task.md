## Description

- **What problem does this PR solve?** CircuitKIT has no cross-language task and no refusal task for chat models, and `binary_align/jailbreak_binary.csv`, despite its README, is used by no task.
- **What type of change is this?** A new task, `multilingual_refusal`.
- **Are there any breaking changes?** No. Existing tasks are untouched.

`multilingual_refusal` finds the refusal circuit of a chat model in one language per run. Each pair is a harmful and a harmless request that differ in one or two words ("trim a drug supply" / "trim a long hedge"). The harmful request is the clean prompt, scored on the first token of a refusal; the harmless one is the corrupted prompt, scored on the first token of a compliance. The metric is the logit difference between the two.

## Related Issues

None.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring or code cleanup
- [ ] Test improvement

## Changes Made

- `src/circuitkit/tasks/builtins/multilingual_refusal.py`: `MultilingualRefusalTaskSpec`, registered in `bootstrap.py` and `builtins/__init__.py`.
- **Pairs.** English pairs come from `jailbreak_binary.csv`. Other languages take a `pairs_path` CSV with `harmful` and `harmless` columns, because a pair has to be translated together, so the swap stays minimal, and reviewed by someone who reads the language. CircuitKIT ships no translations.
- **Chat template.** Prompts go through the chat template, with `prepend_bos` handled as the other chat tasks do, so each prompt has one BOS. The default `chat_template_mode` is `"on"`, not `"auto"`, because a base model has no refusal behaviour to trace, so a tokenizer without a template is an error. `"off"` takes pre-rendered pairs.
- **Template check.** The guard for a missing template accepts a dict or list as well as a string. `_chat._has_chat_template` accepts only a string, and Tiny Aya's `chat_template` is a dict of named templates. That helper is unchanged here, because fixing it changes `"auto"` for every other task; it may deserve its own PR.
- **Token lengths.** A pair whose prompts tokenize to two lengths is dropped, and the kept count is logged. Openers that share a first token are an error, because the metric could not separate them.
- **Openers.** `refuse` and `comply` default per language for `en`, `hi`, `hi-Latn`, `bn`, `ta`, `ar` and `sw`. They are a starting point; what the model actually says in that language is better.
- **Algorithms.** EAP family only; `ibcircuit` and `cdt` are rejected with the usual message.
- **Partitions.** `data_partition` uses a fixed partition seed, so the discovery and evaluation halves stay disjoint whatever the run seed.
- Docs: `docs/user-guide/tasks.md` (table row and a section with an example) and `docs/advanced/chat-templates.md`. CHANGELOG entry under `[Unreleased]`.

## Testing

- [x] Unit tests added/updated
- [ ] Integration tests added/updated
- [x] Manual testing completed
- [x] Existing tests still pass

**Test Coverage**: Windows 11, Python 3.12, `pip install -e ".[dev]"`, branch based on `main` at `e3501bc`. Each command was run twice with the same result.

- `pytest tests/unit/test_multilingual_refusal.py`: 43 passed. The tests use fake tokenizers and models: pair loading and validation, opener checks, chat-template rendering and BOS handling, dropped pairs, partitions, caching, metric sign, algorithm checks and registration.
- `pytest tests/unit tests/test_custom_data.py` (the default fast tier from `pytest.ini`): 1703 passed, 12 failed, 17 errors, 28 skipped. On `main` the same command gives 12 failed, 1660 passed and 17 errors. The same 29 fail on both, in `test_graph`, `test_security_hardening`, `test_steering` and `test_structural_pruner`. They come from this Windows machine (path separators, a temp file held open) and from Hugging Face files missing under the suite's offline mode, not from this change.
- Manual check, done when this was first written against `e71d5e1`: with Tiny Aya's real tokenizer (tokenizer files only, no weights), 239 of the 334 English pairs survive token alignment, each prompt has exactly one BOS, and the default openers start on different tokens in all seven languages.

Not verified: no model was loaded, so no circuit has been discovered with this task through EAP-IG yet, and non-English pairs were tested only with fakes.

## Code Quality

- [x] My code follows the project's code style
- [x] I have added docstrings to new functions/classes
- [x] I have updated documentation where relevant
- [x] No unnecessary debug code or print statements
- [x] No unused imports
- [x] Type hints added for new functions

## Dependencies

- [x] No new dependencies added
- [ ] All new dependencies are properly documented
- [ ] Version constraints are reasonable

## Checklist

- [x] My changes are complete and ready for review
- [x] I have self-reviewed my own code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes do not break any existing tests
- [x] I have added tests that prove my fix is effective or that my feature works

## Additional Context

**Licence.** This contribution is made under CircuitKIT's LSAL v1.2. As `LICENSE.md` section 6 and `CONTRIBUTING.md` say, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
<!-- Team lead: keep the next sentence only once both copyright holders have agreed to the grant; otherwise delete it. -->
It was first written for our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.

The non-English default openers were written for this task and have not been reviewed by native speakers.

