## Problem

`RefusalDetector` matches English phrases only. A refusal in any other language scores as "did not refuse", so refusal rates and attack success on multilingual safety sets are wrong.

## Change

Adds `src/auditkit/metrics/refusal.py` with refusal phrases for English, Hindi, Hinglish (Hindi in Latin script), Bengali, Tamil, Arabic and Swahili, exposed three ways:

- `ak.multilingual_refusal`: an `@ak.scorer` that gives 1.0 when the output opens with a refusal, else 0.0. The language comes from `sample.metadata["lang"]`, else `sample.task`.
- `ak.refusal_scorer(name, direction)`: the same check under another name or direction. The default (`MAXIMIZE`) suits harmful prompts; `refusal_scorer("over_refusal", direction=ak.Direction.MINIMIZE)` suits benign ones.
- `refusal_multilingual`: a red-team detector, registered in `RedTeamRunner` and `auditkit redteam --detectors`. It runs after the runner's invalid-reply and `<think>` handling from 1.1.9, like the other detectors.

Details:

- Text is NFC-normalised, apostrophe look-alikes fold with the shared `fold_apostrophes` helper, and zero-width joiners are dropped, so the way a phrase is encoded can't hide it. NFC matters for Indic text: `फ़` and `য়` each have a one-character and a two-character form, and models emit both.
- English is always checked as well, because models often switch to English to refuse.
- A language tag with no patterns (`zh`) is reported as `not_tested` (`metadata={"unknown": True, "status": "not_tested"}`) and left out of the aggregate, because a 0.0 there would read as a failed refusal and inflate attack success. A task name that isn't a language tag checks every language.
- The scorer reads the answer without its `<think>` reasoning (`auditkit.trace.strip_thinking`), as `RedTeamRunner` has done since 1.1.9, so a refusal or a plan inside the reasoning doesn't count. A `<think>` that is never closed is treated as the runner treats it when the reply wasn't cut at the token limit: the tag is dropped and the text kept.
- An empty answer, or a reply that is only reasoning, is a `not_measured` result (`metadata={"unknown": True, "status": "not_measured"}`), not a refusal. As with other results that could not be decided (#25), it is left out of the aggregate and listed in `RunResult.errors` with its reason ("empty output" or "reasoning only, no answer"). This matches the red-team runner, where such replies are invalid rather than passes (#40).
- Only the first 300 characters of the answer are searched.
- `RefusalDetector` is unchanged.
- Docs: `scorers_reference.md` (new section), `api.md`, `GUIDE.md` and `cli.md` (detector lists), `metrics/README.md`. CHANGELOG entries in `CHANGELOG.md` and `docs/community/changelog.md`.

Limits: this is a phrase list. It misses paraphrases, and it can fire on a reply that only discusses legality. Its precision and recall have not been measured, so use a judge for a final number. In `RedTeamRunner` the detector only sees the answer, since the runner strips reasoning and marks an empty reply invalid before any detector runs.

It is an `@ak.scorer` rather than a `Metric` subclass. Happy to turn it into a `Metric` if you prefer that for built-ins.

## Tests

Windows 11, Python 3.12, `pip install -e ".[dev,transformers]"` (the CI install), branch based on `main` at `1444ba5`.

- `pytest tests/test_refusal_multilingual.py tests/test_redteam.py tests/test_redteam_invalid.py tests/test_apostrophe_fold.py`: 122 passed, run twice. The same four files also pass (122, twice) on Python 3.10 with `.[dev]`.
- Full suite (`pytest`): 4402 passed, 141 skipped, no failures. That is the 4320 tests that pass on `main` plus the 82 new ones, and the same 141 skips as on `main` (optional extras such as `datasets`, `lm_eval` and `bert-score` are not in this install). On `main`, `test_agent_endpoint.py::TestContract::test_default_round_trip` failed once in two runs (`latency_ms` was 0.0); it passed here.

`tests/test_refusal_multilingual.py` covers each language's refusals and plain answers, Unicode variants (nukta as one or two code points, ZWJ and ZWNJ, curly apostrophes), the Hinglish aliases, the 300-character window, `not_tested` through `ak.evaluate`, the over-refusal direction, and the detector through `RedTeamRunner`. It also covers empty and reasoning-only replies as `not_measured` (for both directions, and through `ak.evaluate`) and reasoning being ignored: complete, repeated, template-opened (a lone `</think>`), unclosed, and longer than the 300-character window. Ten of those 13 tests fail against the scorer without this handling.

**Native-speaker review needed.** The Hindi, Hinglish, Bengali, Tamil, Arabic and Swahili test sentences, and the patterns themselves, were written for this change and have not been reviewed by native speakers. A review of those strings would be welcome.

Rebased on 13 November onto `main` at `556eb48`, with no conflicts. Re-run there on macOS, Python 3.12: `pytest tests/test_refusal_multilingual.py tests/test_redteam.py`: 104 passed.

## Licence

This contribution is made under AuditKIT's LSAL v1.2. As `LICENSE.md` section 5 says, it grants Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute and license it under any terms, including commercial ones, and I have the right to submit it.
The patterns were first written for our MIT-licensed project langsafe, and both of its copyright holders (Aditya Arnav and Pallab Das) agree to this grant.
