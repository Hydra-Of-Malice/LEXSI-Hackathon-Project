# Private review workflow

The benchmark has two separate human-evidence gates:

1. **Translation review** checks that every generated or code-mixed prompt preserves the English source.
2. **Judge calibration** checks the model-judge labels against a complete, deterministic hand-check sample.

Neither gate can be completed by rerunning a model, copying a machine label, or changing a manifest field. Review
packets contain harmful prompts and completions, so keep them under the private `runs/` directory. Do not commit or
publish the CSVs, source references, or hand-check sheets.

Your team supplies the human evidence. Agents, local judges, and router models
do not count as human reviewers. Assign Hindi and Hinglish, Bengali, and Tamil
rows to reviewers who can read those languages and understand the safety
meaning; use a second human for `unclear` or rejected rows when needed. The
current full-data packet is
`runs/experiments/full/fire/review/translation_review.with-source.csv` with
1,766 source-backed rows and no labels. Its private source reference is
`runs/experiments/full/fire/review/multijail-source-reference.jsonl`. The
development packet is
`runs/experiments/development/fire/review/translation_review.with-source.csv`
with 50 rows and no labels.

## Translation packet

The packet builder verifies both JSONL files against the immutable hashes in the data manifest and writes a blank
CSV. It never fills `review_label`, `reviewer_id`, or `reviewed_at`.

```bash
.venv/bin/python scripts/review_data.py export \
  --data-dir runs/experiments/development/fire/data \
  --manifest runs/experiments/development/fire/data/manifest.json \
  --out runs/experiments/development/fire/review/translation_review.csv
```

If a translation's English source was dropped by the per-language cap, provide a private JSONL source index. Each
line must contain an `id` matching `source_id` and the original `text`; the index is hashed into every packet row.
The development packet uses `runs/experiments/development/fire/review/multijail-source-reference.jsonl` for five
capped-out MultiJail source IDs:

```bash
.venv/bin/python scripts/review_data.py export \
  --data-dir runs/experiments/development/fire/data \
  --manifest runs/experiments/development/fire/data/manifest.json \
  --source-reference runs/experiments/development/fire/review/multijail-source-reference.jsonl \
  --out runs/experiments/development/fire/review/translation_review.with-source.csv
```

The reviewer edits only these fields:

| Field | Allowed value |
| --- | --- |
| `review_label` | `accepted`, `rejected`, or `unclear` (the importer also accepts `accept`/`reject`) |
| `reviewer_id` | A stable human reviewer identifier; use the same value for that reviewer’s rows |
| `reviewed_at` | An ISO-8601 timestamp with a timezone, for example `2026-11-04T10:00:00+00:00` |
| `review_notes` | Optional short rationale; never put credentials or unrelated personal data here |

Review every source-available row. A row with no source is blocked and cannot receive a label; add the correct
private source reference and export a fresh packet instead. Do not edit the prompt, translation, hashes, IDs, or
manifest columns. The importer rejects edited or stale evidence, duplicate identities, missing reviewer metadata,
machine reviewer IDs, and labels for rows whose source is unavailable.

Validate the completed packet without exposing its text in the terminal:

```bash
.venv/bin/python scripts/review_data.py check \
  --data-dir runs/experiments/development/fire/data \
  --manifest runs/experiments/development/fire/data/manifest.json \
  --source-reference runs/experiments/development/fire/review/multijail-source-reference.jsonl \
  --packet runs/experiments/development/fire/review/translation_review.with-source.csv \
  --out runs/experiments/development/fire/review/translation_review.checked.json
```

The JSON summary is descriptive evidence. `quality_status` becomes `human_reviewed` only when every row is labelled
`accepted`; any `rejected` or `unclear` decision produces `human_reviewed_with_rejections`, and missing labels or
source references leave the status `unreviewed`. The importer does not modify `manifest.json`. The orchestration
layer must carry the summary’s `manifest_sha256`, `data_files`, `source_reference_sha256`, and `packet_id` into its
report so a later data rebuild cannot reuse this review.

The module API is also available for offline tooling:

```python
from langsafe.data.review import export_translation_review, import_translation_review

export_translation_review(data_dir, manifest_path, packet_path, source_reference)
import_translation_review(packet_path, manifest_path, data_dir, checked_json, source_reference)
```

## Judge calibration

Run a judge pass first. The judge writes a private JSONL and creates a sheet containing the deterministic random
sample plus every disagreement or parse error:

```bash
.venv/bin/python -m langsafe.cli --spec configs/development.yaml \
  --runs-dir runs/experiments/development/fire \
  judge --tag fire --judges local --split heldout
```

A reviewer fills `hand_label` with `full_refusal`, `partial_refusal`, `full_compliance`, or `unclear`. Leave no
selected audit or disagreement row blank. Then import it against the exact judged JSONL through the existing
calibration command:

```bash
.venv/bin/python -m langsafe.cli --spec configs/development.yaml \
  --runs-dir runs/experiments/development/fire \
  calibrate --tag fire --judges local \
  --sheet runs/experiments/development/fire/hand_check/fire.local.csv
```

The importer binds each label to the judged row’s generation key, prompt hash, response hash, audit size, audit
seed, rubric, and evaluation digest. It preserves a supplied label only when that exact evidence is still present.
`by_lang.audit_n`, `audit_judge_accuracy`, and `audit_consensus_accuracy` describe the random audit separately from
the disagreement queue; no accuracy value is inferred for missing labels. The report gate requires two independent
judges, measurable per-language kappa at the configured floor, a complete verified human sample, and (for local
judges) a verified shared-sample router comparison. Human accuracy has no invented cutoff in the current spec; low
accuracy remains visible for review rather than being silently converted into a pass.

For a local judge result, run a router judge on the same generations and compare only matching, non-ambiguous rows:

```bash
.venv/bin/python -m langsafe.cli --spec configs/development.yaml \
  --runs-dir runs/experiments/development/fire \
  judge --tag fire --judges router --split heldout
.venv/bin/python -m langsafe.cli --spec configs/development.yaml \
  --runs-dir runs/experiments/development/fire \
  calibrate --tag fire --compare-router
```

Translation quality and judge calibration are independent. A calibrated judge does not make an unreviewed
translation scientifically valid, and a reviewed translation does not establish judge agreement. Keep both summaries
alongside the private run; only aggregate rates and gate status belong in public result artifacts.
