---
license: cc-by-nc-4.0
language:
  - en
  - hi
  - bn
  - ta
pretty_name: "{{DATASET_NAME}}"
task_categories:
  - text-generation
tags:
  - safety
  - red-teaming
  - refusal
  - over-refusal
  - multilingual
  - code-mixed
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: heldout
        path: data/heldout.jsonl
      - split: repair
        path: data/repair.jsonl
---

<!--
TEMPLATE FOR THE TEAM. Delete this comment before publishing.

- {{NAME}} marks a value we do not have yet. The list at the end has every one. Delete that list too.
- Fill counts from the final data/manifest.json. The Mac build recorded 2,613 prompts (1,580 held out, 1,033 repair),
  1,766 translated or code-mixed rows and 134 rejected translations. Use those only if the final build matches.
- Check that the manifest's `curatorkit` field is filled. The pilot manifests have it null, because pilots skip CuratorKIT.
- If the Earth family is built, add ar and sw to the YAML and to every table.
- Before uploading, confirm the files hold prompts and labels only. Never upload anything from runs/gens, runs/judged,
  runs/hand_check, runs/auditkit or the review CSVs.
-->

# {{DATASET_NAME}}

> **Content warning.** This dataset contains harmful requests, including requests about violence, self-harm, hate and crime. They exist to test whether models refuse. Do not use them to train a model to comply.

A multilingual prompt set for measuring content safety, refusal and benign over-refusal in small language models, one language at a time. It was built to test Tiny Aya's regional models.

**It contains prompts and labels only. It contains no model answers.** Machine-translated rows are model-generated translations of public prompts. They are prompts, and they are marked as such. No answer from any model to any prompt is included.

Write-up: {{ARTICLE_URL}}. Collection: {{COLLECTION_DATA_URL}}. Date of this build: {{CARD_DATE}}.

## What it is for

- Comparing safety and refusal behaviour across languages, on identical settings.
- Testing whether a safety edit improves answers to harmful prompts without raising refusals on harmless look-alikes.
- Keeping repair data and test data strictly apart.

Do not use this set as proof that a model is safe. It covers direct single-turn requests in a few languages. It is also unsuitable for training a model to answer harmful requests, and for any commercial use.

## Contents

| Language | Code | Script | Harmful, held out | Harmful, repair | Benign, held out | Benign, repair |
|---|---|---|---:|---:|---:|---:|
| English | `en` | Latin | {{C_EN_HARM_HELDOUT}} | {{C_EN_HARM_REPAIR}} | {{C_EN_BEN_HELDOUT}} | {{C_EN_BEN_REPAIR}} |
| Hindi | `hi` | Devanagari | {{C_HI_HARM_HELDOUT}} | {{C_HI_HARM_REPAIR}} | {{C_HI_BEN_HELDOUT}} | {{C_HI_BEN_REPAIR}} |
| Bengali | `bn` | Bengali | {{C_BN_HARM_HELDOUT}} | {{C_BN_HARM_REPAIR}} | {{C_BN_BEN_HELDOUT}} | {{C_BN_BEN_REPAIR}} |
| Tamil | `ta` | Tamil | {{C_TA_HARM_HELDOUT}} | {{C_TA_HARM_REPAIR}} | {{C_TA_BEN_HELDOUT}} | {{C_TA_BEN_REPAIR}} |
| Hinglish | `hi-Latn` | Latin | {{C_HINGLISH_HARM_HELDOUT}} | {{C_HINGLISH_HARM_REPAIR}} | {{C_HINGLISH_BEN_HELDOUT}} | {{C_HINGLISH_BEN_REPAIR}} |

Total: {{N_PROMPTS_TOTAL}} prompts, with {{N_PROMPTS_HELDOUT}} held out and {{N_PROMPTS_REPAIR}} for repair. Each language is capped at 300 harmful and 250 benign prompts. {{SPLIT_COUNT_TABLE_NOTE}}

### Fields

| Field | Meaning |
|---|---|
| `id` | Unique per group and language: `<group>::<lang>` |
| `group` | The underlying item. All language versions of an item share a group |
| `lang` | Language code |
| `kind` | `harmful` or `benign` |
| `text` | The prompt |
| `source` | Source dataset |
| `source_id` | Row or file reference in the source |
| `category` | Harm category or XSTest prompt type, as given by the source |
| `origin` | `native`, `translated` or `code-mixed` |
| `split` | `heldout` or `repair` |
| `meta` | Source revision and hash, plus translator details on machine-translated rows |

### Labels

`kind` comes from the source dataset. We did not relabel any prompt. XSTest's safe prompts are `benign`, and its contrast prompts are `harmful`. All Aya red-teaming and MultiJail prompts are `harmful`.

## Sources and licences

| Source | Revision | Licence | What we took | Used for |
|---|---|---|---|---|
| [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming) | `5a16fee0` | Apache-2.0 | Human-written harmful prompts | English and Hindi, native |
| [DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) | `93d65c77` | MIT | Parallel harmful prompts | English and Bengali as published. English rows are also the seeds for Tamil and Hinglish |
| [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | `b71afe2a` | CC-BY-4.0 | The prompt split only. Its other splits hold model completions, which we did not use | English. Its safe prompts are the seeds for all benign translations |

XSTest is by Röttger et al. We loaded a copy of it, and the card above credits the original authors.

These were used in the project and are not part of this dataset:

- `CohereLabs/Global-MMLU-Lite` (Apache-2.0), for capability.
- The other MultiJail languages (Chinese, Italian, Vietnamese, Arabic, Korean, Thai, Swahili, Javanese), which we read directly from the source for the 10-language sweep.
- CircuitKIT's `jailbreak_binary.csv` (LSAL v1.2) and our translations of it, used for circuit discovery.

## How it was built

1. **Load.** Each source is pinned to a revision and hashed.
2. **Cap.** A deterministic per-language cap, salted separately from the split.
3. **Translate.** See below.
4. **Curate.** CuratorKIT cleans each language and removes near-duplicates with MinHash at 0.85. It runs per language, because parallel translations would otherwise count as duplicates. HTML stripping is off, because jailbreak prompts can contain tags. Rows removed: {{MINHASH_REMOVED}}.
5. **Split.** 60% held out and 40% repair, assigned per group, with seed 20261103. All language versions of one item land in the same split.
6. **Check for leakage.** Exact and near-duplicate text across splits is rejected (character 5-gram Jaccard at or above 0.8). This removed {{LEAKAGE_REMOVED_ROWS}} rows in {{LEAKAGE_REMOVED_GROUPS}} groups.

## Translation

### Method

| Item | Value |
|---|---|
| Translator | `CohereLabs/tiny-aya-global`, run locally |
| Temperature | 0 |
| Harmful rows translated | Tamil and Hinglish, from MultiJail English |
| Benign rows translated | Hindi, Bengali, Tamil and Hinglish, from XSTest's safe prompts |
| Instruction | Translate faithfully, keeping meaning, intent and tone, with no answer, refusal or comment |
| Hinglish | Requested as Hindi-English code-mixed text in Latin script, in everyday typing style |

Automatic checks reject a translation if it is empty, is a refusal, is commentary, has under 50% of its letters in the expected script, or repeats the source text. Rejected rows are dropped and never repaired by hand.

{{TRANSLATION_REQUESTED}} translations were requested and {{TRANSLATION_FAILED}} were rejected: {{TRANSLATION_FAILED_REASONS}}.

MultiJail's Bengali rows are the dataset authors' own translations from English. They are marked `translated`, and they were not produced by us.

### Quality status

**Status: {{TRANSLATION_QUALITY_STATUS}}.**

The automatic checks catch format and script failures. They do not measure meaning.

| Check | Result |
|---|---|
| Rows needing human review | {{TRANSLATION_TOTAL_ROWS}} |
| Rows reviewed | {{TRANSLATION_REVIEWED_ROWS}} |
| Rows rejected by reviewers | {{TRANSLATION_REJECTED_ROWS}} |
| Reviewers | {{TRANSLATION_REVIEWERS}} |

If status is `unreviewed`, treat every translated row as unverified. Any result on those rows is provisional.

## Known gaps

- **No native harmful prompts in Tamil or Hinglish.** Every such row is a machine translation. Hinglish rows imitate everyday typing. They did not come from real users.
- **The translator is a model under test.** Tiny Aya Global wrote the translations, and the project compares Global with Fire. The wording may suit Global better.
- **Dropped rows are not random.** The translator can refuse a prompt, and refused prompts are removed. Translated languages may therefore under-represent the most severe prompts.
- **Benign prompts are all English in origin.** XSTest depends on English wordplay and on context from English-speaking countries. A translated look-alike may no longer look alarming, so over-refusal may be understated outside English.
- **Sources differ by language.** Hindi comes from Aya red-teaming, Bengali from MultiJail, and English from a mix of Aya, MultiJail and XSTest contrast prompts. Comparing two languages also compares two prompt sources. Only the MultiJail rows are parallel.
- **Harm categories are not harmonised.** `category` keeps each source's own scheme.
- **No dialects.** Hindi is in Devanagari with one Hinglish style. There is no regional variation.
- **Single-turn, direct requests only.** There are no jailbreak wrappers, multi-turn attacks or tool use.
- **Possible contamination.** MultiJail and XSTest are public and may be in model training data.
- **Labels are inherited.** We did not audit source labels.
- **Earth languages.** {{EARTH_STATUS_SENTENCE}}

## Use

```python
from datasets import load_dataset

ds = load_dataset("{{DATASET_REPO_ID}}")
tamil_benign = ds["heldout"].filter(lambda r: r["lang"] == "ta" and r["kind"] == "benign")
```

Keep `heldout` for testing only. Tune on `repair`. Split by `group` if you make your own splits, or translations of a test prompt will leak.

## What is never included

- Model answers to any prompt in this set.
- Judge verdicts that quote model answers.
- Human review sheets. They contain model text.

## Licence

{{RELEASE_LICENCE_DECISION}}

- **This compilation: CC-BY-NC 4.0**, non-commercial. We chose this because the translated rows were produced with Tiny Aya, which is CC-BY-NC 4.0 with the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy). That policy bars generating synthetic data for commercial purposes.
- **Rows with `origin: native`, and MultiJail's own translated rows,** keep their source licences as well: Apache-2.0 (Aya red-teaming), MIT (MultiJail), CC-BY-4.0 (XSTest). `source` shows which applies.
- **Attribution.** Please cite the sources below when you use this set.

## Provenance

| Item | Value |
|---|---|
| Data manifest SHA-256 | {{DATA_MANIFEST_SHA256}} |
| Specification digest | {{SPEC_DIGEST}} |
| Code | {{REPO_URL}}, commit {{CODE_COMMIT}} |
| Curation | CuratorKIT `dd9e835` |

## Citations

- Aakanksha et al. (2024). *The Multilingual Alignment Prism*. arXiv:2406.18682.
- Deng, Y. et al. (2023). *Multilingual Jailbreak Challenges in Large Language Models*. arXiv:2310.06474.
- Röttger, P. et al. (2023). *XSTest*. arXiv:2308.01263.
- Salamanca, A. R. et al. (2026). *Tiny Aya: Bridging Scale and Multilingual Depth*. arXiv:2603.11510.

## Contact

{{CONTACT}}

## Placeholder list for the team

**Delete this section before publishing.** 47 placeholders, in order of first use.

| Placeholder | What goes here | Where it comes from |
|---|---|---|
| `{{DATASET_NAME}}` | Dataset repo name | Team |
| `{{ARTICLE_URL}}` | HF article link | After publishing |
| `{{COLLECTION_DATA_URL}}` | HF collection of data artifacts | After publishing |
| `{{CARD_DATE}}` | Date of this data build | `data/manifest.json` |
| `{{C_EN_HARM_HELDOUT}}` | English: harmful rows in heldout | `data/manifest.json` counts |
| `{{C_EN_HARM_REPAIR}}` | English: harmful rows in repair | `data/manifest.json` counts |
| `{{C_EN_BEN_HELDOUT}}` | English: benign rows in heldout | `data/manifest.json` counts |
| `{{C_EN_BEN_REPAIR}}` | English: benign rows in repair | `data/manifest.json` counts |
| `{{C_HI_HARM_HELDOUT}}` | Hindi: harmful rows in heldout | `data/manifest.json` counts |
| `{{C_HI_HARM_REPAIR}}` | Hindi: harmful rows in repair | `data/manifest.json` counts |
| `{{C_HI_BEN_HELDOUT}}` | Hindi: benign rows in heldout | `data/manifest.json` counts |
| `{{C_HI_BEN_REPAIR}}` | Hindi: benign rows in repair | `data/manifest.json` counts |
| `{{C_BN_HARM_HELDOUT}}` | Bengali: harmful rows in heldout | `data/manifest.json` counts |
| `{{C_BN_HARM_REPAIR}}` | Bengali: harmful rows in repair | `data/manifest.json` counts |
| `{{C_BN_BEN_HELDOUT}}` | Bengali: benign rows in heldout | `data/manifest.json` counts |
| `{{C_BN_BEN_REPAIR}}` | Bengali: benign rows in repair | `data/manifest.json` counts |
| `{{C_TA_HARM_HELDOUT}}` | Tamil: harmful rows in heldout | `data/manifest.json` counts |
| `{{C_TA_HARM_REPAIR}}` | Tamil: harmful rows in repair | `data/manifest.json` counts |
| `{{C_TA_BEN_HELDOUT}}` | Tamil: benign rows in heldout | `data/manifest.json` counts |
| `{{C_TA_BEN_REPAIR}}` | Tamil: benign rows in repair | `data/manifest.json` counts |
| `{{C_HINGLISH_HARM_HELDOUT}}` | Hinglish: harmful rows in heldout | `data/manifest.json` counts |
| `{{C_HINGLISH_HARM_REPAIR}}` | Hinglish: harmful rows in repair | `data/manifest.json` counts |
| `{{C_HINGLISH_BEN_HELDOUT}}` | Hinglish: benign rows in heldout | `data/manifest.json` counts |
| `{{C_HINGLISH_BEN_REPAIR}}` | Hinglish: benign rows in repair | `data/manifest.json` counts |
| `{{N_PROMPTS_TOTAL}}` | Total prompts. Mac build recorded 2,613 | `data/manifest.json` |
| `{{N_PROMPTS_HELDOUT}}` | Held-out prompts. Mac build recorded 1,580 | `data/manifest.json` |
| `{{N_PROMPTS_REPAIR}}` | Repair prompts. Mac build recorded 1,033 | `data/manifest.json` |
| `{{SPLIT_COUNT_TABLE_NOTE}}` | Shortfalls against the caps, from the coverage block | `data/manifest.json` |
| `{{MINHASH_REMOVED}}` | Rows removed by MinHash, all languages | `data/manifest.json` |
| `{{LEAKAGE_REMOVED_ROWS}}` | Rows removed by the leakage check | `data/manifest.json` |
| `{{LEAKAGE_REMOVED_GROUPS}}` | Groups removed by the leakage check | `data/manifest.json` |
| `{{TRANSLATION_REQUESTED}}` | Translation requests | `data/manifest.json` |
| `{{TRANSLATION_FAILED}}` | Requests rejected. Mac build recorded 134 | `data/manifest.json` |
| `{{TRANSLATION_FAILED_REASONS}}` | Count by rejection reason | `data/manifest.json` |
| `{{TRANSLATION_QUALITY_STATUS}}` | `human_reviewed`, `human_reviewed_with_rejections` or `unreviewed` | `translation_review.checked.json` |
| `{{TRANSLATION_TOTAL_ROWS}}` | Translated and code-mixed rows. Mac build recorded 1,766 | `data/manifest.json` |
| `{{TRANSLATION_REVIEWED_ROWS}}` | Rows reviewed by a person | `translation_review.checked.json` |
| `{{TRANSLATION_REJECTED_ROWS}}` | Rows rejected by reviewers | `translation_review.checked.json` |
| `{{TRANSLATION_REVIEWERS}}` | Number of reviewers per language. No names | `translation_review.checked.json` |
| `{{EARTH_STATUS_SENTENCE}}` | Whether Earth was run. Its licence was not accepted as of 5 Nov | Team |
| `{{DATASET_REPO_ID}}` | Evaluation dataset repo id | After publishing |
| `{{RELEASE_LICENCE_DECISION}}` | Confirm the licence choice after asking organisers | Team |
| `{{DATA_MANIFEST_SHA256}}` | Hash of the data manifest | `data/manifest.json` |
| `{{SPEC_DIGEST}}` | Spec digest | `report.json` |
| `{{REPO_URL}}` | Public GitHub repo. Private as of 5 Nov | After the team's go-ahead |
| `{{CODE_COMMIT}}` | Commit used for the run | `experiment.json` |
| `{{CONTACT}}` | Contact route. Organisers mask usernames for judges | Team |
