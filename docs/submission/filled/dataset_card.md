---
license: cc-by-nc-4.0
language:
  - en
  - ar
  - sw
  - jv
  - bn
  - th
  - ko
pretty_name: "{{DATASET_NAME}}"
task_categories:
  - text-generation
tags:
  - safety
  - red-teaming
  - refusal
  - over-refusal
  - multilingual
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
- Fill counts from the Earth build's data/manifest.json. The Mac build's figures (2,613 prompts, 1,766 translated or
  code-mixed rows) were for Fire's languages and do not apply.
- Confirm in the manifest that no harmful row was translated. MultiJail and Aya red-teaming should fill every language to
  the cap of 300. If any harmful row was translated, change the Translation section and the Known gaps.
- Check that the manifest's `curatorkit` field is filled. The pilot manifests have it null, because pilots skip CuratorKIT.
- scripts/export_public.py still lists CARD_LANGUAGES without jv, th and ko, so it refuses those rows. Fix that first.
- Before uploading, confirm the files hold prompts and labels only. Never upload anything from runs/gens, runs/judged,
  runs/hand_check, runs/auditkit or the review CSVs.
-->

# {{DATASET_NAME}}

> **Content warning.** This dataset contains harmful requests, including requests about violence, self-harm, hate and crime. They exist to test whether models refuse. Do not use them to train a model to comply.

A multilingual prompt set for measuring content safety, refusal and benign over-refusal in small language models, one language at a time. It was built to test Tiny Aya Earth against Tiny Aya Global.

**It contains prompts and labels only. It contains no model answers.** Machine-translated rows are model-generated translations of public prompts. They are prompts, and they are marked as such. No answer from any model to any prompt is included.

Write-up: {{ARTICLE_URL}}. Collection: https://huggingface.co/collections/HydraOfMalice/same-model-different-language-evaluation-data-6af426d23fff746c25ca1e1e. Date of this build: 2026-11-07.

## What it is for

- Comparing safety and refusal behaviour across languages, on identical settings.
- Testing whether a safety edit improves answers to harmful prompts without raising refusals on harmless look-alikes.
- Keeping repair data and test data strictly apart.

Do not use this set as proof that a model is safe. It covers direct single-turn requests in a few languages. It is also unsuitable for training a model to answer harmful requests, and for any commercial use.

## Contents

| Language | Code | Script | Harmful, held out | Harmful, repair | Benign, held out | Benign, repair |
|---|---|---|---:|---:|---:|---:|
| Arabic | `ar` | Arabic | 170 | 127 | 152 | 95 |
| Swahili | `sw` | Latin | 176 | 122 | 146 | 85 |
| Javanese | `jv` | Latin | 176 | 123 | 147 | 92 |
| Bengali | `bn` | Bengali | 176 | 123 | 149 | 93 |
| Thai | `th` | Thai | 176 | 123 | 142 | 84 |
| Korean | `ko` | Hangul | 176 | 123 | 147 | 89 |
| English | `en` | Latin | 178 | 119 | 153 | 95 |

Total: 3,757 prompts, with 2,264 held out and 1,493 for repair. Each language is capped at 300 harmful and 250 benign prompts. {{SPLIT_COUNT_TABLE_NOTE}}

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
| `origin` | `native` or `translated` |
| `split` | `heldout` or `repair` |
| `meta` | Source revision and hash, plus translator details on machine-translated rows |

### Labels

`kind` comes from the source dataset. We did not relabel any prompt. XSTest's safe prompts are `benign`, and its contrast prompts are `harmful`. All Aya red-teaming and MultiJail prompts are `harmful`.

### Circuits

`circuits/` holds the layer-level release of the refusal circuits behind the circuit repairs: each circuit's layer list,
per-layer score mass and summary metrics, the editing mask the repairs used, and the settings of every
circuit-restricted checkpoint. Node-level score files are not included. Researchers can ask us for them, and the SHA-256
values in `circuits/` identify the exact files. `circuits/README.md` explains each file.

## Sources and licences

| Source | Revision | Licence | What we took | Used for |
|---|---|---|---|---|
| [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming) | `5a16fee0` | Apache-2.0 | Human-written harmful prompts | English and Arabic, native |
| [DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) | `93d65c77` | MIT | Parallel harmful prompts | English, Arabic, Swahili, Javanese, Bengali, Thai and Korean, as published |
| [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | `b71afe2a` | CC-BY-4.0 | The prompt split only. Its other splits hold model completions, which we did not use | English. Its safe prompts are the seeds for all benign translations |

XSTest is by Röttger et al. We loaded a copy of it, and the card above credits the original authors.

These were used in the project and are not part of this dataset:

- `CohereLabs/Global-MMLU-Lite` (Apache-2.0), for capability.
- The other MultiJail languages (Chinese, Italian, Vietnamese), which we read directly from the source for the 10-language pilot.
- CircuitKIT's `jailbreak_binary.csv` (LSAL v1.2) and our Javanese and Bengali translations of it, used for circuit discovery.
- The first plan's prompts in Hindi, Tamil and Hinglish, built for Tiny Aya Fire.

## How it was built

1. **Load.** Each source is pinned to a revision and hashed.
2. **Cap.** A deterministic per-language cap, salted separately from the split.
3. **Translate.** See below.
4. **Curate.** CuratorKIT cleans each language and removes near-duplicates with MinHash at 0.85. It runs per language, because parallel translations would otherwise count as duplicates. HTML stripping is off, because jailbreak prompts can contain tags. Rows removed: 1.
5. **Split.** 60% held out and 40% repair, assigned per group, with seed 20261103. All language versions of one item land in the same split.
6. **Check for leakage.** Exact and near-duplicate text across splits is rejected (character 5-gram Jaccard at or above 0.8). This removed 21 rows in 4 groups.

## Translation

### Method

| Item | Value |
|---|---|
| Translator | `CohereLabs/tiny-aya-global`, run locally |
| Temperature | 0 |
| Harmful rows translated | None. MultiJail's own versions and Aya red-teaming fill every language to the cap |
| Benign rows translated | Arabic, Swahili, Javanese, Bengali, Thai and Korean, from XSTest's safe prompts |
| Instruction | Translate faithfully, keeping meaning, intent and tone, with no answer, refusal or comment |
| Styles requested | Modern Standard Arabic, standard Swahili, Javanese in Latin script, Bengali script, Thai script, Korean in Hangul |

The translator often wrapped its output in a label and a word-by-word explanation, so the translation is first taken out of the wrapper. Automatic checks then reject a translation if it is empty, is a refusal, is commentary, has under 50% of its letters in the expected script, repeats the source text, runs over three times its source's length, loops, is a list, keeps markdown, holds a broken character, or loses its question mark (not applied to Thai). Rejected rows are dropped and never repaired by hand. For Swahili and Javanese, the script check cannot tell the language from another Latin-script language such as Indonesian. For Javanese, Thai and Korean, the refusal check knows only English phrases.

1500 translations were requested and 68 were rejected: {{TRANSLATION_FAILED_REASONS}}.

MultiJail's non-English rows are the dataset authors' own translations from English. They are marked `translated`, and they were not produced by us.

### Quality status

**Status: unreviewed.**

The automatic checks catch format and script failures. They do not measure meaning.

| Check | Result |
|---|---|
| Rows needing human review | 2,994 |
| Rows reviewed | 0 |
| Rows rejected by reviewers | 0 |
| Reviewers | none yet |

If status is `unreviewed`, treat every translated row as unverified. Any result on those rows is provisional.

## Known gaps

- **No native harmful prompts in Swahili, Javanese, Bengali, Thai or Korean.** Those rows are MultiJail's translations from English. Only English and Arabic have prompts written natively, from Aya red-teaming.
- **The translator is a model under test.** Tiny Aya Global wrote the benign translations, and the project compares Global with Earth. The wording may suit Global better.
- **Dropped rows are not random.** The translator can refuse a prompt or answer it instead, and both are removed. Translated benign sets may therefore under-represent the most alarming-looking prompts, and the plainest factual ones.
- **Benign prompts are all English in origin.** XSTest depends on English wordplay and on context from English-speaking countries. A translated look-alike may no longer look alarming, so over-refusal may be understated outside English.
- **Sources differ by language.** Swahili, Javanese, Bengali, Thai and Korean come from MultiJail alone. Arabic mixes Aya red-teaming and MultiJail. English mixes Aya, MultiJail and XSTest contrast prompts. Comparing two languages can also compare two prompt sources. Only the MultiJail rows are parallel.
- **No harm categories.** Each source has its own category scheme, and the export leaves them out: some category labels share wording with answers the models gave, and the release check blocks any such overlap. Join on `source` and `source_id` to recover them from the source.
- **No dialects.** Translated Arabic is Modern Standard Arabic, and translated Javanese is in Latin script. There is no regional variation.
- **Single-turn, direct requests only.** There are no jailbreak wrappers, multi-turn attacks or tool use.
- **Possible contamination.** MultiJail and XSTest are public and may be in model training data.
- **Labels are inherited.** We did not audit source labels.

## Use

```python
from datasets import load_dataset

ds = load_dataset("{{DATASET_REPO_ID}}")
javanese_benign = ds["heldout"].filter(lambda r: r["lang"] == "jv" and r["kind"] == "benign")
```

Keep `heldout` for testing only. Tune on `repair`. Split by `group` if you make your own splits, or translations of a test prompt will leak.

## What is never included

- Model answers to any prompt in this set.
- Judge verdicts that quote model answers.
- Human review sheets. They contain model text.
- Node-level circuit scores. They are available to researchers on request.

## Licence

{{RELEASE_LICENCE_DECISION}}

- **This compilation: CC-BY-NC 4.0**, non-commercial. We chose this because the translated rows were produced with Tiny Aya, which is CC-BY-NC 4.0 with the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy). That policy bars generating synthetic data for commercial purposes.
- **Rows with `origin: native`, and MultiJail's own translated rows,** keep their source licences as well: Apache-2.0 (Aya red-teaming), MIT (MultiJail), CC-BY-4.0 (XSTest). `source` shows which applies.
- **Attribution.** Please cite the sources below when you use this set.

## Provenance

| Item | Value |
|---|---|
| Data manifest SHA-256 | 389b0882322ae47517dcec9bd428986e61a4e5f005937634f823b9310e37368f |
| Specification digest | bdabb7edab9eead8 |
| Code | https://github.com/Hydra-Of-Malice/langsafe, commit {{CODE_COMMIT}} |
| Curation | CuratorKIT `dd9e835` |

## Citations

- Aakanksha et al. (2024). *The Multilingual Alignment Prism*. arXiv:2406.18682.
- Deng, Y. et al. (2023). *Multilingual Jailbreak Challenges in Large Language Models*. arXiv:2310.06474.
- Röttger, P. et al. (2023). *XSTest*. arXiv:2308.01263.
- Salamanca, A. R. et al. (2026). *Tiny Aya: Bridging Scale and Multilingual Depth*. arXiv:2603.11510.

## Contact

{{CONTACT}}
