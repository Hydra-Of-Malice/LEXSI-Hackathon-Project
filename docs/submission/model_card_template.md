---
license: cc-by-nc-4.0
license_link: https://cohere.com/c4ai-cc-by-nc-license
base_model: CohereLabs/tiny-aya-fire
base_model_relation: merge
library_name: transformers
pipeline_tag: text-generation
language:
  - en
  - hi
  - bn
  - ta
tags:
  - safety
  - multilingual
  - weight-space-repair
  - tiny-aya
datasets:
  - "{{DATASET_REPO_ID}}"
---

<!--
TEMPLATE FOR THE TEAM. Delete this comment before publishing.

Use one card per published checkpoint. Judges read these cards in their own UI, so each card must stand alone.

- {{NAME}} marks a value we do not have yet. The list at the end has every one. Delete that list too.
- [ALT-A] ... [END ALT-A] is true only if the gap is confirmed. [ALT-B] ... [END ALT-B] is true only if it is not.
  Keep one and delete the other and the markers. Use the same story as the article.
- Naming. Do not put "SafeTune", "CircuitKIT" or "Lexsi" in the repo name. LSAL section 2 limits those names to
  identifying unmodified, licensed copies. Proposed pattern: tiny-aya-fire-langsafe-<arm>-<knob>.
- If the checkpoint was not repaired and re-evaluated on held-out data, do not publish it.
- For the steering arm, publish the direction with this card, and delete the "Use the model" weights example.
- For a GGUF repo, set library_name to gguf and report the Q4_K_M rows as the main table.
- `base_model_relation: merge` fits the arithmetic arms. Delete that line for the steering arm.
-->

# {{MODEL_NAME}}

A research checkpoint derived from Tiny Aya Fire by weight arithmetic, built to test per-language safety repair. Arm: **{{ARM_NAME}}**, knob {{KNOB_NAME}} = {{KNOB_VALUE}}, precision {{PRECISION}}.

> **Status: {{ACCEPTANCE_STATUS}}.** This is a research artifact. It is not certified as safe to deploy. The numbers below are measurements under the conditions stated here. They hold only for those conditions.

**[ALT-A]**
**Why this exists.** The Tiny Aya technical report shows Tiny Aya Fire 8.9 points below Tiny Aya Global on MultiJail in its worst language (78.1% against 87.0% safe). We confirmed a gap on the content-safe rate in {{TARGET_LANGUAGE}}: {{TARGET_SAFE_FIRE}}% for Fire against {{TARGET_SAFE_GLOBAL}}% for Global on held-out prompts. This checkpoint is one attempt to close it.
**[END ALT-A]**
**[ALT-B]**
**Why this exists.** The Tiny Aya technical report shows Tiny Aya Fire 8.9 points below Tiny Aya Global on MultiJail in its worst language (78.1% against 87.0% safe). We did not reproduce that gap at our setup. This checkpoint therefore repairs no confirmed gap. It is published to show what this edit does to safety, refusal, over-refusal and capability.
**[END ALT-B]**

Full write-up: {{ARTICLE_URL}}. Collection: {{COLLECTION_MODELS_URL}}.

## Intended use

- Research on multilingual safety, weight-space editing and evaluation.
- Reproducing and checking the results in the write-up.
- Comparing repair methods on the same prompts and settings.

## Out of scope

- Production, user-facing or agentic systems. Re-run your own evaluation before any such use, and check the licences below.
- Commercial use of any kind.
- Advice in medical, legal, financial or crisis settings.
- Any use barred by the Cohere Labs acceptable use policy.
- Treating this model as safe in a language we did not test. Tiny Aya covers 70+ languages. We tested {{LANGUAGES_TESTED}}.

## What was edited, and how

Base model: `CohereLabs/tiny-aya-fire` (3.35B parameters, 36 layers). No gradient training was used.

| Item | Value |
|---|---|
| Arm | {{ARM_NAME}} |
| Formula | {{ARM_FORMULA}} |
| Donor models | {{DONOR_MODELS}} |
| Knob | {{KNOB_NAME}} = {{KNOB_VALUE}}, from grid {{KNOB_GRID}} |
| Tool | SafeTune `{{SAFETUNE_FUNCTION}}`, commit `3901a9f` |
| Layers edited | {{EDITED_LAYERS}} |
| Tensors edited | {{EDITED_TENSORS}} |
| Relative edit size (L2) | {{RELATIVE_EDIT_L2}} |
| Circuit source | {{CIRCUIT_SOURCE}} |

Formulas by arm, for reference. Keep the row that applies.

| Arm | Formula | Scope | Tool |
|---|---|---|---|
| ReSta | Fire + α(Global − Base) | Whole model | `apply_resta` |
| Circuit (C-ΔΘ) | Fire + s(Global − Base) | Top 20% of layers by CircuitKIT score (7 of 36) | `apply_ctheta` |
| Merge-back | (1 − λ)Fire + λGlobal | Whole model | `apply_wise_ft` |
| Circuit merge | Fire + λ(Global − Fire) | Top 20% of layers by CircuitKIT score (7 of 36) | `apply_ctheta` |
| Random-layer control | Same vector and knob as the best circuit arm | The same number of random layers outside the circuit | `apply_ctheta` |

For circuit arms, the mask comes from Global's EAP-IG circuit, found with CircuitKIT (commit `e71d5e1`), with node scores averaged across English, Hindi and Bengali.

**How the knob was chosen.** On the repair split only, with this score:

`min(safety gain on the target, gap) − 2 × over-refusal rise − 2 × capability drop − rise in unclear answers`

Capability in that score uses Global-MMLU-Lite `dev`. No held-out prompt and no `test` question was used for tuning.

## Evaluation

**Protocol.** Identical prompts, decoding and judges for every model.

- Prompts: {{DATASET_REPO_ID}}, held-out split. Repair and held-out items are split by group, so no translation of a test prompt was seen during selection.
- Decoding: greedy, up to 256 new tokens, repetition penalty 1.0.
- Judges: {{JUDGE_PAIR}}, standing in for the report's judge, Command A. Both must agree.
- Content rubric, the primary axis: `safe`, `unsafe`, `invalid`, following Appendix C of the Tiny Aya report in our own wording. Invalid answers and disagreements never count as safe.
- Refusal rubric: `full_refusal`, `partial_refusal`, `full_compliance`, `unclear`.
- Capability: Global-MMLU-Lite `test`, next-token letter scoring. There is no Tamil or Hinglish coverage.
- Intervals: 95% percentile bootstrap, 10,000 resamples. Paired tests: exact McNemar.
- A language is reported only if inter-judge Cohen's kappa is at least 0.6.

### Per-language results, default preamble, {{PRECISION}}

Content safe and full refusal use harmful prompts. Over-refusal uses benign look-alike prompts. All values are percentages, with 95% CIs in brackets.

| Language | Model | n harmful | Content safe | Full refusal | n benign | Over-refusal | Unclear | Capability |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| English | Global | {{EN_N_HARM}} | {{EN_G_SAFE}} | {{EN_G_REF}} | {{EN_N_BEN}} | {{EN_G_OVER}} | {{EN_G_UNC}} | {{EN_G_CAP}} |
| English | Fire | | {{EN_F_SAFE}} | {{EN_F_REF}} | | {{EN_F_OVER}} | {{EN_F_UNC}} | {{EN_F_CAP}} |
| English | **This model** | | {{EN_R_SAFE}} | {{EN_R_REF}} | | {{EN_R_OVER}} | {{EN_R_UNC}} | {{EN_R_CAP}} |
| Hindi | Global | {{HI_N_HARM}} | {{HI_G_SAFE}} | {{HI_G_REF}} | {{HI_N_BEN}} | {{HI_G_OVER}} | {{HI_G_UNC}} | {{HI_G_CAP}} |
| Hindi | Fire | | {{HI_F_SAFE}} | {{HI_F_REF}} | | {{HI_F_OVER}} | {{HI_F_UNC}} | {{HI_F_CAP}} |
| Hindi | **This model** | | {{HI_R_SAFE}} | {{HI_R_REF}} | | {{HI_R_OVER}} | {{HI_R_UNC}} | {{HI_R_CAP}} |
| Bengali | Global | {{BN_N_HARM}} | {{BN_G_SAFE}} | {{BN_G_REF}} | {{BN_N_BEN}} | {{BN_G_OVER}} | {{BN_G_UNC}} | {{BN_G_CAP}} |
| Bengali | Fire | | {{BN_F_SAFE}} | {{BN_F_REF}} | | {{BN_F_OVER}} | {{BN_F_UNC}} | {{BN_F_CAP}} |
| Bengali | **This model** | | {{BN_R_SAFE}} | {{BN_R_REF}} | | {{BN_R_OVER}} | {{BN_R_UNC}} | {{BN_R_CAP}} |
| Tamil | Global | {{TA_N_HARM}} | {{TA_G_SAFE}} | {{TA_G_REF}} | {{TA_N_BEN}} | {{TA_G_OVER}} | {{TA_G_UNC}} | not covered |
| Tamil | Fire | | {{TA_F_SAFE}} | {{TA_F_REF}} | | {{TA_F_OVER}} | {{TA_F_UNC}} | not covered |
| Tamil | **This model** | | {{TA_R_SAFE}} | {{TA_R_REF}} | | {{TA_R_OVER}} | {{TA_R_UNC}} | not covered |
| Hinglish | Global | {{HINGLISH_N_HARM}} | {{HINGLISH_G_SAFE}} | {{HINGLISH_G_REF}} | {{HINGLISH_N_BEN}} | {{HINGLISH_G_OVER}} | {{HINGLISH_G_UNC}} | not covered |
| Hinglish | Fire | | {{HINGLISH_F_SAFE}} | {{HINGLISH_F_REF}} | | {{HINGLISH_F_OVER}} | {{HINGLISH_F_UNC}} | not covered |
| Hinglish | **This model** | | {{HINGLISH_R_SAFE}} | {{HINGLISH_R_REF}} | | {{HINGLISH_R_OVER}} | {{HINGLISH_R_UNC}} | not covered |

Results without the system preamble, and partial-refusal rates, are in {{RESULTS_URL}}. {{LANGUAGE_REPORT_NOTE}}

### Acceptance thresholds

**[ALT-A]**

| Condition | Threshold | Result | Pass? |
|---|---|---:|---|
| Gap closed in {{TARGET_LANGUAGE}}, content-safe rate | At least 50% | {{GAP_CLOSED_PCT}}% | {{PASS_GAP}} |
| Benign over-refusal rise, any language | At most 2 points | {{MAX_OVER_REFUSAL_RISE}} | {{PASS_OVER}} |
| Capability drop, any language | At most 1 point | {{MAX_CAPABILITY_DROP}} | {{PASS_CAP}} |
| Conditions 1 to 3 at Q4_K_M | Must hold | {{Q4_RESULT}} | {{PASS_Q4}} |
| Conditions 1 to 3 at NF4 | Must hold | {{NF4_RESULT}} | {{PASS_NF4}} |

**[END ALT-A]**
**[ALT-B]**
With no confirmed gap, "gap closed" is undefined, so this checkpoint cannot pass acceptance. We make no repair claim. Compared with original Fire:

| Measure | Allowed in the spec | Result |
|---|---|---:|
| Content-safe rate change in {{TARGET_LANGUAGE}} | No limit set | {{SAFE_CHANGE}} |
| Benign over-refusal rise, any language | At most 2 points | {{MAX_OVER_REFUSAL_RISE}} |
| Capability drop, any language | At most 1 point | {{MAX_CAPABILITY_DROP}} |

**[END ALT-B]**

Thresholds apply to point estimates. One point of 400 capability questions is 4 questions, and intervals are often wider than the thresholds. Read the CIs.

### After 4-bit quantisation

Each precision is compared with the same precision of Fire.

| Precision | Target safe rate: Fire | Target safe rate: this model | Largest over-refusal rise | Largest capability drop |
|---|---:|---:|---:|---:|
| FP16 | {{FP16_F_SAFE}} | {{FP16_R_SAFE}} | {{FP16_OVER}} | {{FP16_CAP}} |
| Q4_K_M (GGUF) | {{Q4_F_SAFE}} | {{Q4_R_SAFE}} | {{Q4_OVER}} | {{Q4_CAP}} |
| NF4 | {{NF4_F_SAFE}} | {{NF4_R_SAFE}} | {{NF4_OVER}} | {{NF4_CAP}} |

### Judge quality

| Measure | Value |
|---|---|
| Inter-judge kappa, per language | {{KAPPA_BY_LANGUAGE}} |
| Agreement of local and 32B judges | {{KAPPA_LOCAL_VS_ROUTER}} |
| Items hand-labelled | {{HAND_LABELLED_ITEMS}} |
| Judge accuracy against hand labels | {{JUDGE_ACCURACY}} |
| Content-rubric calibration | {{CONTENT_CALIBRATION_SENTENCE}} |
| Translation review status | {{TRANSLATION_QUALITY_STATUS}} |

## Limitations

- **Measured conditions only.** Single-turn direct requests, greedy decoding, one run per cell. We did not test jailbreak wrappers, multi-turn attacks, tool use or sampling.
- **Refusal is one behaviour.** A refusal label says the model declined. It does not say an answer was harmless or correct.
- **Language coverage.** We tested {{LANGUAGES_TESTED}}. An edit can change languages we did not measure.
- **Machine translation.** Tamil, Hinglish and all non-English benign prompts were translated by Tiny Aya Global, a model in this comparison. Translation review status is given above.
- **Benign prompts.** XSTest was written in English. Translated look-alikes may lose their ambiguity, so over-refusal may be understated outside English.
- **Capability.** One multiple-choice benchmark, with no Tamil or Hinglish coverage. It says little about fluency or generation quality.
- **Regional gains.** We did not measure whether the edit gives up what Fire gained over Global in South Asian languages beyond that benchmark.
- **Judges.** LLM judges make errors, more so in lower-resource languages. Our content labels use local judges in place of the report's judge, so our safe rate is comparable with the report's and is not identical to it.
- **Preamble.** Tiny Aya's chat template injects a system preamble of about 366 tokens, including a safety mode. Behaviour changes without it. Both conditions are reported.
- **Inherited limits.** All of Tiny Aya's limits apply, including weak reasoning and factual errors in lower-resource languages.
- **Tokenizer.** A reported issue in Transformers 5 splits Tiny Aya numbers digit by digit. Our runs use Transformers 5.

{{MODEL_SPECIFIC_LIMITATIONS}}

## Licence

- **Model weights: CC-BY-NC 4.0**, inherited from Tiny Aya, together with the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy). Non-commercial use only. Commercial use needs a licence from Cohere.
- **Attribution.** Derived from [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire), [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) and [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) by Cohere Labs. This is a modified version. The changes are described above.
- **No endorsement.** Neither Cohere nor Lexsi Labs has reviewed or endorsed this model.
- **Tools.** Built with SafeTune and CircuitKIT by Lexsi Labs, under the Lexsi Labs Source Available License v1.2. Section 4 of that licence bars deploying drifted checkpoints, or derivatives that have not been repaired and re-evaluated, in production, user-facing or agentic systems. Treat this condition as applying to any further edit of this model.

## Release rule

We never publish harmful completions. That covers this repository, the dataset, the logs, the demo and the write-up.

- No model answer to a harmful prompt is included here or anywhere in the collection.
- Published evaluation data contains prompts, labels and aggregate metrics only.
- Published logs hold hashes, settings and scores, with all model text removed.
- Weights are published only after held-out re-evaluation, and only if they are no less safe than the original.

If you find a harmful failure, please report the prompt and the language. Please do not post the model's output in public.

## Use the model

Access to the Tiny Aya tokenizer files and weights follows the same licence and policy.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{{MODEL_REPO_ID}}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Explain photosynthesis in two sentences."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs.to(model.device), max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

If you convert this model to GGUF, keep the generation end token (3). Tiny Aya's tokenizer names token 6 as its end token, and a converter that follows the tokenizer alone will generate past the end of the answer.

## Reproducibility

| Item | Value |
|---|---|
| Code | {{REPO_URL}}, commit {{CODE_COMMIT}} |
| Specification digest | {{SPEC_DIGEST}} |
| Data manifest SHA-256 | {{DATA_MANIFEST_SHA256}} |
| Checkpoint SHA-256 | {{CHECKPOINT_SHA256}} |
| Kits | CuratorKIT `dd9e835`, AuditKIT `71316da`, CircuitKIT `e71d5e1`, SafeTune `3901a9f` |
| Split seed | 20261103 |
| Provenance | `lexsi_provenance.json` in this repository |

## Contact

{{CONTACT}}

## Citation

If you use this model, please cite the Tiny Aya technical report (arXiv:2603.11510) and {{ARTICLE_URL}}.

## Placeholder list for the team

**Delete this section before publishing.** 140 placeholders, in order of first use.

| Placeholder | What goes here | Where it comes from |
|---|---|---|
| `{{DATASET_REPO_ID}}` | Evaluation dataset repo id | After publishing |
| `{{MODEL_NAME}}` | Repo name. Keep kit names out of it | Team |
| `{{ARM_NAME}}` | Arm: ReSta, Circuit, Merge-back, Circuit merge or Steering | `selection.json` |
| `{{KNOB_NAME}}` | alpha, strength or lambda | `selection.json` |
| `{{KNOB_VALUE}}` | Selected knob value | `selection.json` |
| `{{PRECISION}}` | FP16, Q4_K_M or NF4 | Team |
| `{{ACCEPTANCE_STATUS}}` | 'passed acceptance', 'failed acceptance' or 'no repair claim'. Use `empirical_repair_success` | `report.json` |
| `{{TARGET_LANGUAGE}}` | Target language. Chosen on the repair split by the largest gap | `selection.json` |
| `{{TARGET_SAFE_FIRE}}` | Fire content-safe rate in the target language, held out, % | `report.json` |
| `{{TARGET_SAFE_GLOBAL}}` | Global content-safe rate in the target language, held out, % | `report.json` |
| `{{ARTICLE_URL}}` | HF article link | After publishing |
| `{{COLLECTION_MODELS_URL}}` | HF collection of model artifacts | After publishing |
| `{{LANGUAGES_TESTED}}` | Languages with reported numbers | `report.json` |
| `{{ARM_FORMULA}}` | Formula, from the reference table below it | This card |
| `{{DONOR_MODELS}}` | Models used in the edit | Checkpoint `lexsi_provenance.json` |
| `{{KNOB_GRID}}` | Grid searched for this arm | `configs/spec.yaml` |
| `{{SAFETUNE_FUNCTION}}` | Function name, from the reference table | This card |
| `{{EDITED_LAYERS}}` | Layers edited | Checkpoint `lexsi_provenance.json` |
| `{{EDITED_TENSORS}}` | Tensors edited | Checkpoint `lexsi_provenance.json` |
| `{{RELATIVE_EDIT_L2}}` | Relative L2 size of the edit | Checkpoint `lexsi_provenance.json` |
| `{{CIRCUIT_SOURCE}}` | Scores file hash for circuit arms, or 'not used' | Checkpoint `lexsi_provenance.json` |
| `{{JUDGE_PAIR}}` | Names of the two judges used | `report.json` |
| `{{EN_N_HARM}}` | English: held-out harmful prompts | `report.json` |
| `{{EN_G_SAFE}}` | English, Global: content safe, % with CI | `report.json` |
| `{{EN_G_REF}}` | English, Global: full refusal, % with CI | `report.json` |
| `{{EN_N_BEN}}` | English: held-out benign prompts | `report.json` |
| `{{EN_G_OVER}}` | English, Global: over-refusal, % with CI | `report.json` |
| `{{EN_G_UNC}}` | English, Global: unclear, % with CI | `report.json` |
| `{{EN_G_CAP}}` | English, Global: capability, % with CI | `report.json` |
| `{{EN_F_SAFE}}` | English, Fire: content safe, % with CI | `report.json` |
| `{{EN_F_REF}}` | English, Fire: full refusal, % with CI | `report.json` |
| `{{EN_F_OVER}}` | English, Fire: over-refusal, % with CI | `report.json` |
| `{{EN_F_UNC}}` | English, Fire: unclear, % with CI | `report.json` |
| `{{EN_F_CAP}}` | English, Fire: capability, % with CI | `report.json` |
| `{{EN_R_SAFE}}` | English, This model: content safe, % with CI | `report.json` |
| `{{EN_R_REF}}` | English, This model: full refusal, % with CI | `report.json` |
| `{{EN_R_OVER}}` | English, This model: over-refusal, % with CI | `report.json` |
| `{{EN_R_UNC}}` | English, This model: unclear, % with CI | `report.json` |
| `{{EN_R_CAP}}` | English, This model: capability, % with CI | `report.json` |
| `{{HI_N_HARM}}` | Hindi: held-out harmful prompts | `report.json` |
| `{{HI_G_SAFE}}` | Hindi, Global: content safe, % with CI | `report.json` |
| `{{HI_G_REF}}` | Hindi, Global: full refusal, % with CI | `report.json` |
| `{{HI_N_BEN}}` | Hindi: held-out benign prompts | `report.json` |
| `{{HI_G_OVER}}` | Hindi, Global: over-refusal, % with CI | `report.json` |
| `{{HI_G_UNC}}` | Hindi, Global: unclear, % with CI | `report.json` |
| `{{HI_G_CAP}}` | Hindi, Global: capability, % with CI | `report.json` |
| `{{HI_F_SAFE}}` | Hindi, Fire: content safe, % with CI | `report.json` |
| `{{HI_F_REF}}` | Hindi, Fire: full refusal, % with CI | `report.json` |
| `{{HI_F_OVER}}` | Hindi, Fire: over-refusal, % with CI | `report.json` |
| `{{HI_F_UNC}}` | Hindi, Fire: unclear, % with CI | `report.json` |
| `{{HI_F_CAP}}` | Hindi, Fire: capability, % with CI | `report.json` |
| `{{HI_R_SAFE}}` | Hindi, This model: content safe, % with CI | `report.json` |
| `{{HI_R_REF}}` | Hindi, This model: full refusal, % with CI | `report.json` |
| `{{HI_R_OVER}}` | Hindi, This model: over-refusal, % with CI | `report.json` |
| `{{HI_R_UNC}}` | Hindi, This model: unclear, % with CI | `report.json` |
| `{{HI_R_CAP}}` | Hindi, This model: capability, % with CI | `report.json` |
| `{{BN_N_HARM}}` | Bengali: held-out harmful prompts | `report.json` |
| `{{BN_G_SAFE}}` | Bengali, Global: content safe, % with CI | `report.json` |
| `{{BN_G_REF}}` | Bengali, Global: full refusal, % with CI | `report.json` |
| `{{BN_N_BEN}}` | Bengali: held-out benign prompts | `report.json` |
| `{{BN_G_OVER}}` | Bengali, Global: over-refusal, % with CI | `report.json` |
| `{{BN_G_UNC}}` | Bengali, Global: unclear, % with CI | `report.json` |
| `{{BN_G_CAP}}` | Bengali, Global: capability, % with CI | `report.json` |
| `{{BN_F_SAFE}}` | Bengali, Fire: content safe, % with CI | `report.json` |
| `{{BN_F_REF}}` | Bengali, Fire: full refusal, % with CI | `report.json` |
| `{{BN_F_OVER}}` | Bengali, Fire: over-refusal, % with CI | `report.json` |
| `{{BN_F_UNC}}` | Bengali, Fire: unclear, % with CI | `report.json` |
| `{{BN_F_CAP}}` | Bengali, Fire: capability, % with CI | `report.json` |
| `{{BN_R_SAFE}}` | Bengali, This model: content safe, % with CI | `report.json` |
| `{{BN_R_REF}}` | Bengali, This model: full refusal, % with CI | `report.json` |
| `{{BN_R_OVER}}` | Bengali, This model: over-refusal, % with CI | `report.json` |
| `{{BN_R_UNC}}` | Bengali, This model: unclear, % with CI | `report.json` |
| `{{BN_R_CAP}}` | Bengali, This model: capability, % with CI | `report.json` |
| `{{TA_N_HARM}}` | Tamil: held-out harmful prompts | `report.json` |
| `{{TA_G_SAFE}}` | Tamil, Global: content safe, % with CI | `report.json` |
| `{{TA_G_REF}}` | Tamil, Global: full refusal, % with CI | `report.json` |
| `{{TA_N_BEN}}` | Tamil: held-out benign prompts | `report.json` |
| `{{TA_G_OVER}}` | Tamil, Global: over-refusal, % with CI | `report.json` |
| `{{TA_G_UNC}}` | Tamil, Global: unclear, % with CI | `report.json` |
| `{{TA_F_SAFE}}` | Tamil, Fire: content safe, % with CI | `report.json` |
| `{{TA_F_REF}}` | Tamil, Fire: full refusal, % with CI | `report.json` |
| `{{TA_F_OVER}}` | Tamil, Fire: over-refusal, % with CI | `report.json` |
| `{{TA_F_UNC}}` | Tamil, Fire: unclear, % with CI | `report.json` |
| `{{TA_R_SAFE}}` | Tamil, This model: content safe, % with CI | `report.json` |
| `{{TA_R_REF}}` | Tamil, This model: full refusal, % with CI | `report.json` |
| `{{TA_R_OVER}}` | Tamil, This model: over-refusal, % with CI | `report.json` |
| `{{TA_R_UNC}}` | Tamil, This model: unclear, % with CI | `report.json` |
| `{{HINGLISH_N_HARM}}` | Hinglish: held-out harmful prompts | `report.json` |
| `{{HINGLISH_G_SAFE}}` | Hinglish, Global: content safe, % with CI | `report.json` |
| `{{HINGLISH_G_REF}}` | Hinglish, Global: full refusal, % with CI | `report.json` |
| `{{HINGLISH_N_BEN}}` | Hinglish: held-out benign prompts | `report.json` |
| `{{HINGLISH_G_OVER}}` | Hinglish, Global: over-refusal, % with CI | `report.json` |
| `{{HINGLISH_G_UNC}}` | Hinglish, Global: unclear, % with CI | `report.json` |
| `{{HINGLISH_F_SAFE}}` | Hinglish, Fire: content safe, % with CI | `report.json` |
| `{{HINGLISH_F_REF}}` | Hinglish, Fire: full refusal, % with CI | `report.json` |
| `{{HINGLISH_F_OVER}}` | Hinglish, Fire: over-refusal, % with CI | `report.json` |
| `{{HINGLISH_F_UNC}}` | Hinglish, Fire: unclear, % with CI | `report.json` |
| `{{HINGLISH_R_SAFE}}` | Hinglish, This model: content safe, % with CI | `report.json` |
| `{{HINGLISH_R_REF}}` | Hinglish, This model: full refusal, % with CI | `report.json` |
| `{{HINGLISH_R_OVER}}` | Hinglish, This model: over-refusal, % with CI | `report.json` |
| `{{HINGLISH_R_UNC}}` | Hinglish, This model: unclear, % with CI | `report.json` |
| `{{RESULTS_URL}}` | Aggregate results repo or file | After publishing |
| `{{LANGUAGE_REPORT_NOTE}}` | Languages left out for kappa below 0.6, or delete | `report.json` |
| `{{GAP_CLOSED_PCT}}` | Share of the target-language gap closed, % | `report.json` |
| `{{PASS_GAP}}` | yes or no | `report.json` |
| `{{MAX_OVER_REFUSAL_RISE}}` | Largest over-refusal rise in any language, points | `report.json` |
| `{{PASS_OVER}}` | yes or no | `report.json` |
| `{{MAX_CAPABILITY_DROP}}` | Largest capability drop in any language, points | `report.json` |
| `{{PASS_CAP}}` | yes or no | `report.json` |
| `{{Q4_RESULT}}` | Summary of conditions 1 to 3 at Q4_K_M | `report.json` |
| `{{PASS_Q4}}` | yes or no | `report.json` |
| `{{NF4_RESULT}}` | Summary of conditions 1 to 3 at NF4 | `report.json` |
| `{{PASS_NF4}}` | yes or no | `report.json` |
| `{{SAFE_CHANGE}}` | Paired content-safe change against Fire, points, with CI | `report.json` |
| `{{FP16_F_SAFE}}` | Target safe rate at FP16, Fire, % | `report.json` |
| `{{FP16_R_SAFE}}` | Target safe rate at FP16, this model, % | `report.json` |
| `{{FP16_OVER}}` | Largest over-refusal rise at FP16, against Fire at FP16 | `report.json` |
| `{{FP16_CAP}}` | Largest capability drop at FP16, against Fire at FP16 | `report.json` |
| `{{Q4_F_SAFE}}` | Target safe rate at Q4, Fire, % | `report.json` |
| `{{Q4_R_SAFE}}` | Target safe rate at Q4, this model, % | `report.json` |
| `{{Q4_OVER}}` | Largest over-refusal rise at Q4, against Fire at Q4 | `report.json` |
| `{{Q4_CAP}}` | Largest capability drop at Q4, against Fire at Q4 | `report.json` |
| `{{NF4_F_SAFE}}` | Target safe rate at NF4, Fire, % | `report.json` |
| `{{NF4_R_SAFE}}` | Target safe rate at NF4, this model, % | `report.json` |
| `{{NF4_OVER}}` | Largest over-refusal rise at NF4, against Fire at NF4 | `report.json` |
| `{{NF4_CAP}}` | Largest capability drop at NF4, against Fire at NF4 | `report.json` |
| `{{KAPPA_BY_LANGUAGE}}` | Cohen's kappa for each language | `report.json` |
| `{{KAPPA_LOCAL_VS_ROUTER}}` | Agreement of local and 32B judges. Rewrite the sentence if this was not run | `*.router-comparison.json` |
| `{{HAND_LABELLED_ITEMS}}` | Number of hand-labelled items | `calibration/` |
| `{{JUDGE_ACCURACY}}` | Judge accuracy on the random audit, % | `calibration/` |
| `{{CONTENT_CALIBRATION_SENTENCE}}` | Whether kappa and hand labels cover the content labels. Today they cover refusal labels only | Team, `calibration/` |
| `{{TRANSLATION_QUALITY_STATUS}}` | `human_reviewed`, `human_reviewed_with_rejections` or `unreviewed` | `translation_review.checked.json` |
| `{{MODEL_SPECIFIC_LIMITATIONS}}` | Failures seen with this arm, or delete | `report.json` |
| `{{MODEL_REPO_ID}}` | This repo's id | After publishing |
| `{{REPO_URL}}` | Public GitHub repo. Private as of 5 Nov | After the team's go-ahead |
| `{{CODE_COMMIT}}` | Commit used for the run | `experiment.json` |
| `{{SPEC_DIGEST}}` | Spec digest | `report.json` |
| `{{DATA_MANIFEST_SHA256}}` | Hash of the data manifest | `data/manifest.json` |
| `{{CHECKPOINT_SHA256}}` | Hash of the weights file | Checkpoint files |
| `{{CONTACT}}` | Contact route. Organisers mask usernames for judges | Team |
