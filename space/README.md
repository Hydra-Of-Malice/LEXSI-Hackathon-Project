---
title: Same Model, Different Language
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Per-language safety of Tiny Aya Earth and its repairs
suggested_hardware: cpu-basic
datasets:
  - HydraOfMalice/tiny-aya-earth-langsafe-eval
models:
  - CohereLabs/tiny-aya-earth
  - CohereLabs/tiny-aya-global
  - HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5
  - HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5-gguf
  - HydraOfMalice/tiny-aya-earth-langsafe-resta-0p1
  - HydraOfMalice/tiny-aya-earth-langsafe-circuit-0p1
  - HydraOfMalice/tiny-aya-earth-langsafe-merge-back-0p5
  - HydraOfMalice/tiny-aya-earth-langsafe-steer-1p0
---

# Same model, different language, different safety

The demo for our Lexsi Labs AI Safety & Alignment Hackathon project (Track 2b).

## In plain words

Language models are usually tested for safety in English. We asked whether a small multilingual model turns down
harmful requests as reliably in other languages. We tested Cohere's **Tiny Aya Earth** against the general-purpose
**Tiny Aya Global**, on the same held-out prompts in seven languages: English as the control, Arabic and Swahili from
Earth's region, and Javanese, Bengali, Thai and Korean, where an early pilot found Earth weakest.

Then we tried to close the gap without retraining. Five repairs edit Earth's weights, or in one case its activations,
and three random-layer controls check whether the choice of layers matters. Every repair was tested again on prompts it
had not been tuned on.

This page is a results explorer. It shows rates, 95% intervals, sample sizes and paired tests, all read from one file of
aggregate numbers. **It never shows a model's answer, it has no text box, and it runs no model.**

**Every number is provisional.** Two local models judged the answers. Native readers checked blind samples in Thai and
Javanese only, outside the report, and the translated prompts are unreviewed. The banner at the top of the page lists what the report itself says is unvalidated.

## What the page shows

- **Safety by language.** Three headline findings, then a chart of one measure for Global, Earth and the selected
  repair, per language, with 95% intervals. You choose the measure (content-safe rate, over-refusal, capability,
  degenerate answers), the system preamble (default or removed) and the precision (FP16, Q4_K_M or NF4). Below it, the
  gap test for each language: Global minus Earth on the content-safe rate, paired on the same items, with a Bonferroni
  correction for the number of tests.
- **Repairs.** For any repair or control, its paired change against original Earth, per language, with 95% intervals,
  and its pre-registered acceptance gates. Then the release rule, checked for every model.
- **All numbers.** One row per model, preamble, precision and language, with every measure, the unsafe, invalid and
  disputed shares, and the partial and unclear shares kept apart.
- **About and links.** The published dataset, models and code, and what the page leaves out.

Charts carry a table of the same numbers, and hovering over a dot shows its exact values.

## The release rule

We publish a repair's weights only after held-out re-evaluation, and only if it is not significantly less safe than
original Earth in any language, in either preamble condition: the paired 95% interval of its change in content-safe
rate must not lie wholly below zero. The **Repairs** tab checks this for every model from the same aggregates, so the
page and the published set can be compared directly.

## Published artifacts

- Dataset (prompts, judge labels, aggregate reports, provenance; no model answers):
  [HydraOfMalice/tiny-aya-earth-langsafe-eval](https://huggingface.co/datasets/HydraOfMalice/tiny-aya-earth-langsafe-eval)
- Code: [Hydra-Of-Malice/LEXSI-Hackathon-Project](https://github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project)
- Models:
  - [circuit-merge-0p5](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5): the selected
    repair, FP16
  - [circuit-merge-0p5-gguf](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5-gguf): the
    selected repair, Q4_K_M
  - [resta-0p1](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-resta-0p1),
    [circuit-0p1](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-0p1) and
    [merge-back-0p5](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-merge-back-0p5): the other weight
    arms, FP16
  - [steer-1p0](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-steer-1p0): the steering arm's refusal
    direction, with no weights

The random-layer controls and the NF4 copy are not published. The page still shows their results.

## Where the numbers come from

`results.json` is built by `scripts/build_space_report.py` from the dataset's `reports/report.json`, which
`scripts/export_public.py` had already sanitised. Nobody types numbers in by hand. The script:

- refuses to run if that report does not match the hash in the export's manifest, or if the run's own `report.json`
  has changed since the export;
- keeps only per-language rates with their intervals and sample sizes, the gap tests, the paired comparisons, the
  acceptance gates and each arm's chosen knob, and stops if any value is anything other than a number, a flag or a
  short identifier;
- adds the acceptance limits from `configs/spec.yaml` only when that spec's digest matches the report's;
- records the source file's sha256, which the page shows under **About and links**.

From the project root:

```bash
.venv/Scripts/python.exe scripts/build_space_report.py            # rewrite space/results.json
.venv/Scripts/python.exe scripts/build_space_report.py --check    # exit 1 if it is out of date
```

## Why there is no live model or text box

Many evaluation prompts are harmful requests, and the models' answers to them can be harmful too. Cohere's Tiny Aya
models are CC-BY-NC 4.0 under Cohere Labs' acceptable use policy, and our own rule is that the demo must never show a
harmful completion. Aggregates alone cannot. A free CPU Space would also be slow and short of memory for two models.

The code keeps an optional **Side-by-side** tab for local use. It is off unless `LANGSAFE_ENABLE_LIVE=1` is set, and it
needs `llama-cpp-python`, which `requirements.txt` leaves out on purpose, so the Space cannot run it even if the
variable were set there. When it is on:

- it compares original Earth with the selected repair, both as Q4_K_M GGUF files, on the CPU;
- `prompts.json` offers six everyday benign prompts (photosynthesis, the monsoon, tea, a polite note to a teacher,
  fractions, saving water) in Earth's seven languages;
- the server receives a language code and a prompt ID and looks the text up itself, so text sent by a client never
  reaches a model; there is no text box, chat box or upload, events are not published as an API, the MCP server is off
  and uploads are refused.

The prompt list was reviewed on 11 November 2026. Six XSTest-style look-alikes were removed: they borrow violent verbs
on purpose, and their translations are unreviewed, so they were not clearly benign in every language. The Hindi and
Tamil strings, left from an earlier comparison with Tiny Aya Fire, were removed too.

## Run it locally

Use Python 3.12. From the project root:

```bash
uv venv --python 3.12 cache/space-venv
uv pip install --python cache/space-venv/Scripts/python.exe -r space/requirements.txt
cache/space-venv/Scripts/python.exe space/app.py        # bin/python on macOS and Linux
```

Then open <http://127.0.0.1:7860>. Set `GRADIO_SERVER_PORT` to use another port.

To try the side-by-side tab, install `space/requirements-live.txt` instead (add `--index-strategy unsafe-best-match`),
then set:

| Variable | Purpose |
|---|---|
| `LANGSAFE_ENABLE_LIVE` | `1` turns the tab on. Anything else leaves it off |
| `LANGSAFE_EARTH_ORIGINAL_GGUF`, `LANGSAFE_EARTH_REPAIRED_GGUF` | Paths to original Earth and the selected repair as Q4_K_M GGUF files. The repair is published as `tiny-aya-earth-langsafe-circuit-merge-0p5-gguf`; export Earth through `src/langsafe/quant/gguf.py` |
| `LANGSAFE_N_THREADS` | CPU threads per model. Defaults to the core count, at most 4 |
| `LANGSAFE_MAX_NEW_TOKENS` | Answer length, from 16 to 256. Defaults to 200 |

A plain llama.cpp conversion of a Tiny Aya checkpoint records the wrong stopping token and runs past the end of the
answer. The app pins the stopping token when it loads a file. On two CPU threads of a busy laptop, a new comparison of
two 200-token answers took about a minute once the models were loaded, and each loaded model used about 3.7 GB.

`LANGSAFE_REPORT_PATH` points the page at another results file or a bare `report.json`. If that file cannot be read, the
page shows an error and no numbers.

## Tests

```bash
uv pip install --python cache/space-venv/Scripts/python.exe pytest==9.2.0
cache/space-venv/Scripts/python.exe -m pytest space/test_app.py -q -p no:cacheprovider
```

The tests need no model files, GPU or network. They read the shipped `results.json`, so they also check it: only
numbers, flags and short identifiers, every published model present, and the release rule as the team recorded it.

## Translations need a native reader

Only the English prompts are source text. The others are unreviewed machine translations, and the side-by-side tab
says so beside each one. To clear a language, have a native reader check it against the guidance in `prompts.json`,
then set its `review_status` to `reviewed` and `reviewed_by` to their name. Review Javanese first.

## Licence

- **This Space** and the results it shows: CC-BY-NC 4.0. The models it reports on are Tiny Aya derivatives, CC-BY-NC
  4.0 under the Cohere Labs Acceptable Use Policy, so non-commercial use only.
- **Code:** the project's code, including this folder, is also available under the MIT licence in the GitHub repo.
