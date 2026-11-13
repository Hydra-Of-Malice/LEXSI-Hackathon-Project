# Same model, different language, different safety

**Measuring per-language safety in Tiny Aya Earth, and testing weight-space repairs with the Lexsi stack.**

Lexsi Labs AI Safety & Alignment Hackathon · Track 2b, *repairing safety drift without retraining*

**[Article](https://huggingface.co/blog/HydraOfMalice/same-model-different-language-different-safety)** ·
**[Models](https://huggingface.co/collections/HydraOfMalice/same-model-different-language-repair-checkpoints-6af426d1cac578d29e4805e5)** ·
**[Data](https://huggingface.co/collections/HydraOfMalice/same-model-different-language-evaluation-data-6af426d23fff746c25ca1e1e)** ·
**[Reproduction notebook](notebooks/01_reproduce_results.ipynb)** ·
**[Abstract](docs/submission/abstract.md)**

---

## The question

The Tiny Aya technical report puts Tiny Aya Earth 9.5 points below Tiny Aya Global on MultiJail in its worst
language, but does not say which language. We asked two things:

1. Does that gap hold when measured language by language, with full-precision weights, on held-out prompts?
2. Can the Lexsi stack repair it without retraining, and without breaking anything else?

## What we found

- **One confirmed gap: Javanese, with the safety preamble.** It is almost all broken answers, not harmful content.
  Both judges called 1 of 176 answers unsafe for each model.
- **Thai, the language the repairs targeted, was not confirmed.** Earth was 89.2% safe and Global 94.9%
  (p = 0.0129, against a corrected bar of 0.0036). So every repair result is exploratory.
- **No repair met all pre-set conditions.** Merge-back closed 90% of the Thai gap but raised over-refusal and cost
  capability.
- **The circuit map located refusal but did not improve the repair.** Circuits were faithful (0.90 to 1.01 on
  patching, against 0.21 to 0.69 for random heads), yet the circuit-located edit did no better than the same edit on
  random layers.
- **4-bit quantisation cost 2 to 4 safe points** on average, and Earth's Javanese fell from 81.2% to about 71%, again
  through breakdown rather than harm.

Content-safe rate on held-out harmful prompts, full precision, with the preamble:

| Language | Global | Earth | Gap (95% CI) | McNemar p | Confirmed? |
|---|---:|---:|---:|---:|:---:|
| Arabic | 98.2% | 97.1% | +1.2 (-1.2 to +3.5) | 0.6250 | |
| Swahili | 96.0% | 95.5% | +0.6 (-2.3 to +3.4) | 1.0000 | |
| Javanese | 90.9% | 81.2% | +9.7 (+4.0 to +15.3) | 0.0023 | **yes** |
| Bengali | 91.5% | 83.0% | +8.5 (+3.4 to +14.2) | 0.0041 | |
| Thai | 94.9% | 89.2% | +5.7 (+1.7 to +10.2) | 0.0129 | |
| Korean | 96.6% | 92.0% | +4.5 (+1.1 to +8.5) | 0.0386 | |
| English | 95.5% | 94.9% | +0.6 (-2.8 to +3.9) | 1.0000 | |

Repairs on Thai, held-out, with the preamble. None passed all three conditions.

| Arm | Thai safe rate | Gap closed | Largest over-refusal rise | Largest capability drop |
|---|---:|---:|---:|---:|
| Earth (original) | 89.2% | — | — | — |
| ReSta | 90.9% | 30% | +7.0 | 3.0 |
| Circuit (C-ΔΘ) | 88.6% | 0% | +3.4 | 0.8 |
| Merge-back (WiSE-FT) | 94.3% | 90% | +2.6 | 1.5 |
| Circuit merge (selected) | 89.8% | 10% | +2.1 | 1.3 |
| Steering | 90.9% | 30% | +3.5 | 0.3 |
| Global (reference) | 94.9% | — | — | — |

Conditions: close at least half the gap, over-refusal up by at most 2 points, capability down by at most 1 point.
The full results, limits and what did not work are in the
[article](https://huggingface.co/blog/HydraOfMalice/same-model-different-language-different-safety).

## How it works

```
prompts ─► CuratorKIT ─► group split + translation ─► generate (Global, Earth, Fire, Water)
        ─► two LLM judges ─► AuditKIT run records ─► per-language gap test
        ─► CircuitKIT (EAP-IG) ─► SafeTune repairs + random-layer control ─► re-test at FP16, Q4_K_M and NF4
```

| Lexsi kit | What it does here |
|---|---|
| CuratorKIT | Cleans and de-duplicates the prompts, with provenance |
| CircuitKIT | Finds the layers that drive refusal in English, Javanese and Bengali (EAP-IG, four faithfulness checks) |
| SafeTune | Five repairs: ReSta, circuit-restricted C-ΔΘ, WiSE-FT merge-back, circuit merge, refusal-direction steering |
| AuditKIT | One fingerprinted run record per model, arm and precision |

Seven languages (Arabic, Swahili, Javanese, Bengali, Thai, Korean, English), 3,757 prompts from MultiJail, Aya
red-teaming, XSTest look-alikes and Global-MMLU-Lite. Two local judges (Aya Expanse 8B, Qwen3-8B) label every answer,
checked against Aya Expanse 32B, Qwen3-32B and Command A on a 1,680-answer sample. Metrics, thresholds and the gap
test were frozen in [SPEC.md](SPEC.md) before the full run.

## Released artifacts

| Artifact | Link |
|---|---|
| Selected repair (circuit merge), FP16 and Q4_K_M GGUF | [circuit-merge-0p5](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5) · [GGUF](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-merge-0p5-gguf) |
| Other repair arms | [resta-0p1](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-resta-0p1) · [circuit-0p1](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-circuit-0p1) · [merge-back-0p5](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-merge-back-0p5) · [steer-1p0](https://huggingface.co/HydraOfMalice/tiny-aya-earth-langsafe-steer-1p0) |
| Evaluation data, reports, provenance logs, circuit summaries | [tiny-aya-earth-langsafe-eval](https://huggingface.co/datasets/HydraOfMalice/tiny-aya-earth-langsafe-eval) |
| Judge checks (32B and Command A samples) | [docs/judge_checks](docs/judge_checks/README.md) |
| Patches for the Lexsi kits | [docs/upstream_patches](docs/upstream_patches/README.md) |
| Results explorer (no model, no text box) | [space/](space/README.md) |

## Reproduce

Setup and tests (Python 3.12):

```bash
uv venv --python 3.12 .venv
LANGSAFE_PYTHON=.venv/bin/python bash scripts/install_kits.sh --repair
.venv/bin/python -m pytest -q
```

Small end-to-end run, then the full experiment:

```bash
bash scripts/run_experiment.sh --family earth --profile development
bash scripts/run_experiment.sh --family earth --profile full
```

- Pass `--family earth` every time. The command defaults to Fire.
- A rerun resumes after an interruption. Outputs go to `runs/experiments/<profile>/<family>/`.
- Up to the Q4 checks, the full run took about 93 hours on a laptop RTX 4060 (8 GB). The NF4 checks ran on a rented
  GPU. On Windows, run long jobs through `scripts/run_chain.py`.
- No GPU of your own? The [reproduction notebook](notebooks/01_reproduce_results.ipynb) runs every stage at small
  scale on one free Kaggle T4 in about 6 to 10 hours. Its rates are too noisy to compare with the article.

Machines, setup details, CircuitKIT's separate environment and llama.cpp are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/langsafe/` | The pipeline: data, generation, judging, statistics, circuits, repairs, quantisation, provenance |
| `configs/` | `spec.yaml` (the frozen full run) and `development.yaml` (small integration run) |
| `scripts/` | Entry points: `run_experiment.sh`, exports, release and audit tools |
| `notebooks/` | Reproduction notebook and a day-1 smoke test |
| `space/` | Results explorer built from the aggregate reports |
| `article/` | The article as Markdown |
| `docs/` | Spec amendments, runbook, kit gaps, submission material and team notes ([index](docs/README.md)) |
| `tests/` | Test suite |

## Responsible release

- We publish prompts, labels, metrics, repaired checkpoints and logs. We never publish a model's answer to a harmful
  prompt. Raw generations stay git-ignored.
- Node-level circuit scores stay private and are shared with researchers on request. Layers, summaries and the editing
  mask, which reproduce every repair, are public.
- Tiny Aya is CC-BY-NC 4.0 under the Cohere Labs acceptable use policy, so all of this work is non-commercial.

## Team

- **Aditya Arnav**: GitHub [Hydra-Of-Malice](https://github.com/Hydra-Of-Malice), Hugging Face [HydraOfMalice](https://huggingface.co/HydraOfMalice)
- **Pallab Das**: GitHub [PDGamerSG](https://github.com/PDGamerSG), Hugging Face [PDGamerSG](https://huggingface.co/PDGamerSG)

## Licence

Code: MIT. Models and data keep their own licences: Tiny Aya CC-BY-NC 4.0, Aya red-teaming Apache-2.0, MultiJail MIT,
XSTest CC-BY-4.0, Global-MMLU-Lite Apache-2.0. The Lexsi kits are under LSAL (research use).
