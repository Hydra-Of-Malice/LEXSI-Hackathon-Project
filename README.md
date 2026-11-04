# Same model, different language, different safety

**Find, repair and verify the per-language safety drift in Tiny Aya's regional models, using the Lexsi alignment and
safety stack.** Lexsi Labs AI Safety & Alignment Hackathon, Track 2b (*Repairing safety drift without retraining*).

Cohere's Tiny Aya Fire (South Asia) and Earth (Africa, West Asia) are almost as safe as Tiny Aya Global **on average**.
In their **weakest language**, though, they are 9–10 points less safe on MultiJail: 87.0% safe for Global, 78.1% for
Fire and 77.5% for Earth (Tiny Aya tech report, Table 7). This project:

1. measures that gap properly, per language;
2. finds where refusal lives with CircuitKIT;
3. repairs it with SafeTune, comparing a circuit-restricted edit against whole-model and naive baselines;
4. checks that the repair costs nothing elsewhere and survives 4-bit quantisation.

> **Thesis.** A repair located with CircuitKIT and applied with SafeTune restores per-language safety without raising
> benign over-refusal or hurting capability, and it still holds at 4-bit.

## Start here (team)

| Read | For |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Which machine runs what, Mac (M1 Max) setup, the day-1 command |
| [docs/brief/mentor_brief.pdf](docs/brief/mentor_brief.pdf) | One-page summary we share with mentors |
| [docs/proposal/](docs/proposal/Lexsi_Hackathon_Proposal_Multilingual_Safety_Repair.pdf) | Full 10-page proposal with references |
| [SPEC.md](SPEC.md) | Pre-registered metrics, thresholds and rules (don't change after freeze without an amendment) |
| [docs/research/](docs/research/README.md) | Hackathon rules, judging, models, kit deep-dives, Discord clarifications, idea ranking |
| [docs/UPSTREAM.md](docs/UPSTREAM.md) | Gaps in the Lexsi kits and the PRs we plan |

## Pipeline

```
CuratorKIT ─► langsafe data ─► generate ─► 2 LLM judges ─► AuditKIT run ─► CircuitKIT (EAP-IG) ─► SafeTune arms ─► re-test ─► NF4 / GGUF re-test
 clean+dedup   group splits     Global/Fire/   Aya Expanse 32B   fingerprint,    refusal circuit      ReSta, C-ΔΘ,       3 axes     same 3 axes
 provenance    translations     Earth          + Qwen3-32B       compare         per language         merge-back, steer
```

| Stage | Lexsi kit | langsafe module |
|---|---|---|
| Clean, near-duplicate filter, provenance | CuratorKIT | `data/curate.py` |
| Sources, group-aware split, leakage check, translation | — (missing upstream) | `data/` |
| Generation (preamble on/off, cached, fingerprinted) | — | `gen/` |
| Multilingual refusal judging, agreement, hand-check queue | — (AuditKIT's detector is English) | `judge/`, `eval/judge_run.py` |
| Run records, fingerprints, compare | AuditKIT | `eval/auditkit_bridge.py` |
| Statistics (bootstrap CIs, exact McNemar) | — (AuditKIT PR #9 open) | `eval/stats.py`, `eval/axes.py` |
| Refusal circuits, faithfulness, cross-language overlap | CircuitKIT | `circuits/` |
| Repair arms | SafeTune | `repair/arms.py`, `repair/select.py` |
| 4-bit verification | bitsandbytes, llama.cpp | `gen/model.py`, `quant/gguf.py` |
| Tamper-evident run ledger + `lexsi_provenance.json` | — | `provenance.py` |

The full design is in [SPEC.md](SPEC.md) (pre-registered metrics, thresholds and rules). Gaps found in the kits are in
[docs/UPSTREAM.md](docs/UPSTREAM.md).

## Status

The executable experiment now connects data preparation, original-model baselines, circuit discovery, all four
repair arms, selection on repair data, held-out evaluation, the two local judges, AuditKIT, Q4 export, and reports.
Generation and capability support both preamble conditions. Stages resume only when their inputs, settings, and
saved artifacts match. The development profile uses small real samples and is separate from the canonical spec.

The real development run completed all four arms and nine FP16/Q4 model variants: 702 held-out responses,
1,404 individual judge verdicts, and 1,080 capability decisions. All 126 tests pass, the 65 active stage records
verify, and the selected checkpoint runs the benign demo. Q4 inference measured about 78–80 decode tokens/s
on the M1 Max. **No repair passed acceptance:** improvements are mixed, capability drops remain, and judging
is uncalibrated. See [development results](docs/DEVELOPMENT_RESULTS.md) for scores, limits, and local artifacts.
The full benchmark needs more storage; human/translation review, mentor freeze, and CUDA NF4 remain pending.

The current Mac inventory, completed full-data stage, storage constraint, and
review-sheet paths are in [Mac status](docs/MAC_STATUS.md). The portable CUDA
plan is in [CUDA evaluation](docs/CUDA_EVALUATION.md).

Scientific acceptance is separate from software completion. Unreviewed translations, missing human/router
calibration, missing precision conditions, and incomplete evaluation cells cannot produce a successful repair claim.
The draft spec remains unfrozen; nobody's labels or mentor approval are generated automatically.

## Quick start

```bash
uv venv --python 3.12 .venv
LANGSAFE_PYTHON=.venv/bin/python bash scripts/install_kits.sh --repair
.venv/bin/python -m pytest -q
```

**Who runs what, where** (M1 Max, Kaggle, laptop, HF router), plus the Mac setup: [docs/RUNBOOK.md](docs/RUNBOOK.md).
Day-1 smoke test on any machine: `JUDGES=local bash scripts/day1.sh`. On Kaggle/Colab, open
[`notebooks/00_day1_smoke_test.ipynb`](notebooks/00_day1_smoke_test.ipynb).

Run the complete local integration experiment, then the larger configured experiment:

```bash
bash scripts/run_experiment.sh --family fire --profile development
.venv/bin/python -m langsafe.cli status --directory runs/experiments/development/fire

bash scripts/run_experiment.sh --family fire --profile full
```

Rerun the same command after an interruption. Results, checkpoints, logs, audit sheets, and the hash-verified stage
manifest are isolated under `runs/experiments/<profile>/<family>/`. `report.md` and `report.json` contain aggregate
results and explicit validation gaps. See the [runbook](docs/RUNBOOK.md) for CircuitKIT's separate environment,
llama.cpp, calibration, and the interactive demo. Only one model job should use the accelerator at a time.

This is the hackathon's repair-without-retraining track: the weight arms apply compatible checkpoint deltas or
interpolation, and steering adds an activation direction. No gradient-training claim is made for these methods.

## Responsible release
- Only prompts, labels, metrics, adapters/checkpoints of **repaired** models and logs are published.
- Model outputs on harmful prompts (`runs/gens/`, `runs/generations/`) are git-ignored and never uploaded.
- Tiny Aya is CC-BY-NC 4.0 under the Cohere Labs acceptable-use policy, so all work here is non-commercial.

## Team
Aditya Arnav ([@Hydra-Of-Malice](https://github.com/Hydra-Of-Malice)) · Pallab Das ([@PDGamerSG](https://github.com/PDGamerSG))

## Licence
Code: MIT. Models and data keep their own licences: Tiny Aya CC-BY-NC 4.0, Aya red-teaming Apache-2.0, MultiJail MIT,
XSTest CC-BY-4.0, Global-MMLU-Lite Apache-2.0. The Lexsi kits are under LSAL (research use).
