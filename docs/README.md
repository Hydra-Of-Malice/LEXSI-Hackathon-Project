# Documentation

## For readers

| File | Contents |
|---|---|
| [../SPEC.md](../SPEC.md) | The pre-registered specification: metrics, thresholds, the gap test, and every amendment with its date |
| [RUNBOOK.md](RUNBOOK.md) | Machines, setup, CircuitKIT's separate environment, llama.cpp, long jobs on Windows |
| [UPSTREAM.md](UPSTREAM.md) | Gaps we found in the Lexsi kits and how we worked around them |
| [upstream_patches/](upstream_patches/README.md) | Six patches for the kits, with pull request texts |
| [judge_checks/](judge_checks/README.md) | Aggregates behind the judge-agreement numbers (32B judges and Command A samples) |
| [submission/abstract.md](submission/abstract.md) | The submission abstract |
| [submission/roi.md](submission/roi.md) | Cost comparison: self-hosting Tiny Aya against closed-model APIs |
| [REMOTE_NF4.md](REMOTE_NF4.md) | How the NF4 checks ran on a rented GPU |

## Submission sources

The article and the model and dataset cards are generated from templates and measured values by
`scripts/fill_submission.py` and `scripts/release_cards.py`.

| File | Contents |
|---|---|
| [submission/article.md](submission/article.md) | Article template with placeholders. The published text is [../article/README.md](../article/README.md) |
| [submission/sentences.json](submission/sentences.json) | Hand-written values and links that fill the templates |
| [submission/model_card_template.md](submission/model_card_template.md), [submission/dataset_card_template.md](submission/dataset_card_template.md) | Card templates |
| [submission/filled/](submission/filled/) | Generated cards, ROI page and values |
| [submission/checklist.md](submission/checklist.md), [submission/decisions.md](submission/decisions.md) | Submission rules, their status, and team decisions |

## Team notes and history

| File | Contents |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Team log, newest entry first |
| [DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md) | The 4 November development run on Fire, before the move to Earth |
| [MAC_STATUS.md](MAC_STATUS.md), [CUDA_EVALUATION.md](CUDA_EVALUATION.md), [CUDA_HANDOFF.patch](CUDA_HANDOFF.patch) | Machine notes from the Fire stage |
| [REVIEW_WORKFLOW.md](REVIEW_WORKFLOW.md) | The human translation-review and judge-calibration workflow |
| [research/](research/README.md) | Research notes from before the hackathon started (2 November) |
| [proposal/](proposal/), [brief/](brief/) | The first plan (Fire) and the mentor brief |
