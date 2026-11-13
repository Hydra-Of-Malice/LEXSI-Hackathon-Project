# What is still open

Run `runs/experiments/full/earth`, spec `bdabb7edab9eead8`, frozen: True, validated: False. Blocks kept: ALT-A, ALT-A2.

## article.md (12 open)

- `PR_AUDITKIT_SCORER`: Link to the auditkit scorer pull request, or 'planned'. None opened yet
- `PR_AUDITKIT_LOADERS`: Link to the auditkit loaders pull request, or 'planned'. None opened yet
- `PR_SAFETUNE_TOP_LAYERS`: Link to the safetune top layers pull request, or 'planned'. None opened yet
- `PR_SAFETUNE_LOADERS`: Link to the safetune loaders pull request, or 'planned'. None opened yet
- `PR_CIRCUITKIT_TASK`: Link to the circuitkit task pull request, or 'planned'. None opened yet
- `PR_CURATORKIT_SPLIT`: Link to the curatorkit split pull request, or 'planned'. None opened yet
- `SPACE_URL`: Demo Space, if built
- `DEMO_DESCRIPTION`: What the demo does. It must not show harmful completions
- `HAND_LABELLED_ITEMS`: Number of hand-labelled items
- `HAND_CHECK_QUEUE_TOTAL`: All rows flagged for a hand check, sampled or not
- `JUDGE_ACCURACY`: Judge accuracy on the random audit, %
- `HF_USERNAME_PALLAB`: HF username. Must match the registration form

## model_card_template.md (14 open)

- `DATASET_REPO_ID`: Evaluation dataset repo id
- `NAME`
- `MODEL_NAME`: Repo name. Keep kit names out of it
- `ARTICLE_URL`: HF article link
- `CIRCUIT_SOURCE`: Scores file hash for circuit arms, or 'not used'
- `LANGUAGE_REPORT_NOTE`: Languages left out for AC1 below 0.6, or delete
- `KAPPA_BY_LANGUAGE`: Cohen's kappa on refusal labels, for each language
- `HAND_LABELLED_ITEMS`: Number of hand-labelled items
- `JUDGE_ACCURACY`: Judge accuracy on the random audit, %
- `MODEL_SPECIFIC_LIMITATIONS`: Failures seen with this arm, or delete
- `MODEL_REPO_ID`: This repo's id
- `CODE_COMMIT`: Commit used for the run
- `CHECKPOINT_SHA256`: Hash of the weights file
- `CONTACT`: Contact route. Organisers mask usernames for judges

## dataset_card_template.md (9 open)

- `DATASET_NAME`: Dataset repo name
- `NAME`
- `ARTICLE_URL`: HF article link
- `SPLIT_COUNT_TABLE_NOTE`: Shortfalls against the caps, from the coverage block
- `TRANSLATION_FAILED_REASONS`: Count by rejection reason
- `DATASET_REPO_ID`: Evaluation dataset repo id
- `RELEASE_LICENCE_DECISION`: Confirm the licence choice after asking organisers
- `CODE_COMMIT`: Commit used for the run
- `CONTACT`: Contact route. Organisers mask usernames for judges

## roi.md (17 open)

- `NAME`
- `FERTILITY_AR`: Closed-tokenizer tokens per Tiny Aya token, Arabic sample
- `FERTILITY_SW`: Closed-tokenizer tokens per Tiny Aya token, Swahili sample
- `FERTILITY_JV`: Closed-tokenizer tokens per Tiny Aya token, Javanese sample
- `FERTILITY_BN`: Closed-tokenizer tokens per Tiny Aya token, Bengali sample
- `FERTILITY_TH`: Closed-tokenizer tokens per Tiny Aya token, Thai sample
- `FERTILITY_KO`: Closed-tokenizer tokens per Tiny Aya token, Korean sample
- `UTILISATION`: Share of peak throughput to plan on, as a fraction
- `OPS_COST_MONTH`: Engineering and operations cost per month. Label as an assumption
- `N_GPUS_600K`: GPUs needed at 600,000 requests
- `C_SELF_600K`: Self-hosted cost at 600,000 requests
- `C_SELF_100K`: Self-hosted cost at 100,000 requests
- `C_SELF_3M`: Self-hosted cost at 3,000,000 requests
- `C_SELF_10M`: Self-hosted cost at 10,000,000 requests
- `N_GPUS_100K`: GPUs needed at 100,000 requests
- `N_GPUS_3M`: GPUs needed at 3,000,000 requests
- `N_GPUS_10M`: GPUs needed at 10,000,000 requests
