# Judge checks: aggregates only

The numbers behind the article's judge-agreement sentences (`KAPPA_LOCAL_VS_ROUTER` and `COMMAND_A_SENTENCE` in
`docs/submission/sentences.json`). Each file holds kappa, agreement and label counts per language. None holds a
prompt, an answer or a verdict. The judged rows they come from stay in `runs/`, which git ignores, because they quote
model answers.

| Files | What was judged | Made by | Cost |
|---|---|---|---|
| `global`, `earth`, `earth-circuit_merge-0p5` `.router-sample-comparison.json` | 1,680 held-out answers (20 per language, prompt type and preamble cell) re-judged by the 32B pair (Aya Expanse 32B, Qwen3-32B), compared with the local 8B pair | `scripts/router_sample.py --tags global,earth,earth-circuit_merge-0p5 --per-cell 20 --execute`, 10 Nov | about $1.0 |
| `global-q4`, `earth-q4`, `earth-circuit_merge-0p5-q4` `.router-sample-comparison.json` | The same design on the three Q4_K_M models | the same script with the `-q4` tags, 11 Nov | about $1.0 |
| `*.command-a-sample-comparison.json` | The first sample judged by Command A alone, compared with both pairs. `verified` is false by design: the spec calibrates a pair against a pair | `--judge-set command-a`, 11 to 12 Nov | about $4.60 |
| `router-sample-usage.json`, `command-a-sample-usage.json` | Billed calls and tokens per run. The first router entry over-counts calls; its `correction` entry explains why | | |

The calibration files the report uses are written separately, by `--write-calibration`, into the run's
`calibration/` folder. Only the three full-precision tags have them, so the final report is unchanged by the 4-bit
and Command A checks.
