# ROI: self-hosting Tiny Aya against closed-model token billing

This is the optional ROI comparison from the hackathon rules. It compares cost only. It is the same under both stories in the article, because it does not depend on whether the safety gap is confirmed.

> **This comparison is hypothetical.** Tiny Aya is CC-BY-NC 4.0, so commercial use needs a licence from Cohere, and we do not know the price. The Lexsi kits also bar selling a service built on them without a commercial licence. Any real deployment must add those costs.

**Status.** Prices were checked on 5 November 2026. Every computed figure below comes from those prices and the assumptions in section 2. Batched throughput is measured on a laptop GPU only, so we cannot yet say whether one rented GPU can carry the load. Those gaps are marked `{{NAME}}`.

## 1. What the numbers say so far

- At our assumed 600,000 requests a month, the cheapest closed models cost less than one always-on GPU: $63.48 for `gpt-5-nano` and $90.96 for Claude Haiku 5.5, against $197.10 to $365.00 for a GPU.
- Self-hosting costs less than the mid-priced models at this volume. Claude Haiku 4.5 costs $909.60.
- Against the cheapest closed models, one GPU breaks even at about 1.3 to 3.4 million requests a month, if it can sustain the throughput. We have not measured that on a rented GPU yet.
- These GPU costs leave out engineering time and licences. Both would make self-hosting look worse.

So cost alone does not justify self-hosting at this volume against the cheapest closed models. The case rests on higher volume, or on what a closed API cannot offer: data that stays on your own hardware, offline use, and testing the exact file you deploy.

## 2. Inputs

Type: **Measured** means we measured it. **Public** means a published price. **Assumption** means we chose it. **To measure** means it is still open.

| Symbol | Input | Value | Type |
|---|---|---:|---|
| R | Requests per month | 600,000 (20,000 a day for 30 days) | Assumption |
| T_user | User tokens per request | 150 | Assumption |
| T_sys | System tokens per request | 366 | Measured for Tiny Aya's default preamble. Assumed equal for closed models |
| T_in | Input tokens per request (T_user + T_sys) | 516 | Derived |
| T_out | Output tokens per request | 200 | Assumption |
| H | Hours in a month | 730 | Assumption, as in Hugging Face's own examples |
| k | Peak-to-average traffic ratio | 3 | Assumption |
| f | Closed-model tokens per Tiny Aya token | 1.0 | Assumption. To measure: {{FERTILITY_AR}} (Arabic), {{FERTILITY_SW}} (Swahili), {{FERTILITY_JV}} (Javanese), {{FERTILITY_BN}} (Bengali), {{FERTILITY_TH}} (Thai), {{FERTILITY_KO}} (Korean) |
| TPS | Sustained batched output tokens per second | 159 on a laptop RTX 4060 (8 GB), with llama.cpp serving the full-precision model to 12 requests at once | Measured 6 Nov; 164 on 12 Nov with 96 prompts |
| TPS_in | Prompt tokens per second under the same load | about 3,000 (3,030 measured with 12 requests at once) | Measured 12 Nov, `scripts/bench_generation.py` |
| P95 | 95th-percentile latency at that load | 9.0 seconds (median 5.9) for a 382-token prompt and up to 96 new tokens, with 12 requests at once | Measured 12 Nov, same run |
| U | Share of a GPU's peak throughput we can safely plan on | {{UTILISATION}} | To measure |
| C_ops | Engineering and operations per month | {{OPS_COST_MONTH}} | Assumption, to be set by the team |
| C_lic | Commercial licences per month | unknown | Unknown. Needs quotes from Cohere and Lexsi Labs |
| C_rep | One-time cost of the repair and its evaluation | About $18 to $40 for the whole 16-model study, at RunPod Secure rates ($0.27 to $0.59 an hour), from about 67 GPU hours | Run ledger, 7 to 10 Nov, gaps over 6 hours counted as idle |

The volume is a mid-sized service chosen for illustration, with no real customer behind it. Costs scale linearly, so section 7 shows other volumes.

## 3. Formula

**Closed model, per request:**

```
c = f × (T_in × P_in + T_out × P_out) / 1,000,000
```

**Closed model, per month:**

```
C_api = R × c
```

**Self-hosted, per month:**

```
peak_tps = k × R × T_out / (H × 3,600)
N        = ceiling( peak_tps / (TPS × U) )          GPUs needed, always on
C_self   = N × H × P_gpu + C_ops + C_lic + C_rep / months_amortised
```

**Break-even volume, for one always-on GPU:**

```
R* = (H × P_gpu) / c
```

P_in and P_out are prices per 1M tokens. P_gpu is the price per GPU-hour.

## 4. Public prices

### Closed models

Standard list prices in USD per 1M tokens, checked 5 November 2026. Vendor pages give no "last updated" date.

| Model | Input | Output | Source |
|---|---:|---:|---|
| `gpt-5-nano` | $0.05 | $0.40 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| `gpt-6-luna` | $0.10 | $0.50 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| Claude Haiku 5.5 (prompts up to 100,000 tokens) | $0.10 | $0.50 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| `gpt-5.4-nano` | $0.20 | $1.25 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 | [Gemini pricing](https://aistudio.google.com/docs/pricing) |
| `gpt-5-mini` | $0.25 | $2.00 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 | [Gemini pricing](https://aistudio.google.com/docs/pricing) |
| `gemini-3.6-flash`, through 31 Dec 2026 | $0.75 | $3.75 | [Gemini pricing](https://aistudio.google.com/docs/pricing) |
| `gpt-5.4-mini` | $0.75 | $4.50 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| Claude Haiku 4.5 | $1.00 | $5.00 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| `gemini-3.6-flash`, from 1 Jan 2027 | $1.50 | $7.50 | [Gemini pricing](https://aistudio.google.com/docs/pricing) |

**Unconfirmed.** Cohere's Command R7B is reported at $0.0375 in and $0.15 out by [a third-party tracker](https://pricepertoken.com/pricing-page/model/cohere-command-r7b-12-2024). The [Cohere pricing page](https://cohere.com/pricing) text we retrieved did not show it, so it stays out of the main tables. If the price is right, it would cost $29.61 a month at this volume, which is the lowest of all.

### GPUs

| GPU | Provider | $/hour | One always-on GPU, per month (× 730) | Source |
|---|---|---:|---:|---|
| RTX A5000 | RunPod Community | $0.16 | $116.80 | [RunPod pricing](https://www.runpod.io/pricing), updated 4 Nov 2026 |
| RTX A5000 | RunPod Secure | $0.27 | $197.10 | Same |
| L4 | RunPod Secure | $0.59 | $430.70 | Same |
| T4 | HF Inference Endpoints | $0.50 | $365.00 | [HF pricing](https://huggingface.co/docs/inference-endpoints/en/support/pricing) |
| L4 | HF Inference Endpoints, GCP | $0.70 | $511.00 | Same |
| L4 | HF Inference Endpoints, AWS | $0.80 | $584.00 | Same |

Tiny Aya fits on all of these. A Q4_K_M file measured 1.997 GiB, and an FP16 checkpoint is about 6.24 GiB. Community-cloud GPUs run on third-party hosts, so they may not suit private data. We use RunPod Secure and HF Endpoints as the two reference prices below.

## 5. Worked arithmetic

**Monthly tokens**

- Input: 600,000 × 516 = 309,600,000, which is 309.6M.
- Output: 600,000 × 200 = 120,000,000, which is 120M.

**Example, `gpt-5-nano`**

- Per request: (516 × $0.05 + 200 × $0.40) / 1,000,000 = (25.8 + 80) / 1,000,000 = $0.0001058.
- Per month: 309.6 × $0.05 + 120 × $0.40 = $15.48 + $48.00 = **$63.48**.

**All closed models at 600,000 requests a month, with f = 1.0**

| Model | Input cost | Output cost | Monthly total | Cost per 1,000 requests |
|---|---:|---:|---:|---:|
| `gpt-5-nano` | $15.48 | $48.00 | $63.48 | $0.1058 |
| `gpt-6-luna` | $30.96 | $60.00 | $90.96 | $0.1516 |
| Claude Haiku 5.5 | $30.96 | $60.00 | $90.96 | $0.1516 |
| `gpt-5.4-nano` | $61.92 | $150.00 | $211.92 | $0.3532 |
| `gemini-3.1-flash-lite` | $77.40 | $180.00 | $257.40 | $0.4290 |
| `gpt-5-mini` | $77.40 | $240.00 | $317.40 | $0.5290 |
| `gemini-3.5-flash-lite` | $92.88 | $300.00 | $392.88 | $0.6548 |
| `gemini-3.6-flash`, through 31 Dec 2026 | $232.20 | $450.00 | $682.20 | $1.1370 |
| `gpt-5.4-mini` | $232.20 | $540.00 | $772.20 | $1.2870 |
| Claude Haiku 4.5 | $309.60 | $600.00 | $909.60 | $1.5160 |
| `gemini-3.6-flash`, from 1 Jan 2027 | $464.40 | $900.00 | $1,364.40 | $2.2740 |

**Self-hosted, one GPU, before operations and licences**

- RTX A5000, RunPod Secure: 730 × $0.27 = **$197.10**.
- T4, HF Endpoints: 730 × $0.50 = **$365.00**.

## 6. Throughput: can one GPU do it?

Seconds in a month: 730 × 3,600 = 2,628,000.

At 600,000 requests:

- Average output: 120,000,000 / 2,628,000 = 45.7 tokens per second.
- Peak output, with k = 3: 137.0 tokens per second.
- Average input: 309,600,000 / 2,628,000 = 117.8 tokens per second.

**What we have measured.** On an Apple M1 Max, with Q4_K_M weights and one request at a time, Fire decoded at a median of 77.81 tokens per second, and 64.92 tokens per second including prompt processing. This came from a development run of 78 requests, before the project moved to Earth.

That single stream would cover the 45.7 average and fall short of the 137.0 peak. It is a laptop figure, and it does not predict batched GPU throughput. It also says nothing about cost, because that machine has no hourly price.

On a laptop RTX 4060 (8 GB) with full-precision Earth weights, batched generation reached 144.9 output tokens per second at batch 16 in Transformers, and 169.2 through llama-server with 16 slots. That run used 48 prompts of about 383 tokens and at most 96 new tokens. Those figures sit above the 137.0 peak, but this was a short run on a laptop GPU, with answers shorter than the 200 tokens we assume. They do not replace a measurement on a rented GPU, and they are not Q4_K_M figures.

**What we still need.** Batched throughput on a rented-class GPU, with Earth:

| Measure | Value |
|---|---:|
| Output tokens per second under load | 159 |
| GPU | a laptop RTX 4060 (8 GB), with llama.cpp serving the full-precision model to 12 requests at once |
| Runtime and batch size | llama.cpp llama-server, FP16 GGUF, 12 parallel slots of 768 tokens, greedy decoding |
| Prompt tokens per second under load | about 3,000 (3,030 measured with 12 requests at once) |
| 95th-percentile latency | 9.0 seconds (median 5.9) for a 382-token prompt and up to 96 new tokens, with 12 requests at once |

GPUs needed at 600,000 requests: N = ceiling(137.0 / (159 × {{UTILISATION}})) = {{N_GPUS_600K}}.

Self-hosted total at 600,000 requests: {{C_SELF_600K}} per month.

## 7. Break-even and other volumes

**Break-even for one always-on GPU.** R* = monthly GPU cost / cost per request. The last two columns show the throughput one GPU would need at that volume, as average and peak output tokens per second.

Against RTX A5000, RunPod Secure ($197.10):

| Closed model | Break-even requests a month | Average tok/s needed | Peak tok/s needed (k = 3) |
|---|---:|---:|---:|
| `gpt-5-nano` | 1,862,949 | 141.8 | 425.3 |
| Claude Haiku 5.5 | 1,300,132 | 98.9 | 296.8 |
| `gemini-3.1-flash-lite` | 459,441 | 35.0 | 104.9 |
| `gpt-5-mini` | 372,590 | 28.4 | 85.1 |
| Claude Haiku 4.5 | 130,013 | 9.9 | 29.7 |

Against T4, HF Endpoints ($365.00):

| Closed model | Break-even requests a month | Average tok/s needed | Peak tok/s needed (k = 3) |
|---|---:|---:|---:|
| `gpt-5-nano` | 3,449,905 | 262.5 | 787.6 |
| Claude Haiku 5.5 | 2,407,652 | 183.2 | 549.7 |
| `gemini-3.1-flash-lite` | 850,816 | 64.8 | 194.3 |
| `gpt-5-mini` | 689,981 | 52.5 | 157.5 |
| Claude Haiku 4.5 | 240,765 | 18.3 | 55.0 |

If measured throughput falls below the peak figure, one GPU cannot carry that volume, and the real break-even moves higher.

**Closed-model cost at other volumes, per month**

| Model | 100,000 | 600,000 | 3,000,000 | 10,000,000 |
|---|---:|---:|---:|---:|
| `gpt-5-nano` | $10.58 | $63.48 | $317.40 | $1,058.00 |
| Claude Haiku 5.5 | $15.16 | $90.96 | $454.80 | $1,516.00 |
| `gemini-3.1-flash-lite` | $42.90 | $257.40 | $1,287.00 | $4,290.00 |
| `gpt-5-mini` | $52.90 | $317.40 | $1,587.00 | $5,290.00 |
| Claude Haiku 4.5 | $151.60 | $909.60 | $4,548.00 | $15,160.00 |

**Peak output needed at those volumes (k = 3):** 22.8, 137.0, 684.9 and 2,283.1 tokens per second.

**Self-hosted cost at those volumes:** {{C_SELF_100K}}, {{C_SELF_600K}}, {{C_SELF_3M}} and {{C_SELF_10M}}, with {{N_GPUS_100K}}, {{N_GPUS_600K}}, {{N_GPUS_3M}} and {{N_GPUS_10M}} GPUs.

## 8. What could change the answer

**In favour of closed models**

- **Prompt caching.** Closed APIs charge less for a repeated system prompt. With the 366-token prompt cached, `gpt-5-nano` falls from $63.48 to $53.60 (cached input at $0.005 per 1M), and Claude Haiku 5.5 from $90.96 to $71.20 (cache hits at $0.01 per 1M). These figures ignore Anthropic's cache-write charge.
- **Shorter system prompts.** A closed model does not need Tiny Aya's preamble. We assumed an equal length to keep the comparison simple.
- **Engineering time.** C_ops is left out of the tables above. Running a GPU service needs monitoring, updates and on-call cover.
- **Licences.** C_lic is unknown and could be the largest item.

**In favour of self-hosting**

- **Tokenizer efficiency.** Tiny Aya has a 262,144-token vocabulary built for many languages. Closed tokenizers may split Bengali, Thai, Arabic or Korean text into more tokens, which raises billed cost by the factor f. We left f at 1.0 until we measure it.
- **Scale to zero.** Traffic with long idle periods can use fewer GPU-hours. HF Endpoints bill per minute.
- **Owned hardware.** A machine you already own costs only power and depreciation. We did not model this.

**Either way**

- **Reasoning tokens.** Some closed models bill hidden reasoning as output. Gemini's page says its output prices include thinking tokens. Our T_out assumes reasoning is off or minimal. If it is on, closed costs rise.
- **Price changes.** `gemini-3.6-flash` doubles on 1 January 2027, from $682.20 to $1,364.40 at this volume.
- **Quality.** We did not run any closed model on our safety or capability tests. A cheap closed model may refuse better or worse than Tiny Aya in a given language. This document cannot say which model is the better choice.

## 9. To finish this document

1. Measure batched throughput and latency on a T4, using Earth, with the same prompt lengths as section 2. Kaggle's free T4 is enough. Fill section 6.
2. Count tokens for a shared sample of Arabic, Swahili, Javanese, Bengali, Thai and Korean prompts with each closed tokenizer we can access. Fill f. If a tokenizer is not available, say so and leave f at 1.0.
3. Set U from the throughput run. Compute N and C_self.
4. Record GPU hours for the repair and its evaluation, and fill C_rep.
5. Re-check all prices on the day of submission and update section 4.
