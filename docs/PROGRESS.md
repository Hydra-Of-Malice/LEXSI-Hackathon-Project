# Progress log

Newest entry first. Updated with every push, so anyone on the team can see what is running and who owns what. Pull
`main` before starting a run.

## Who is doing what

| Person / machine | Now | Next |
|---|---|---|
| Aditya's laptop (RTX 4060, 1.2 TB free) | 10-language MultiJail sweep with both judges; downloading the full Global, Fire and Base weights | Full Fire run: baseline, circuits, repairs, 4-bit checks |
| Pallab's M1 Max | Free | Please pull `main` before restarting anything (see 4 Nov) |
| Aditya | Accept the `tiny-aya-earth` licence (needed only for the Earth run) | Send the brief to mentors |

## Still needs a person
- Native readers for Hindi/Hinglish, Bengali and Tamil, to review translations and hand-label a sample.
- HF credits, for the 32B judges that verify the free local judges.
- A go-ahead before anything is published to Hugging Face, and the final submission form.
- The rules ask for a public demo or repo by 14 Nov, so this repo must go public before the deadline. That's the
  team's call. Raw model outputs stay out of it either way.

## 5 November 2026
- **11:45** **10-language sweep, first half** (with preamble, Qwen judge only, 4-bit, 131 repair prompts per language;
  "safe" means anything but full compliance). The 9-point gap in the report did not reproduce: mean safe is 76.9% for
  Global and 76.5% for Fire, and both are worst in Javanese (62.6% and 59.5%). Fire trails most in Vietnamese (6.1 points,
  p = 0.04) and Swahili (5.3, p = 0.17), and leads in Bengali (5.3, p = 0.04) and Arabic (6.1, p = 0.06). With 10
  tests, those p-values can be chance. Absolute levels sit well below the report's, so our judge or 4-bit weights may
  be measuring something different; this does not disprove the report yet. Now rerunning with both judges and adding
  the no-preamble half, which crashed the first time. **The full Fire grid stays on hold, and the mentor brief should
  not go out until this is settled:** it states the gap as fact.
- **11:40** The model downloader was not resuming after dropped connections, so it is replaced by
  `scripts/fetch_models.py`, which continues partial files. Please use that one on slow connections.
- **05:30** **Pilot result, and it matters.** On the repair split (4-bit, one judge, default preamble, full refusal), Fire is
  *not* less safe than Global in English, Hindi or Bengali:

  | Language | n | Global | Fire | p |
  |---|---|---|---|---|
  | English | 120 | 77.5% | 81.7% | 0.27 |
  | Hindi | 111 | 75.7% | 79.3% | 0.42 |
  | Bengali | 122 | 72.1% | 74.6% | 0.55 |

  The report's 78.1% is Fire's worst of 10 MultiJail languages, and we assumed that was an Indic language. Now running
  all 10 languages with and without the preamble (`scripts/pilot_multijail.py`, repair split only) to see where the gap
  really is. **Hold the full Fire grid until this is back:** the answer may change the target languages.
- **05:15** The first weight download died in a network outage. `scripts/fetch_models.sh` now retries until complete.
- **00:35** HF login done on the laptop. Global, Fire, Base and Aya Expanse are accessible; Earth still needs its
  licence accepted. Full weights are downloading while the 4-bit pilot runs.

## 4 November 2026
- **23:10** Still waiting on the HF login on the laptop. llama.cpp (CUDA build) is installed. The first 4-bit download
  died on the flaky connection after two hours, so downloads now go through `scripts/fetch_models.py`, which resumes.
- **20:50** Waiting on the HF login. Everything up to here is on `main`.
- **20:48** Pushed the design fixes (`7f598be`). The development run showed the old setup could not answer our main
  question, so **please don't resume the old full grid**; its spec has changed anyway. See "Amendments" in
  [SPEC.md](../SPEC.md) for all nine changes:
  - the circuit arm now edits the top 20% of layers, not 30 of 36;
  - the circuit arm shares ReSta's step sizes;
  - there is a random-layer control and a circuit-restricted merge-back arm;
  - the target is the largest gap, not the lowest refusal rate;
  - selection uses the judges plus MMLU `dev` accuracy instead of phrase matching.
- **19:50** Pallab pushed the end-to-end pipeline and the development results
  ([DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md)). No repair passed on the small sample. The full Fire run stopped
  on Mac disk space.
