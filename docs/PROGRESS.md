# Progress log

Newest entry first. Updated with every push, so anyone on the team can see what is running and who owns what. Pull
`main` before starting a run.

## Who is doing what

| Person / machine | Now | Next |
|---|---|---|
| Aditya's laptop (RTX 4060, 1.2 TB free) | 10-language MultiJail sweep with both judges; downloading the full Global, Fire and Base weights | Full Fire run: baseline, circuits, repairs, 4-bit checks |
| Pallab's M1 Max | Free | Please pull `main` before restarting anything (see 4 Nov) |
| Aditya | Accept the `tiny-aya-earth` licence (**now blocking**: Earth is the main model) | Send the brief to mentors |

## Still needs a person
- Native readers for Hindi/Hinglish, Bengali and Tamil, to review translations and hand-label a sample.
- HF credits, for the 32B judges that verify the free local judges.
- A go-ahead before anything is published to Hugging Face, and the final submission form.
- The rules ask for a public demo or repo by 14 Nov, so this repo must go public before the deadline. That's the
  team's call. Raw model outputs stay out of it either way.

## 5 November 2026
- **21:15** **The team decided to retarget to Earth.** Earth is now the main model, with Javanese, Bengali, Thai and Korean
  added next to Arabic and Swahili, and Fire as a comparison. See SPEC.md amendments 12 to 14. The gap must be confirmed at
  full precision on held-out items before any repair claim. Waiting on the `tiny-aya-earth` licence for the full weights.
- **20:55** **The report's gap reproduces once we score content, and it sits outside the regional models' own regions.**
  With-preamble half of the sweep: 131 repair-split prompts per language, 4-bit, Aya Expanse 8B and Qwen3-8B; "safe" means
  both judges say safe. Local judges stand in for Command A, so this is a planning signal, not a result.

  | | Global | Fire | Earth |
  |---|---|---|---|
  | Mean of 10 languages (report: 91.1 / 90.0 / 87.8) | 94.9 | 94.1 | 91.6 |
  | Worst language (report: 87.0 / 78.1 / 77.5) | 87.0 | 80.9 | 77.1 |
  | Javanese | 87.0 | 80.9 (p = 0.15) | 77.1 (p = 0.04) |
  | Bengali | 90.1 | 90.8 | 82.4 (p = 0.02) |
  | Thai | 96.9 | 96.9 | 92.4 (p = 0.15) |

  So Earth has the clearest gap, Fire's is only in Javanese and isn't significant, and Fire shows no gap in Bengali, its
  own region. With 20 tests, neither significant p-value survives correction. Our Global worst language differs from the
  report's (Javanese, not Arabic), and a parser bug that zeroed one judge is fixed: Aya's verdicts were being cut off.

  **Measurement warning:** the judges rarely agree beyond chance. Aya calls almost everything safe, so kappa is under 0.6 in
  most cells and the signal comes mostly from Qwen. The 32B router judges matter more than we planned.
  The no-preamble half is still running. **Keep holding the full grid and the mentor brief.**
- **12:40** Content safety is now the primary axis in the pipeline (`f09a5a1`), and the submission drafts are in
  `docs/submission/`. Aditya has confirmed both teammates are registered with matching HF usernames. See
  `docs/submission/checklist.md` for every rule and its status.
- **12:10** **Why we couldn't reproduce the report: we were measuring something else.** Tiny Aya's report scores the *content*
  of an answer (safe, unsafe or invalid, by Command A in contextual safety mode). Our judges score *refusal*, so a harmless
  answer to a harmful prompt counted against the model. This could explain most of the 76.9% vs 91.1% mean. It also
  matters that:
  - the report's mean gap is small (Global 91.1, Fire 90.0), and the 9-point gap is in one unnamed worst language;
  - only Global's per-language numbers are public, and its worst language is Arabic (87), so nobody outside Cohere can
    say which language is Fire's 78.1;
  - the report ties safety to having tuning data in a language, and Fire's cluster covers only English and Bengali of the
    10 MultiJail languages, so the loss may sit *outside* its region, not in it. That is an inference, not a finding;
  - the regional models are merges with Global, and the final recipe isn't published.

  A new sweep is rescoring the cached answers with the report's three-label rubric, and adds Earth. This replaces the
  refusal-rubric sweep, which was stopped partway. Local judges stand in for Command A, so expect a difference.
- **12:05** The laptop rebooted to clear a leak: `OCControl.Service` (Alienware Command Center) had grown to 88 GB of
  commit memory and 129,000 handles over six days. That starved Windows, which crashed one download and the sweep. It's
  back to 84 MB. It may grow again, so watch it on long runs.
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
