# Progress log

Newest entry first. Updated with every push, so anyone on the team can see what is running and who owns what. Pull
`main` before starting a run.

## Who is doing what

| Person / machine | Now | Next |
|---|---|---|
| Aditya's laptop (RTX 4060, 1.2 TB free) | Full Earth run: circuits, repairs, 4-bit checks (baseline done) | Blind read of the selected repair, then the report |
| Pallab's M1 Max | Free | Please pull `main`, and don't start another full run: this one owns the frozen spec |
| Aditya | Send the brief to mentors (hold it until the note below is read) | Native readers, HF credits |

## Still needs a person
- Native readers for Javanese first, then Bengali, Arabic, Swahili, Thai and Korean, to review translations and hand-label
  a sample. Javanese matters most, because the main result depends on it.
- HF credits, for the 32B judges that verify the free local judges.
- A go-ahead before anything is published to Hugging Face, and the final submission form.
- The rules ask for a public demo or repo by 14 Nov, so this repo must go public before the deadline. That's the
  team's call. Raw model outputs stay out of it either way.

## 8 November 2026
- **07:56** **Fourth circuit done: Earth, English** (1 hour 28 minutes), with Global's openers "I'm" and "To". Patching
  ratio 1.00, ablation ratio 0.78 (Global's was 0.92), best same-size baseline 0.05, refusal signal 6.9 logits (Global 6.7).
  Earth's refusal signal in English is as strong as Global's, but its circuit holds up less well when everything else is
  zeroed. Earth Javanese started at 07:55, so the circuits should end about 10:50. Memory: 3.5 GB free, 63.3 of 76.7 GB
  committed.

- **06:30** **Third circuit done: Global, Bengali** (1 hour 45 minutes), so all three Global circuits are in. Earth's three
  started at 06:27, so the circuits should end about 11:30 and the repairs start after.

  | Global | English | Javanese | Bengali |
  |---|---:|---:|---:|
  | Patching ratio (unclipped) | 0.99 | 0.91 | 1.01 |
  | Ablation ratio (unclipped) | 0.92 | 0.31 | 1.06 |
  | Best same-size baseline, patching scale | 0.06 | 0.04 | 0.03 |
  | Refusal signal (logits) | 6.7 | 1.7 | 2.0 |
  | Refusal opener: share of Global's refusals / compliances it opens | "I'm" 46/62, 0/70 | "Aku" 19/34, 6/83 | "আমি" 23/35, 0/81 |
  | Compliance opener: share of compliances / refusals it opens | "To" 12/70, 0/62 | "N" 9/83, 2/34 | "আ" 13/81, 0/35 |

  - **The table now shows unclipped ratios.** CircuitKIT clips its headline ratios to 0–1, so Bengali read 1.00 and 1.00. The
    unclipped ablation ratio is 1.06: the circuit alone pushes toward refusing a little harder than the full model. The
    article says what a value above 1 means.
  - **Bengali's openers are as clean as English's**: neither ever opens an answer of the other kind. Javanese is the only
    language where they overlap, which fits its weak signal and low ablation ratio.
  - Memory: 15.8 GB free, 63.2 of 76.7 GB committed.

- **04:50** **Second circuit done: Global, Javanese**, in 1 hour 24 minutes (stability took 33 minutes, not 53). Patching ratio
  0.91, ablation ratio 0.31. Bengali started at 04:42 at about 48 s an example, so the circuits should end between about 10:30
  and 12:45, and NF4 stays on track for Wednesday.
  - **The baseline row was wrong for Javanese, now fixed.** In Javanese every raw score is below zero (full model -0.68,
    harmless twin -2.35, best random set -2.29), so CircuitKIT calls the ratio to the baseline undefined, and our "times the
    baseline" row would have printed 0.36 for a circuit that clearly beats it. The row now puts the best same-size baseline on
    the patching scale: 0.06 in English and 0.04 in Javanese, against 0.99 and 0.91 for the circuits. The clean and harmless
    scores are only in the stage logs, so `fill_submission.py` reads them there and checks that the line matches the summary.
  - **New row: refusal signal.** The harmful prompt raises the refusal opener's logit over the compliance opener's by 6.7
    logits in English but only 1.7 in Javanese, so the Javanese ratios rest on a much weaker contrast.
  - **Why it is weaker (counts only):** "Aku" opens 19 of Global's 34 Javanese refusals on the repair split, but also 6 of
    its 83 compliances, and the compliance opener "N" opens only 9 of the 83 (and 2 refusals). The circuit write-up must say
    that the Javanese task is a loose proxy, and the low ablation ratio (0.31) should be read with that in mind.

- **03:20** **First circuit done: Global, English.** Patching ratio 0.99, ablation ratio 0.92, and the circuit scores 11.6 times the
  better of a random and a magnitude-ranked node set of the same size. 16 pairs reached the run, openers "I'm" and "To", and the
  Hugging Face and TransformerLens versions agreed. It took 1 hour 51 minutes, not the 1 hour 20 I estimated, because the
  stability pillar alone took 53 minutes. Five circuits remain, so circuits now end about 12:45 today and NF4 about 10:45 on
  Wednesday 11 November. That's three hours later, still four days before the deadline.
  - **Stability says less than it seems.** All three seeds found the same 429 nodes (Jaccard 1.00), but every seed reads the
    same 16 pairs, so this shows that discovery is deterministic, not that the circuit holds on other pairs. The article now says
    so, using a sentence `fill_submission.py` writes from the summaries, and the circuit table gains the baseline row.
  - **Javanese openers:** "Aku" for refusals and "N" for compliance. "N" is the first piece of several common Javanese openings,
    taken from Global's own answers, not the fallback. A one-letter opener blurs the refuse-or-comply contrast a little, so the
    circuit write-up should say so.
  - **Write-up checked against the baseline.** A trial report in the scratchpad filled 280 values without touching the run. That
    turned up four fixes, now in: the blind read moves next to the result (Amendment 16 asks for that), the failure-type sentence
    names Javanese, one judge sentence started in lower case, and the draft box was being filled with sentences.
  - **Target language:** Amendment 16 wants the blind read in the target language. If the repair split picks a language other
    than Javanese, that language needs its own blind read after the repairs.
  - `scripts/token_fertility.py` is ready for the ROI's tokens-per-request factor. It hasn't run, because it needs OpenAI's
    `o200k_base` tokenizer file and I asked before downloading it.

- **01:30** **The run is back after the restart.** The laptop restarted at 21:04 and the chain was relaunched at 01:26, so it sat idle
  for about 4 hours 20 minutes and lost the first circuit, which was 11 of 14 passes in. Finished stages were skipped in about 10
  seconds, which also proved `scripts/resume_full_run.ps1` works. Circuits are running faster than scaled (about 1 hour 20 minutes
  each, not 2), which wins back most of that time. Expected finish of each step: circuits about 09:30 today, repairs early Tuesday,
  Q4 Tuesday evening, NF4 Wednesday 11 November about 07:30, so the final report still lands on Wednesday.

## 7 November 2026
- **21:05** **The laptop is about to restart, and the run is ready for it.** Everything is pushed. After the restart,
  `scripts/resume_full_run.ps1` relaunches the chain, skipping finished stages, so only the running stage is lost: circuit 1 of 6,
  due to finish around 21:15. It takes about 1 hour 20 minutes, against the 2 hours I had scaled from the small run.
- **20:12** **Blind controls: Earth's Javanese result holds, but the regional pattern from 19:40 isn't confirmed.** All reads use the
  default preamble, and the readers are models, not people (Amendment 28).
  - **Second read of the same 120 Javanese answers:** 94% of labels match the first read. The gap is 13.3 points (95% CI 3.3 to 25.0)
    against 20.0 the first time, because this reader was stricter. Both put the gap in invalid answers: Earth's unsafe count is 1 of
    60 in both reads, and Global's is 2 and 3. All readers are the same model, so this shows consistency, not correctness.
  - **English control:** no gap (3.3 points, p = 0.63), with 1 of 60 invalid for each model. So the readers aren't marking Earth
    down in general, and the Javanese result isn't a reader habit.
  - **Fire and Water on the same 60 Javanese prompts:** Global 80.0% safe, Earth 60.0%, Fire 75.0%, Water 76.7%. Fire's gap is 5.0
    points (95% CI -3.3 to 13.3), so this read can't confirm the 10.8 points the judges show, and it can't rule them out either.
    Fire and Water were read in a separate batch from Global and Earth.
  - **Bengali, all four models, partial:** Global 87.8% safe, Earth 82.2%, Fire 82.6%, Water 81.2%, on 41 to 48 answers each. No gap is
    clear on about 30 pairs. Water's 9.7 points on the judges' numbers is mostly disputed answers, and the readers found only 2.9.
  - **Correction:** at 19:40 I said the drop sits outside each model's own region. That holds on the judges' numbers only. The blind
    reads are too small to confirm it, so it stays an open question and the write-up says so. Earth's Javanese finding doesn't depend
    on it.
  - **Missing labels:** a safety filter stopped one reader's write, so 60 of the 240 Bengali answers have no labels. That reader
    suggested re-running in smaller slices to avoid the filter. We didn't, because that would be working around a safety control.
    Bengali was an optional check, so it is scored with `--partial`.
  - **Also new:** readers described Earth's invalid answers mostly as off-topic or misreading the prompt, as for Global, plus
    repetition, which was nearly Earth-only (9 against 1). `docs/submission/sentences.json` holds the sentences that interpret the
    baseline, and `scripts/fill_submission.py` uses them only for this run's spec. `scripts/blind_audit.py` gained `--name`, `agree`
    and `--partial`, with tests that passed twice.
- **19:40** **Baseline result: Earth's Javanese gap is confirmed, but it is the model breaking down, not harmful content.** Global
  against Earth, full precision, held-out items, local judges, 14 tests at p below 0.0036. One test passes: Javanese with the
  preamble, 90.9% safe against 81.2%, a gap of 9.7 points (95% CI 4.0 to 15.3, p = 0.0023). Bengali with the preamble just misses
  (8.5 points, p = 0.0041), and so does Javanese without it (5.7 points, p = 0.031). The pilot pointed the other way, with only the
  no-preamble gap surviving. Judge agreement (AC1) is 0.82 or higher in every language, above the 0.6 floor. These are local
  judges with no human calibration, so `report.json` will still mark the result not validated.
  - **What the gap is made of:** Earth has 17 fewer safe Javanese answers with the preamble: 10 more degenerate, 7 more disputed,
    and no more that both judges call unsafe (1 of 176 for each model). The breakdown also shows on harmless prompts, where 7.5%
    of Earth's Javanese answers are degenerate against 0.7% of Global's.
  - **Blind read (Amendment 16):** four model readers labelled 240 random answers, without seeing the judges' labels or which
    model wrote them. With the preamble they found Global 80.0% safe and Earth 60.0%, a gap of 20.0 points (95% CI 8.3 to 31.7,
    p = 0.0042, 60 pairs). Without it, 88.3% and 75.0% (13.3 points, p = 0.039). The whole gap is invalid answers: 38.3% against
    16.7% with the preamble, while unsafe stays at 1 of 60 for Earth and 2 of 60 for Global in each condition. So the judges
    understate the drop: 34 of the 207 answers they called safe read as invalid, and 6 as unsafe.
  - **Disputed answers (Amendment 27, chosen after seeing the rows):** of Earth's 27, the reader found 11 safe, 3 unsafe and 13
    invalid. Of Global's 18, 8, 2 and 8. So disputes don't hide a rise in unsafe content either.
  - **Context, not a result:** Fire shows the same Javanese drop (10.8 and 6.8 points) and Water doesn't (2.3 and -0.6). In Bengali
    it flips: Water drops (9.7 and 10.2) and Fire doesn't (0.6 and -0.6). So on our reading of the regions, the drop sits outside
    each model's own region. These models are outside the 14 tests, and we saw this pattern after the numbers came in.
    **Corrected at 20:12: the blind reads don't confirm this pattern.**
  - **What it means:** the thesis says specialisation weakens refusals in the languages a model targets. The data so far says
    something else: outside its region, a regional model more often produces broken text, on harmful and harmless prompts alike.
    The readers are models, not people, so a native Javanese reader is what would settle this. The run carries on unchanged. A
    refusal-style edit may not help a model that loops, and the repairs will show that. The selected repair gets the same blind
    read. `scripts/blind_audit.py` has a `--disputed` mode, and it now has tests, which passed twice.
- **15:50** **Finish estimate, from this run's own timings.** Generation for all four models ended at 12:41 (75 minutes per
  model's held-out set), and judging runs at about 2 answers a second. Expected finish of each step: baseline tonight between
  19:30 and 21:00, circuits Sunday morning, repairs early Tuesday, Q4 Tuesday evening, NF4 early Wednesday 11 November, so about
  3.6 days from now. The final report and write-up would land on 11 November, four days before the deadline. Circuits are the
  loosest figure: they have never run at full size, so about 2 hours each is scaled up from the small run. Q4 and NF4 are scaled too.
- **11:35** **The article and cards now fill themselves from the run.** `scripts/fill_submission.py` writes filled copies of the
  article, model card, dataset card and ROI note from the run's own files, so about 1,300 numbers won't be typed by hand. It
  applies the article's ALT rule, and it leaves a value as `{{NAME}}` if the files don't hold it. An arm that meets every number
  without human calibration reads "numbers pass, not validated", not "no". `remaining.md` lists what is left for people. On the
  development run it filled 448 values, with no placeholder name wrong. The RUNBOOK still said the baseline step doesn't judge,
  which stopped being true at Amendment 20, so that section is corrected, and the manual judge commands are gone with it.
- **06:30** **The full run hit a speed problem in its first minutes, and it's fixed.** In `global-repair`, 48 of 70 batches ran at about
  191 ms per step, against 78 for the rest, split cleanly at 387 input tokens. Removing the 2 GiB fingerprint spike had removed a
  big reserved block that later allocations happened to reuse, so cached blocks piled up until VRAM spilled. The run now frees the
  cache between batches, which changes no output. `global-heldout` then ran at 72 to 79 ms per step on all 17 batches checked, at
  twice the tokens. Stages started from `c231daa`; `global-repair`, already running, finished on the old code. A model's held-out
  set takes about 88 minutes. Please don't run CPU-heavy jobs on the laptop: they roughly doubled generation time.
- **06:00** **The spec is frozen and the full Earth run is starting.** Spec hash `bdabb7edab9eead8`. Mentors hadn't commented on the
  thresholds, which the original rule asked for. Freezing later would restart a run of about four days, so we froze anyway, and any
  change a mentor asks for becomes a dated amendment (`SPEC.md` Amendment 26). Everything the audit raised was settled first:
  - **Speed:** on the longest prompts at full settings, batch 16 ran at 76 to 127 ms per step with no slowdown, peaking at 7.46 GiB.
    The margin to 8 GB is small, so speed gets watched early in the run. The audit's 95-hour estimate fits the time left.
  - **Circuit pairs:** Bengali had 6 of the 16 it needed, so pairs rose from 64 to 256. English now keeps 185, Javanese 79, Bengali 34.
  - **A real Javanese circuit** ran end to end for the first time, in 12 minutes at small size. It chose `Aku` from Global's own
    answers, parity held, all 612 nodes scored, and scores merged with English.
- **05:15** **A full-size data build found that the translator had been wrapping its output, in half of Javanese.** The 8-row
  development sample hid this. Of 250 Javanese benign outputs, 172 came with a label (`**Teks:**`), and most of those also carried a
  word-by-word explanation, so the median was 3.2 times the source length. Swahili (86) and Bengali (38) did it too. The translation
  inside was usually fine, so the build now takes it out of the wrapper, then rejects what still isn't a translation: answers,
  lists, repetition and broken characters. A prompt that ended in a question mark must keep one, except in Thai, where 156 real
  questions have none because the written language doesn't use it. Run over all 1,500 cached outputs, 1,432 pass. Javanese keeps
  241 of 250 with a median ratio of 1.06, and no labels remain. Nothing checks meaning: some accepted rows are simply wrong, such as
  Thai "child abuse" coming back as a question about fetuses. Native review stays a hard requirement.
- **The 661 ms per step slowdown has a cause.** In both Fire runs I checked, the first batch held a 965-token runaway translation. It
  overloaded memory, and the next batches stayed slow until they got small. The longest batch known to run fine is 616 tokens and
  the known-bad one 1,093, and the full run could reach about 744, so that range gets measured before the freeze. Tests running on
  the CPU also roughly doubled generation time, so they shouldn't run during measurements.
- **The comparison check passed.** `baseline-gap.json` has `earth`, `fire` and `water`, 14 tests each, on identical items and the right
  weights. The report's new section matches it exactly, and the public export kept all 28 cells after clearing 1,104 private rows.
- **04:10** **An independent audit of the full-run path found real problems before the freeze.** Each was checked against code and
  data here. Nothing about the full run has started yet.
  - **Bad translations were kept.** Global sometimes answers the prompt, or adds a word-by-word explanation, instead of translating:
    3 of 48 development rows, 2 of them among 8 in Javanese, at 8 to 22 times the source length against a median of 1.05. They now
    fail, by length or repetition. The cutoff is provisional until a full-size build shows the real distribution.
  - **Circuits were scoring tokens the models rarely say.** CircuitKIT reads only an answer's first token. English refusals open
    with `I'm` 69% of the time, and we scored `I` (2%). Javanese opens with `Aku` 58% of the time, and we scored the `N` of `Nuwun`
    (2%). `Sure` almost never appears. Circuits now use the tokens Global really opens with, read from its repair-split answers
    only, the same pair for both models.
  - **A 2 GiB memory spike.** The weight fingerprint widened the whole embedding before sampling it. That's fixed, and the
    fingerprint value is unchanged. One development run ran at 661 ms per step against 65 to 87 elsewhere. The cause isn't proven,
    so speed gets measured at full settings before the freeze.
  - **A data build with an empty cell now stops at once,** not days later at the NF4 check.
  - **One claim didn't hold.** The audit said a 510-token prompt would overflow the judges' 3,072-token slots. Measured with the
    judges' own tokenizers, that worst case is about 1,840. Only a 600-token runaway translation could overflow, and those now fail.
  - **Still to measure before freezing:** whether Javanese and Bengali circuit pairs can be built at all, since neither has ever run.
- **03:55** **NF4 passed on the Earth development run in 18 minutes,** the first real pass through that path on this laptop. All three
  NF4 models reached the report, next to fp16 and Q4_K_M, and none of their answers match fp16. The check also exposed a hole:
  answers never recorded their precision and NF4 shares fp16's weight fingerprint, so the "source must be full precision" test
  could never fail. Each answer now records it, and the NF4 output check requires it. The development NF4 answers predate this;
  the ledger and `validate_nf4_model` show they were 4-bit.
- **The report now has a Fire and Water section.** Rates, broken answers, gaps and judge agreement, marked as context, never as
  confirmed. `report.json` carries it as `comparison_models`, and the public export allows it, so the article's Water and Fire
  sentences now read from there. The comparison check is running in `runs/experiments/development/earth-compare`.
- **03:30** **The Fire end-to-end test finished at 03:07,** with no failures or retries. Like Earth's, its cells are too small to mean
  anything as results. The NF4 test on the Earth development run started at 03:26. `scripts/run_chain.py` has a new `e2e-baseline`
  step for the comparison check, which writes to its own folder.
- **02:00** **The Earth end-to-end test finished, the first pass through every stage.** It ran baselines, circuits, four repair arms,
  three 4-bit models and the report. Each cell holds about six harmful prompts, so its numbers show only that the pipeline
  works. It failed once, at 21:43: a local connection to the 4-bit model server dropped while the server sat idle, and that
  stage had no retry. Fixed (`a5fb754`), and it resumed from the same stage.
- **Water's 4-bit pilot shows no Javanese gap.** A planning signal only: 4-bit weights, repair split, local judges.

  | Javanese | Global | Fire | Earth | Water |
  |---|---|---|---|---|
  | Safe, with preamble | 87.0 | 80.9 | 77.1 | 92.4 |
  | Safe, no preamble | 91.6 | 80.9 | 74.8 | 89.3 |
  | Degenerate, with preamble | 6.1 | 16.0 | 8.4 | 3.8 |
  | Degenerate, no preamble | 7.6 | 21.4 | 19.8 | 6.9 |

  That fits a gap that follows region, because Water's card covers Asia Pacific (placing Javanese there is our reading). Water's
  Bengali dips instead: 85.5 and 84.7 against Global's 90.1 and 89.3, with degenerate answers at 5.3 and 7.6 against 0.0. Neither
  Water gap is significant (p = 0.17 to 0.51). Over all 60 comparisons (0.00083), only Earth's no-preamble Javanese gap survives.
- **Checks on the real run.** `scripts/export_public.py` blocked its first try because two broken answers repeat a public prompt
  exactly. With `--allow-prompt-echo` it passed, and a separate search of the export found no model-written text.
  `render_results.py --plots` works. `localisation_analysis.py` crashed on a Windows console; fixed. Capability at batch 8
  matches batch 1 on 239 of 240 items.
- **My slip.** My watcher only noticed failures, so I missed the moment to update the development config, and the Fire test
  started without the comparison models. That path is still untested, so it gets a short run after Fire.

## 6 November 2026
- **21:35** **Water's licence is accepted**, so its full weights are downloading and it joins the full run as a comparison on the
  same held-out items, alongside Fire (Amendment 21). Like Fire it is never repaired, and it stays outside the 14 tests. The
  Earth end-to-end test has run circuits and all four repair arms without a failure, and is judging held-out answers. Its
  numbers come from about two prompts per cell, so they mean nothing as results. It also started before the baseline-first
  code landed, so the Fire test will be the one that exercises that step. Selection messages said "refusal" while using the
  content axis; the messages were wrong, the selection wasn't.
- **19:45** **Corrections to what we logged earlier.** A check of the pilot numbers found several statements too strong:
  - **"Javanese is the robust gap" (5 Nov, 21:30) overstated it.** Both judges together call only 3 or 4 of Earth's 131
    Javanese answers `unsafe`. The rest of the gap is judge disputes (26 to 28 answers) and broken answers. Count an answer
    as safe when either judge says so, and Earth's gap falls from 9.9 to 2.3 points with the preamble and from 16.8 to 6.9
    without. Disputes do hide real unsafe answers: the blind readers found 24 of 68 Aya-safe, Qwen-not answers unsafe. So the
    true size is unknown until held-out data and the audit.
  - **"Fire's gap is real" (21:30) was too strong.** Counting all 40 pilot comparisons (0.00125), only Earth's no-preamble
    Javanese gap survives (p = 0.00006). Fire's (p = 0.0026) survives only the looser correction for 10 languages.
  - **"The report's gap reproduces" (5 Nov, 20:55) was too strong.** Our worst Global language is Javanese at 87.0; the report's
    is Arabic at 87. Our Arabic is 96.2, so matching numbers on different languages are a coincidence. Our means also run
    higher than the report's. What did reproduce is the pattern: a low worst-language score that averages hide.
  - **My first AC1 floor of 0.8 was a mistake.** It came from a pooled figure, and Earth's Javanese cells are 0.77 and 0.75.
    The floor is back to 0.6, the number fixed before any of this; the reasoning is in Amendment 16.
  - **Kappa of 0.30** was for safe against not safe. On all three labels it is 0.23.

  Other changes: baselines are now judged before any repair, so the gap test comes first. Fire joins the held-out run as a
  comparison. **Water's full weights are gated**, so it needs its licence accepted; the 4-bit control doesn't.
- **19:30** **A blind audit changed what we think the Javanese gap is.** Agreement between the judges looked terrible (kappa 0.30),
  which would have marked every language `not_measurable` under our own rule. So two model readers, blind to the judges'
  labels, read 153 pilot answers (repair split, oversampling disagreements). This is a model reading, not a human one.

  | Judges said | n | Reader: safe | unsafe | invalid |
  |---|---|---|---|---|
  | Both safe | 42 | 39 | 1 | 2 |
  | Aya safe, Qwen not | 68 | 28 | 24 | 16 |
  | Qwen safe, Aya not | 20 | 13 | 1 | 6 |
  | Neither safe | 23 | 3 | 3 | 17 |

  - **"Both safe" holds up.** Aya misses real unsafe content and Qwen over-flags, but requiring both still works as a
    conservative measure. The low kappa is mostly a statistical artefact: the judges agree on 95% of rows, and kappa collapses when
    nearly everything is safe.
  - **Most answers both judges reject aren't harmful: they're broken.** Repetition loops and prompt echoes, which the judges
    rarely call `invalid`. Without the preamble, 19.8% of Earth's Javanese answers are degenerate by a simple rule, against
    7.6% of Global's. Fire shows the same (21.4%).
  - **So part of the "safety gap" may be a language-competence gap.** That changes the story and what a repair must do: a
    refusal-style edit can't fix a model that loops. We now report unsafe and invalid separately. This is still pilot
    evidence; held-out data decides it.

  Changes, all made before any held-out data (SPEC.md, Amendments 15 to 18): a rule labels degenerate answers `invalid`; the
  content axis gates on Gwet's AC1 instead of kappa; the gap is tested in both preamble conditions, corrected for all 14
  tests; batch size goes from 2 to 16.
- **19:15** **Speed sorted.** Batch 16 runs 6.5x faster than batch 2 and fits the 8 GB GPU (145 against 22 tokens/s, 7.2 GiB), so the
  full run looks like about 3 to 4 days instead of over a week. llama-server on F16 only reached 169 tokens/s, not enough to
  justify a second backend.
- **19:00** The first end-to-end test exposed a real bug: every NF4 load failed its placement check, because Transformers 5 keeps no
  device map for a single-device load. That would have stopped the full run at the 4-bit stage. Fixed (`d4a833e`). Long jobs
  now run through `scripts/run_chain.py`, which survives a session ending. Earth's weights are in.

## 5 November 2026
- **21:30** **Full sweep done: Javanese is the robust gap.** Content-safe rate, both judges agree, 131 repair-split prompts per
  language, 4-bit, local judges in place of Command A. A planning signal, not a result.

  | Javanese | Global | Fire | Earth |
  |---|---|---|---|
  | With preamble | 87.0 | 80.9 (p = 0.15) | 77.1 (p = 0.04) |
  | Without preamble | 91.6 | 80.9 (p = 0.003) | 74.8 (p < 0.001) |

  Without the preamble both gaps survive correction for 10 languages (0.005). No other language does: Earth's Bengali gap
  shrinks from 7.6 to 3.1 points. Means stay close (95.0, 94.6, 93.5), which is why averages hide it. **Fire's gap is real
  too**, which corrects the 20:55 note. Javanese belongs to neither model's region, so next we test Tiny Aya Water, which
  covers it: if the region explains the loss, Water shouldn't show this gap. The end-to-end test on real models has started,
  using Fire until Earth's licence is accepted.
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
