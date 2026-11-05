# Submission checklist

As of 5 November 2026, commit `f09a5a1`. Sources: hackathon site and kickoff slides (`docs/research/01_official_brief.md`), Discord clarifications (`docs/research/07_discord_intel.md`), and a read of the code and repo state.

**Deadline: 14 November, midnight AoE. That is 15 November, 17:30 IST.** Late or incomplete submissions may be excluded. An internal team checks validity before any judge reads the work: public artifacts, write-up, checklist, no duplicates.

Status means one of three things:

- **done**: nothing is left, or nothing is needed.
- **waiting on results**: the structure exists, and it needs numbers from a run.
- **needs a person**: only the team can do it, or the team must decide.

## Do these first

| # | Action | Why now | Status |
|---|---|---|---|
| 1 | Confirm the team is registered, with both HF usernames matching the form | **Registration closes today, 5 November.** We have no record here that this was checked | needs a person |
| 2 | Decide the rule for choosing ALT-A or ALT-B, and log it in `SPEC.md` | It must be fixed before the sweep rerun finishes, or the choice can be accused of following the data. A proposed rule is at the top of `article.md` | needs a person |
| 3 | Settle the primary axis, and calibrate it | `configs/spec.yaml` sets content safety as primary, while `SPEC.md` Amendment 10 says that is still undecided. Kappa and hand labels cover the refusal rubric only. See "Risks", item 2 | needs a person |
| 4 | Decide what happens to the Fire grid if there is no gap | The selection score and acceptance gate both stop working without a gap. See "Risks", item 3 | needs a person |
| 5 | Break the freeze deadlock | The spec freezes after mentors comment, but the mentor brief is on hold because it states the gap as fact. Rewrite the brief, or freeze without mentor comments and say so | needs a person |
| 6 | Find native readers for Hindi/Hinglish, Bengali and Tamil | 1,766 translated rows and the hand-check sample need people. Our report blocks any success claim without them | needs a person |
| 7 | Redeem HF credits and check the balance | The 32B judges need them, and the local judges cannot be used for headlines without a shared-sample check against the 32B pair | needs a person |
| 8 | Give the go-ahead (or not) to make the repo public and publish to HF | Required by the rules. Sections below list what must happen first | needs a person |

## Rules

| # | Rule | Status | Where we stand | What is left |
|---|---|---|---|---|
| 1 | Teams of 1 to 3, one team per person, at least one member on Discord | needs a person | Team of two. Discord has been read by a team member | Confirm the registration entry (action 1) |
| 2 | Built during the hackathon, 1 to 14 November | needs a person | First commit is 3 November | Confirm that no Kavach code was reused. If any was, disclose it in `{{PRIOR_WORK_STATEMENT}}` |
| 3 | At least 2 repos chained as one pipeline, 3+ preferred | waiting on results | Four repos in code: CuratorKIT, AuditKIT, CircuitKIT, SafeTune. The development run exercised the chain | Confirm the final manifest's `curatorkit` field is filled. Pilots skip it, so both local manifests show `null`. AuditKIT records do not carry the content axis yet |
| 4 | Core model is one of the 7 listed | done | Tiny Aya Fire is the core. Global and Base are also listed. Qwen3 is judge only, which the organisers allow | Nothing |
| 5 | Models, adapters, logs, datasets and benchmarks public on HF | needs a person | Nothing is published | Team go-ahead, then see "Before anything goes on Hugging Face" |
| 6 | Public deployment by the deadline: a public demo or repo link | needs a person | **The GitHub repo is private as of 5 November.** A draft Gradio demo sits untracked in `space/` | **The repo must be public by the deadline. Publishing needs the team's go-ahead.** See "Before the repo goes public" |
| 7 | Before-and-after metrics, more than one | waiting on results | Three axes plus a secondary metric are designed and tested on the development run | Full run |
| 8 | HF article: metrics and why, reasoning for data, tasks and model, how the result is better in use | waiting on results | Draft in `article.md` | Fill placeholders, choose the story, publish |
| 9 | ROI comparison (optional, preferred) | waiting on results | `roi.md` has the formula, prices and arithmetic | Measure batched tokens per second. Re-check prices on the day |
| 10 | Originality; licences respected | needs a person | Dataset and model licences are checked against HF metadata. Kit licences read in full | Kavach question (rule 2). Questions for organisers below |
| 11 | Code of conduct | done | No action needed | Nothing |
| 12 | Teams keep IP; organisers may showcase | done | No action needed | Note that LSAL section 6 gives Lexsi broad rights over any PR we open |
| 13 | Organisers may change the rules | needs a person | The 9-item list was marked "complete details will be shared soon" | Watch email and Discord. The form arrives a couple of days before the deadline and may ask for more |

## The 9 article items

| # | Item | Status | Where we stand | What is left |
|---|---|---|---|---|
| 1 | Problem statement and approach | waiting on results | Written, with ALT passages | Choose the story |
| 2 | Hackathon repos used, for what, which features | waiting on results | Written from the code. Claims match what the pipeline calls | Two patches are drafted in `docs/upstream_patches/` (AuditKIT refusal scorer, SafeTune layer ranking). None is opened. Their author line is a placeholder. Fill PR links, or mark them "planned" |
| 3 | Models used and licences | needs a person | Written and checked | Say how far Earth ran. The sweep adds it from the 4-bit files. `docs/PROGRESS.md` still lists its licence as not accepted |
| 4 | Datasets and benchmarks, with licences | waiting on results | Written and checked | Final counts and translation review status |
| 5 | Model artifacts public as an HF collection | waiting on results | Card template in `model_card_template.md` | Selected checkpoint, then go-ahead, then publish |
| 6 | Data artifacts public as an HF collection | waiting on results | Card template in `dataset_card_template.md` | Final data, a sanitised log export, go-ahead, publish |
| 7 | Experiments, findings, what else was tested | waiting on results | Development run and planning checks written. Result tables ready | Fill section 7 |
| 8 | "With more credits": approach and before-and-after results | waiting on results | Plan written | Fill "before" values and the cost estimate |
| 9 | Teammates tagged | needs a person | Names are in. HF usernames are unknown to this draft | Add both usernames. GitHub handles may differ, so do not assume |

## Judged surfaces

| Item | Status | Where we stand | What is left |
|---|---|---|---|
| Write-up clarity (/10) | waiting on results | Follows the 9 items in order and stands alone | Read it cold once it is filled in. Delete every team note |
| Practical usefulness (/10) | waiting on results | A named kind of user, an audit recipe, ROI | Make sure section 7.9 matches the final story |
| Code and notebook quality (/10) | needs a person | Only `notebooks/00_day1_smoke_test.ipynb` exists. It is a smoke test | **Build a reproduction notebook.** The proposal promised one. Pin versions and seeds |
| Model cards, which judges see in their own UI | waiting on results | Template is complete | One card per published checkpoint |
| Track tag | needs a person | We target 2b, problem 2 | Under ALT-B, organisers may tag it 2a, problem 4. Say so in the article, or ask |
| Demo Day readiness (if shortlisted) | needs a person | Live demo, ideally two presenters. A draft Space exists in `space/` | Plan only if time allows |

## Our own science gates

The report refuses to claim success until these pass. They are separate from the rules, and a submission can be valid without them, provided we say so plainly.

| Gate | Status | What is left |
|---|---|---|
| 10-language sweep on the content rubric, with Earth, both judges, both preamble conditions | waiting on results | Running. It replaces the refusal-rubric sweep, which was stopped partway. No aggregate from it exists yet |
| Held-out confirmation of any gap | needs a person | **Not planned yet.** The sweep script uses repair-split items only, and its output is labelled "not a result". Held-out MultiJail items are still unused |
| Spec frozen | needs a person | `frozen: false`. See action 5 |
| Two judges on every cell | waiting on results | Local pair available. 32B pair needs credits |
| Judge kappa at or above 0.6 per language | waiting on results | Some languages fell below this in the development run. The gate reads refusal-rubric kappa only |
| Content rubric calibrated | needs a person | No kappa, hand labels or router comparison exist for `safe`, `unsafe`, `invalid`. The code computes none of these yet |
| Hand labels: 200 random items plus all disagreements | needs a person | Zero labels so far |
| Translation review: 1,766 rows | needs a person | Zero labels so far. Status `unreviewed` |
| Full Fire grid: five arms, random-layer control | waiting on results | On hold until the sweep is settled |
| Circuit runs on Global and Fire in en, hi, bn | waiting on results | CPU on the laptop, since it does not fit in 8 GB |
| Q4_K_M and NF4 re-tests | waiting on results | Both are required by `quantized_must_hold` |
| `scripts/localisation_analysis.py` on the full run | waiting on results | Script is in the repo, and it measures refusal only. It needs the content axis to match section 7.7 |
| Earth family, full grid | needs a person | The sweep covers Earth at 4-bit. A full Earth run needs the full weights and the licence. Optional |

## Before the repo goes public

The repo is private today, and all history goes public with it. No generation, judged-output or hand-check path has ever been committed, which is good. These items need a decision first.

- [ ] `docs/research/07_discord_intel.md` names other participants and gives a competitive read of their projects. Remove it or move it out.
- [ ] `docs/research/08_kavach_crossover.md` describes a private repo and a prior submission.
- [ ] Removing a file does not remove it from history. Decide whether to rewrite history or start a clean public repo.
- [ ] `README.md`, `SPEC.md` and `docs/brief/mentor_brief.pdf` state the 9 to 10 point gap and the thesis as the project's premise. `SPEC.md`'s metrics and thresholds still name refusal. Update them to match the final story.
- [ ] `README.md` lists "compare" under AuditKIT. `compare_runs` exists in `src/langsafe/eval/auditkit_bridge.py`, but nothing calls it. Either call it or drop the claim. `article.md` does not claim it.
- [ ] `README.md` says "all four repair arms" and quotes test counts from the development run. The spec now has five arms and a control.
- [ ] `runs/ledger.jsonl` is tracked, and ledger entries store the machine's hostname. Check it is acceptable.
- [ ] `docs/CUDA_EVALUATION.md` says some source-tree details should stay private. Check `docs/CUDA_HANDOFF.patch` against that.
- [ ] `space/` and `docs/upstream_patches/` are untracked. Decide whether they go in. `space/sample_report.json` is a fake fixture.
- [ ] Run a secret scan over the tree and history.
- [ ] `.gitignore` still blocks every output path. Confirm before the final push.

## Before anything goes on Hugging Face

- [ ] **Never upload `runs/auditkit/`.** AuditKIT's JSON saves full predictions, including `prompt` and `raw_output`. These are harmful completions.
- [ ] **Build a sanitised log export.** The rules ask for public logs, and no such exporter exists today. Keep hashes, settings, scores and aggregates, and drop all model text.
- [ ] Keep `runs/gens/`, `runs/judged/`, `runs/hand_check/`, `runs/calibration/` and all review CSVs private.
- [ ] Decide which losing arms to publish. The proposal said every arm. It also said no weights less safe than the originals. Those conflict if an arm lowers refusal in any language. Set one rule, and use it in all cards.
- [ ] Publish a checkpoint only after held-out re-evaluation. LSAL section 4 bars derivatives that were not repaired and re-evaluated from user-facing systems.
- [ ] **Demo.** The draft in `space/` meets our release rule by design: no text box, a fixed list of benign prompts, and no API. Before it goes public:
  - Replace the FAKE DATA fixture with a real `report.json`, and confirm the fixture cannot load on the public Space.
  - Add the content-safe rate. The Results explorer shows refusal and compliance only.
  - Review `report.json` first. Its README warns that it holds local file paths and per-item vectors.
  - Decide what "repaired" means on the page if ALT-B is the story. Side-by-side needs a real re-evaluated checkpoint.
  - Check it against LSAL section 4, since it loads original Fire in a user-facing page.
  - Get its Hindi, Bengali, Tamil and Arabic prompts reviewed. A repo link alone also meets the rule if time runs out.
- [ ] Keep "SafeTune", "CircuitKIT" and "Lexsi" out of repo names. LSAL section 2 limits those names to unmodified copies.
- [ ] Put CC-BY-NC 4.0, the Cohere policy, attribution and a "modified" notice on every model repo.
- [ ] Consider including the LSAL notices with released checkpoints. Section 1 asks for them on copies and derivatives of the Licensed Work. Whether our checkpoints count is unclear, and including them costs nothing.
- [ ] Do not gate the repos unless organisers agree. Validity needs public artifacts.
- [ ] Decide how much circuit detail to publish. The proposal flagged this as a dual-use question.
- [ ] Add steering vectors to the collection. They are optional but preferred.

## Risks of failing a rule, most serious first

1. **Repo or artifacts not public by the deadline.** This fails validity before any judge reads it. It depends on a team decision, plus the clean-up above.
2. **The primary axis is not validated.** Selection and the acceptance gates now run on the content-safe rate. But the kappa gate, the hand-check sheets and the router comparison all use refusal labels. So a language could pass the reporting gate with poor agreement on the label that decides the result. Both judges must say `safe`, so low agreement also pushes the rate down. Local judges stand in for Command A, and the pipeline has no check of them on this rubric. Fix this before any headline number.
3. **The grid cannot answer the question if there is no gap.** With no positive gap, `gap_closed` is undefined, so `empirical_repair_success` is false by construction. The selection score also gives no reward for safety gains, so it favours the least damaging knob. Under ALT-B the arms would be compared on a score that ignores their purpose. This needs a logged amendment before the grid runs.
4. **The target language may not be in the pipeline.** The Fire set is en, hi, bn, ta and Hinglish. The report's own reasoning suggests the loss may sit outside Fire's region, and on the stopped refusal sweep the largest Fire deficits were Vietnamese and Swahili. `languages.py` has 7 languages, and `MULTIJAIL_COLUMNS` has only en, bn, sw and ar. Adding one means code changes, benign translations and a capability check.
5. **Picking the largest gap favours noise.** Choosing the target as the largest repair-split gap selects for chance. The gap may shrink or vanish on held-out data. Report that openly.
6. **Harmful text leaking through logs.** See the AuditKIT item above. This would break our release rule, and possibly the Cohere policy.
7. **No reproduction notebook.** Code and notebook quality is a third of the sub-scores.
8. **Time.** Nine days remain. The grid is on hold, with human review not started. If it cannot finish, the honest fallback is the measurement story with a smaller grid, and the article can carry that.
9. **Machine-translated data from a model under test.** The proposal said "a stronger model". The config uses Tiny Aya Global. The article and card say so. The translator choice remains a real weakness.
10. **Thresholds tighter than the data can resolve.** One point of 400 questions is 4 questions. Report intervals and do not call a point estimate a pass without them.

## Questions for organisers

Discord advises using a ticket. We have no record that one was opened.

1. Is a public safety evaluation set (prompts and labels, including machine-translated harmful prompts made with Tiny Aya) acceptable under the Cohere Labs acceptable use policy?
2. Is Tiny Aya Global acceptable as the reference for drift in Fire?
3. If the reported gap does not reproduce, does a measurement-led submission stay in Track 2b, or should it move to 2a?
4. May Command A be used as a judge to replicate the report's method?
5. Does the checklist add anything beyond the 9 items?
6. Does publishing the checkpoints require including the LSAL notice?
7. What is the right way to meet the "public logs" rule when raw logs contain harmful completions?
