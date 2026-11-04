# Progress log

Newest entry first. Updated with every push, so anyone on the team can see what is running and who owns what. Pull
`main` before starting a run.

## Who is doing what

| Person / machine | Now | Next |
|---|---|---|
| Aditya's laptop (RTX 4060, 1.2 TB free) | Downloading the 4-bit models for the pilot | Full Fire run: baseline, circuits, repairs, 4-bit checks |
| Pallab's M1 Max | Free | Please pull `main` before restarting anything (see 4 Nov) |
| Aditya | HF login and the four model licences | Send the brief to mentors |

## Still needs a person
- Hugging Face login and licences on the laptop (blocking the full run).
- Native readers for Hindi/Hinglish, Bengali and Tamil, to review translations and hand-label a sample.
- HF credits, for the 32B judges that verify the free local judges.
- A go-ahead before anything is published to Hugging Face, and the final submission form.

## 4 November 2026
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
