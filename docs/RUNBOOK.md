# Runbook: who runs what, where

We have three machines and no paid inference yet. Everything up to the final headline numbers runs for free.

| Machine | Good for | Not for |
|---|---|---|
| **Pallab's MacBook M1 Max, 64 GB** (main) | generation for every model and arm (MPS, fp16); the free local judge pair; translations; Global-MMLU-Lite; the CPU weight edits (ReSta, C-ΔΘ, merge-back: about 20 GB); GGUF Q4_K_M conversion and testing with llama.cpp on Metal; CircuitKIT EAP-IG (try MPS first) | NF4 (bitsandbytes needs CUDA) |
| **Kaggle T4** (free, 30 GPU h/week) | NF4 4-bit re-tests; CircuitKIT if it fails on MPS; a second copy of any run | long jobs over 12 h per session |
| **Aditya's laptop, RTX 4060 8 GB** | tests, code, small generation runs | CircuitKIT (doesn't fit in 8 GB) |
| **HF router** (when the $20 credits land) | the pre-registered 32B judges for headline numbers, plus a shared sample to measure agreement with the local judges | — |

## 1. Mac setup (once, about 30 minutes plus downloads)

```bash
# Tools: Python 3.12, git, llama.cpp (Metal build)
brew install python@3.12 git llama.cpp

git clone https://github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project.git
cd LEXSI-Hackathon-Project
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/install_kits.sh            # langsafe + SafeTune, CircuitKIT, AuditKIT, CuratorKIT at pinned commits
hf auth login                           # your own token (read access is enough); type it yourself
```

The install should end with a line like `... torch 2.x | cuda False`. On a Mac that's expected: we use MPS. Check it:

```bash
python -c "from langsafe.gen.model import best_device; print(best_device())"   # expect: mps
```

### Models

Accept the licence once on each page (it's automatic): `CohereLabs/tiny-aya-global`, `-fire`, `-earth`, `-base`, and
`CohereLabs/aya-expanse-8b` (local judge and translator). `Qwen/Qwen3-8B` is not gated.

- **Downloaded with `hf download` / `huggingface-cli`**: they're already in the HF cache, so nothing else is needed.
- **Downloaded into a folder** (safetensors, with `config.json`): name the subfolders `tiny-aya-global`, `tiny-aya-fire`
  and so on, then `export LANGSAFE_MODELS=/path/to/that/folder`.
- **Downloaded through Ollama or LM Studio**: those are GGUF files. They're only useful for the 4-bit check, and the
  experiment itself needs the safetensors weights:
  `hf download CohereLabs/tiny-aya-global` (repeat for fire, earth, base; about 7 GB each).

Disk: about 30 GB for the four Tiny Aya models, plus about 32 GB for the two 8B judges.

## 2. Day-1 smoke test (about 1–2 hours on the M1 Max)

```bash
source .venv/bin/activate
JUDGES=local bash scripts/day1.sh
```

It builds the data, runs the tests, and generates with Global and Fire (20 prompts per language and kind, with and
without the preamble). It then judges twice: first with the instant rule detector, then with the local judge pair. It
prints the per-language tables, runs a tiny CircuitKIT EAP-IG on Global, checks that SafeTune can read the result, and
verifies the ledger. Everything goes into `runs/day1_report.md`.

**Pass:** no errors; Fire refuses less than Global in at least one Fire language; judge kappa is printed; CircuitKIT
writes a scores file.

If CircuitKIT fails on MPS, run the rest with `SKIP_CIRCUIT=1 JUDGES=local bash scripts/day1.sh` and we'll do circuits on
Kaggle. If an MPS op is missing, `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set by the script) runs it on CPU. As a last
resort, `LANGSAFE_DEVICE=cpu` forces CPU for everything.

**Send back:** commit `runs/day1_report.md` and `runs/ledger.jsonl` to a branch, or paste the report in the chat.
Never commit or share `runs/gens/`, `runs/hand_check/` or `runs/llm_cache.sqlite`. They contain model outputs on harmful
prompts, and `.gitignore` already blocks them.

## 3. Kaggle notebook

Upload `notebooks/00_day1_smoke_test.ipynb` (File → Import notebook). In the notebook settings, set Accelerator to
**GPU T4 x2** and turn **Internet on** (this needs a phone-verified account). Under Add-ons → Secrets, add `HF_TOKEN` and
`GH_TOKEN`. The GitHub token should be fine-grained, limited to this repo, with Contents set to read-only.

## 4. After day 1 (full pipeline; commands in README "Quick start")

1. `langsafe build-data --translate --translator local`: Tamil, Hinglish and Hindi MultiJail, plus non-English benign
   look-alikes, translated by Aya Expanse 8B on the Mac.
2. Full baseline: Global, Fire and Earth on all held-out prompts, both preamble conditions, then `judge --judges local`
   and `capability`.
3. Circuits per language and model, then the repair arms (knobs chosen on the repair split), then re-test.
4. 4-bit: GGUF Q4_K_M on the Mac (official `CohereLabs/tiny-aya-*-GGUF` for the unmodified models, plus our own
   conversion of each repaired checkpoint) and NF4 on Kaggle.
5. When credits arrive: `judge --judges router` on the final runs, and agreement between local and router judges on a
   shared sample.
