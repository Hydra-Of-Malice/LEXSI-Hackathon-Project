# Runbook: who runs what, where

We have three machines and no paid inference yet. Everything up to the final headline numbers runs for free.

| Machine | Good for | Not for |
|---|---|---|
| **Pallab's MacBook M1 Max, 64 GB** (main) | generation for every model and arm (MPS, fp16); the free local judge pair; translations; Global-MMLU-Lite; the CPU weight edits (ReSta, C-ΔΘ, merge-back: about 20 GB); GGUF Q4_K_M conversion and testing with llama.cpp on Metal; CircuitKIT EAP-IG (try MPS first) | NF4 (bitsandbytes needs CUDA) |
| **Kaggle T4** (free, 30 GPU h/week) | NF4 4-bit re-tests; CircuitKIT if it fails on MPS; a second copy of any run | long jobs over 12 h per session |
| **Aditya's laptop, RTX 4060 8 GB, 1.2 TB free** | the full grid (no disk limit), NF4 (it has CUDA), Q4_K_M with llama.cpp, judges as Q4 GGUF; CircuitKIT on the CPU (`LANGSAFE_CIRCUIT_DEVICE=cpu`) | CircuitKIT on the GPU (doesn't fit in 8 GB). Its connection is about 0.5 MB/s, so plan downloads first |
| **HF router** (when the $20 credits land) | the pre-registered 32B judges for headline numbers, plus a shared sample to measure agreement with the local judges | — |

## 1. Mac setup (once, about 30 minutes plus downloads)

```bash
# Tools: Python 3.12, git, llama.cpp (Metal build)
brew install python@3.12 git llama.cpp

git clone https://github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project.git
cd LEXSI-Hackathon-Project
python3.12 -m venv .venv && source .venv/bin/activate
LANGSAFE_PYTHON=.venv/bin/python bash scripts/install_kits.sh --repair
hf auth login                           # your own token (read access is enough); type it yourself
```

The installer preserves working Torch/Transformers versions. The repair profile installs the exact SafeTune
arithmetic and steering dependencies without CUDA-only bitsandbytes. If the generation environment has Hub 2.x,
keep CircuitKIT's Hub 1.x dependency in its own environment:

```bash
uv venv --python 3.12 .venv-kits
uv pip install --python .venv-kits/bin/python 'packaging' 'huggingface-hub>=1.5,<2'
LANGSAFE_PYTHON=.venv-kits/bin/python bash scripts/install_kits.sh --circuits
```

The experiment automatically uses `.venv-kits/bin/python` for circuits. Override it with
`LANGSAFE_CIRCUIT_PYTHON` if needed. The `--all` installer profile also works in a compatible fresh environment.
Do not run the two environments' model jobs together.

The install should end with a line like `... torch 2.x | cuda False`. On a Mac that's expected: we use MPS. Check it:

```bash
python -c "from langsafe.gen.model import best_device; print(best_device())"   # expect: mps
```

### Models

Accept access on each required model's separate page using the account that owns the token:
`CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-fire`, `CohereLabs/tiny-aya-base`, and
`CohereLabs/aya-expanse-8b`. Base is required by ReSta and the circuit weight edit. Earth is optional for a Fire
experiment. `Qwen/Qwen3-8B` is not gated. Tiny Aya Global performs translations; Expanse 8B is one of the local judges.

- **Downloaded with `hf download` / `huggingface-cli`**: they're already in the HF cache, so nothing else is needed.
- **Downloaded into a folder** (safetensors, with `config.json`): name the subfolders `tiny-aya-global`, `tiny-aya-fire`
  and so on, then `export LANGSAFE_MODELS=/path/to/that/folder`.
- **Downloaded through Ollama or LM Studio**: those are GGUF files. They're only useful for the 4-bit check, and the
  experiment itself needs the safetensors weights:
  `hf download CohereLabs/tiny-aya-global` (repeat for fire, earth, base; about 7 GB each).

Pre-download the original weights before evaluation:

```bash
.venv/bin/hf download CohereLabs/tiny-aya-global
.venv/bin/hf download CohereLabs/tiny-aya-fire
.venv/bin/hf download CohereLabs/tiny-aya-base
.venv/bin/hf download CohereLabs/aya-expanse-8b
.venv/bin/hf download Qwen/Qwen3-8B
```

Disk: about 20 GB for the three Tiny Aya originals, about 32 GB for the two judges, about 7 GB per weight-edited
checkpoint, and about 7 GB plus 2 GB for each FP16/Q4 GGUF pair. The full parameter grid needs substantially more
space than the development profile. The script does not delete checkpoints to free space.

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

## 4. Resumable end-to-end experiment

The converter needs a llama.cpp checkout in addition to Homebrew's binaries:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp cache/llama.cpp
uv pip install --python .venv/bin/python -r cache/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
mkdir -p cache/llama.cpp/build/bin
ln -s "$(command -v llama-quantize)" cache/llama.cpp/build/bin/llama-quantize

bash scripts/run_experiment.sh --family fire --profile development
```

The development config limits real data and uses one candidate per arm. The full config retains the proposed
thresholds, larger datasets, and parameter grid. Run it with `--profile full`; it writes to a different directory.
Add `--stop-after data`, `baseline`, `circuit`, `repairs`, or `quantization` to stop at a recorded boundary. Explicit
`--skip-circuit` / `--skip-quantization` runs record the omissions and cannot establish the complete hypothesis.

The experiment executes: group-safe data and translations; Global/Fire repair and held-out generations; capability
with and without the preamble; HF/TransformerLens parity and EAP-IG; ReSta, circuit edit, merge-back, and steering;
repair-split selection; held-out evaluation and two-judge consensus; AuditKIT; conversion and Q4 evaluation of
Global, Fire, and the selected weight repair; reports and ledger verification. Intermediates use atomic writes.
No paid endpoint is called with the default `--judges local`.

The GGUF converter preserves the checkpoint's generation EOS token, in addition to the rendered HF input tokens.
Tiny Aya's tokenizer calls token 6 EOS, but its generation configuration stops at token 3; following the tokenizer
alone generates past the answer. The conversion protocol is part of the resumable stage identity, so exports
created before this fix and their dependent stage records are invalidated.

```bash
.venv/bin/python -m langsafe.cli status --directory runs/experiments/development/fire
.venv/bin/python -m langsafe.cli --runs-dir runs/experiments/development/fire ledger-verify
```

`logs/` has per-stage logs, `experiment.json` records hashes, `current-stage.json` identifies the latest stage,
and `last-failure.json` explains an unsuccessful stage. Rerun the original experiment command to resume. A changed
checkpoint, dataset, score file, calibration import, or prompt condition invalidates its dependent cached work.
`capability/<tag>.json` is the default condition and `<tag>.none.json` is the stripped-preamble condition.
Capability records reject a different checkpoint, dataset, or configuration under an existing tag. When replacing
an export intentionally, preserve its old capability JSON files in a separate evidence directory before rerunning;
the controller then rebuilds those records. An ordinary interrupted run with unchanged settings resumes in place.

Render aggregate results into a self-contained interactive HTML page and PNG/SVG figures:

```bash
uv pip install --python .venv/bin/python -e '.[reporting]'
.venv/bin/python scripts/render_results.py runs/experiments/development/fire/report.json --plots
```

The renderer reads `report.json` only. `report.html` filters by metric, language, preamble, and precision;
`report-fp16.png` / `report-q4.png` compare Global, the regional original, and the repair selected on repair data.
The figures retain confidence intervals and the provisional development label. Capability is unavailable for
languages absent from Global-MMLU-Lite; missing cells are not filled with estimated values.

Plan storage before the larger run. Each original FP16 checkpoint here is about 6.24 GiB. This development
profile saves three weight checkpoints plus three FP16 GGUF exports and three Q4 files. The full grid retains
eight weight checkpoints in its separate directory, in addition to its exports, datasets, and caches. Keeping
both profiles can exceed the available disk even though one model at a time fits unified memory.
The full grid needs roughly 75 GiB for its weight checkpoints and GGUF exports alone, plus room for temporary
conversion files, data, and caches. An external disk can hold the isolated run without moving the original models:

```bash
LANGSAFE_RUNS_DIR=/Volumes/EXTERNAL/lexsi/full/fire bash scripts/run_experiment.sh --family fire --profile full
```

Replace `EXTERNAL` with a mounted volume that has enough free space. Preserve the development artifacts when
comparing the two profiles; they use different data budgets and cannot be combined into one benchmark.

### On Windows (Aditya's laptop)

`scripts/run_experiment.sh` assumes a POSIX venv, so call the CLI directly from PowerShell:

```bash
$env:LANGSAFE_CIRCUIT_DEVICE = 'cpu'
$env:PYTHONUTF8 = '1'
.venv\Scripts\python.exe -u -m langsafe.cli experiment --family fire --profile full
```

llama.cpp lives in `cache/llama.cpp`: the release's CUDA binaries unzipped into `build/bin`, next to a checkout of the
same tag for the converter. Local judges use `cache/gguf/*.gguf` when those files exist, and transformers otherwise.

### Pilot without a login

`python scripts/pilot_gap.py --regional fire` compares Cohere's official ungated 4-bit Global and Fire on the repair
split. It is a planning check, never a reported result.

## 5. Human and router calibration

The private `hand_check/<tag>.local.csv` sheets contain a deterministic random sample plus every disagreement.
Reviewers fill `hand_label` with `full_refusal`, `partial_refusal`, `full_compliance`, or `unclear`. Leave no required
item unlabelled. Actual supplied labels are required; model-generated labels do not count as human calibration.

```bash
.venv/bin/python -m langsafe.cli --spec configs/development.yaml --runs-dir runs/experiments/development/fire \
  calibrate --tag fire --judges local --sheet runs/experiments/development/fire/hand_check/fire.local.csv
```

The local judge pair needs agreement with the router pair on a shared sample before headline use. After inference
credits and router access are available, run `judge --tag <tag> --judges router`, followed by
`calibrate --tag <tag> --compare-router` in the same experiment directory. Rebuild the report or rerun the experiment
to include new calibration. Low or undefined per-language kappa, unreviewed translations, and missing evidence
remain visible. Development results never become a full-experiment success claim.

## 6. Interactive demonstration

```bash
.venv/bin/python -m langsafe.cli demo --model CohereLabs/tiny-aya-global --lang en
.venv/bin/python -m langsafe.cli demo --model CohereLabs/tiny-aya-fire --lang hi
# Substitute the selected checkpoint path recorded in selection.json:
.venv/bin/python -m langsafe.cli demo --model runs/experiments/development/fire/checkpoints/fire-resta-a0.1
```

Use `--prompt 'Explain photosynthesis in two sentences.'` for a single response, or enter prompts interactively.
Use `--steering <direction-directory> --strength <selected-value>` with Fire to demonstrate the steering arm.

## 7. CUDA/NF4 verification

NF4 cannot execute on this Mac. On a CUDA machine with the same checkpoints and experiment data, generate and
score `global-nf4`, `fire-nf4`, and `<selected-repair-tag>-nf4` using `generate --quant nf4` and
`capability --quant nf4 --preamble default` / `--preamble none`. Judge and audit each tag, then include all three
NF4 tags in `report --tags ...`. Do not compare a quantized repair against only FP16 baselines. The report blocks
the cross-precision acceptance claim until all configured precision conditions and calibration evidence exist.
