# Abstract

**Same model, different language, different safety: measuring per-language safety in Tiny Aya Earth, and testing
weight-space repairs with the Lexsi stack**

The Tiny Aya technical report puts Tiny Aya Earth 9.5 points below Tiny Aya Global in its worst MultiJail language,
but does not say which language. We asked whether that gap holds when measured language by language, with
full-precision weights, on held-out prompts, and whether the Lexsi stack can repair it without retraining.

We chained four Lexsi kits into one pipeline. CuratorKIT cleans and de-duplicates the prompts. CircuitKIT (EAP-IG)
locates the layers that drive refusal in English, Javanese and Bengali. SafeTune applies five repairs: ReSta, a
circuit-restricted C-ΔΘ edit, WiSE-FT merge-back, a circuit-restricted merge and refusal-direction steering. A
random-layer control is run beside them. AuditKIT records every run. We tested seven languages (Arabic, Swahili,
Javanese, Bengali, Thai, Korean, English) on 3,757 prompts from MultiJail, Aya red-teaming, XSTest look-alikes and
Global-MMLU-Lite. The primary metric is the report's own content-safe rate. Benign over-refusal, refusal, unsafe and
broken answers, and capability are reported beside it, never combined. Two local judges (Aya Expanse 8B, Qwen3-8B)
label every answer. We checked them against Aya Expanse 32B and Qwen3-32B (kappa 0.79 to 1.00) and against Command A,
the report's judge, on a 1,680-answer sample. The specification, thresholds and gap test were frozen before the full
run.

Findings:

- Only one gap passed our corrected test: Javanese with the safety preamble. It is almost all broken answers, not
  harmful content. Both judges called 1 of 176 answers unsafe for each model.
- Thai, the language the repairs targeted, showed 89.2% safe for Earth and 94.9% for Global (p = 0.0129, against a bar
  of 0.0036). The gap was not confirmed, so every repair result is exploratory.
- No repair met all pre-set conditions. Merge-back closed 90% of the Thai gap but raised over-refusal and cost
  capability. The circuit-located edit did no better than the same edit on random layers, although the circuits
  themselves were faithful: 0.90 to 1.01 on patching, against 0.21 to 0.69 for random heads.
- 4-bit quantisation (Q4_K_M and NF4) cost 2 to 4 safe points on average, and Earth's Javanese fell from 81.2% to
  about 71%, again through breakdown rather than harm.
- Command A was stricter than our judges: on the sample, Global minus Earth was 3.8 points with Command A and 1.9 with
  ours.

We release six model repos (four weight arms, the steering vector and a GGUF of the selected arm), an evaluation
dataset with reports, provenance logs and circuit summaries, the code with a reproduction notebook, and six pull requests
to the kits. No released file holds a model's answer to a harmful prompt. The main run used one laptop RTX 4060. The NF4 checks and one baseline ran on rented GPUs, for about
$15 of Hugging Face credits.

Team: Aditya Arnav (HydraOfMalice) and Pallab Das (PDGamerSG). Track 2b, problems 2, 4 and 5, and Track 2a, problem 4.
