"""CircuitKIT stage: EAP-IG refusal-circuit discovery per (model, language), faithfulness, and cross-language overlap.

Needs a 16 GB GPU (Colab/Kaggle T4); an 8 GB card does not fit Tiny Aya under TransformerLens.
CircuitKIT notes that matter here:
- `discover` returns the nodes to PRUNE; the circuit is the top-scoring nodes in the *_scores.json it writes;
- reload the model between runs (discovery mutates model.cfg flags);
- the custom-task cache is keyed by task name, so the name includes model, language and pair-file hash;
- only the Tiny Aya base checkpoint is parity-tested upstream, so we run the parity check on Global and Fire first.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import ROOT
from ..provenance import Ledger

OUT = ROOT / "runs" / "circuits"
PILLARS = ["patching", "ablation", "baselines", "stability"]   # robustness pillar is not meaningful for signed metrics


def discover(model_id: str, model_tag: str, lang: str, task_csv: Path, n_examples: int = 64, ig_steps: int = 3,
             precision: str = "float16", sparsity: float = 0.3, seed: int = 0) -> dict:
    import circuitkit  # noqa: F401  (must be imported before transformer_lens)
    from circuitkit.pipeline import Pipeline

    digest = hashlib.sha256(task_csv.read_bytes()).hexdigest()[:8]
    task_name = f"refusal_{lang}__{model_tag}__{digest}"
    out_dir = OUT / f"{model_tag}__{lang}"
    pipe = Pipeline.from_custom_data(
        model_id, str(task_csv), clean_prompt="{harmful}", clean_answer="{refuse}",
        corrupt_prompt="{harmless}", corrupt_answer="{comply}", task_name=task_name,
        precision=precision, output_dir=str(out_dir))
    pipe.discover(algorithm="eap-ig", level="node", sparsity=sparsity, n_examples=n_examples, batch_size=1,
                  ig_steps=ig_steps, seed=seed)
    report = pipe.evaluate(pillars=PILLARS, n_examples=n_examples, n_stability_runs=3)
    scores = sorted(out_dir.glob("*_scores.json"))
    res = {"model": model_id, "model_tag": model_tag, "lang": lang, "task": task_name,
           "scores_json": str(scores[-1].relative_to(ROOT)) if scores else None,
           "faithfulness": _plain(report)}
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    Ledger.default().append("circuit-discover", {k: res[k] for k in ("model", "lang", "task", "scores_json")})
    return res


def _plain(x):
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "__dict__"):
        return {k: _plain(v) for k, v in vars(x).items() if not k.startswith("_")}
    return x


# ---- overlap across languages and models -------------------------------------------------------------------------

def node_scores(scores_json: str | Path) -> dict[str, float]:
    return json.loads(Path(ROOT / scores_json if not Path(scores_json).is_absolute() else scores_json)
                      .read_text(encoding="utf-8"))["node_scores"]


def top_k(scores: dict[str, float], k: int) -> set[str]:
    return set(sorted(scores, key=scores.get, reverse=True)[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    import numpy as np

    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float("nan")
    ra = np.argsort(np.argsort([a[k] for k in keys]))
    rb = np.argsort(np.argsort([b[k] for k in keys]))
    return float(np.corrcoef(ra, rb)[0, 1])


def overlap_matrix(circuits: dict[str, str | Path], k: int = 50) -> dict:
    """circuits: {"global/hi": scores_json, "fire/hi": ...} -> pairwise top-k Jaccard and full-ranking Spearman."""
    s = {name: node_scores(p) for name, p in circuits.items()}
    names = sorted(s)
    return {"k": k,
            "jaccard": {a: {b: jaccard(top_k(s[a], k), top_k(s[b], k)) for b in names} for a in names},
            "spearman": {a: {b: spearman(s[a], s[b]) for b in names} for a in names}}


def merge_scores(paths: list[str | Path], out: Path, model: str) -> Path:
    """Average node scores across languages into one scores.json SafeTune can read (node_scores-only form)."""
    acc: dict[str, float] = {}
    for p in paths:
        for n, v in node_scores(p).items():
            acc[n] = acc.get(n, 0.0) + v / len(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "algorithm": "eap-ig", "level": "node", "node_scores": acc,
                               "metadata": {"merged_from": [str(p) for p in paths]}}, indent=2), encoding="utf-8")
    return out
