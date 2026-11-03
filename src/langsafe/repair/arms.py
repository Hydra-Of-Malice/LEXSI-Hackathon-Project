"""SafeTune repair arms. Weight-space arms run on CPU (about 20 GB of RAM for three fp16 models); steering runs on GPU.

Arms (configs/spec.yaml):
  resta        Fire + a * (Global - Base)                        whole model           SafeTune apply_resta
  circuit      Fire + s * (Global - Base), only CircuitKIT layers   circuit-restricted    SafeTune apply_ctheta
  merge_back   (1 - l) * Fire + l * Global                       naive baseline        SafeTune apply_wise_ft
  steer        add a refusal direction at one layer at run time   inference-time        SafeTune RefusalDirectionModel

`resta` and `circuit` use the same safety vector, so their difference isolates the effect of CircuitKIT's localisation.
Knobs (alpha, strength, lambda, steering strength) are chosen on the *repair* split only (see select.py), never on
held-out data. SafeTune's defaults (ReSta alpha=1, steering alpha=20) break Tiny Aya, so every knob is explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ROOT, load_spec
from ..provenance import Ledger

CKPT = ROOT / "runs" / "checkpoints"


def _load_cpu(name: str):
    import torch
    from safetune.runner.utils.model_utils import load_model_cpu

    return load_model_cpu(name, dtype=torch.float16)


def _save(model, tok_name: str, out: Path, method: str, inputs: list[str], params: dict) -> Path:
    from safetune.provenance import write_provenance
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(tok_name).save_pretrained(out)
    write_provenance(out, f"langsafe.{method}", inputs=[{"kind": "model", "ref": r} for r in inputs],
                     base_model=inputs[0], params=params)
    Ledger.default().append("repair", {"method": method, "out": str(out.relative_to(ROOT)), "inputs": inputs,
                                       "params": params})
    return out


def resta(drifted: str, alpha: float, family: str) -> Path:
    from safetune.recover import apply_resta

    m = load_spec()["models"]
    model = _load_cpu(drifted)
    apply_resta(model, base=_load_cpu(m["base"]), aligned=_load_cpu(m["reference"]), alpha=alpha, device="cpu")
    return _save(model, drifted, CKPT / f"{family}-resta-a{alpha}", "resta",
                 [drifted, m["base"], m["reference"]], {"alpha": alpha})


def circuit(drifted: str, scores_json: str, strength: float, family: str, top_fraction: float = 0.2) -> Path:
    """C-ΔΘ restricted to the layers/modules CircuitKIT ranked highest (its *_scores.json, read by SafeTune's adapter)."""
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.recover import apply_ctheta

    m = load_spec()["models"]
    info = load_circuit_info_from_file(scores_json, top_fraction=top_fraction)
    if info is None:
        raise ValueError(f"no usable circuit info in {scores_json}")
    model = _load_cpu(drifted)
    apply_ctheta(model, _load_cpu(m["reference"]), _load_cpu(m["base"]), info, strength=strength)
    params = {"strength": strength, "top_fraction": top_fraction, "scores": scores_json,
              "layer_subset": list(info.layer_suggestions.layer_subset or []),
              "target_modules": list(info.layer_suggestions.target_modules or [])}
    return _save(model, drifted, CKPT / f"{family}-circuit-s{strength}-top{top_fraction}", "ctheta",
                 [drifted, m["reference"], m["base"]], params)


def merge_back(drifted: str, lam: float, family: str) -> Path:
    from safetune.recover import apply_wise_ft

    m = load_spec()["models"]
    model = _load_cpu(drifted)
    apply_wise_ft(model, _load_cpu(m["reference"]), alpha=lam)
    return _save(model, drifted, CKPT / f"{family}-merge-l{lam}", "wise_ft", [drifted, m["reference"]], {"lambda": lam})


# ---- steering (inference time) ------------------------------------------------------------------------------------

def extract_direction(model, tok, harmful: list[str], harmless: list[str], family: str, pick_layer: int = 24) -> Path:
    """Refusal direction from repair-split prompts. pick_layer=24 of 36 follows SafeTune's Tiny Aya note."""
    import torch
    from safetune.steer import RefusalDirectionConfig, extract_refusal_direction

    direction, layer, _ = extract_refusal_direction(model, tok, harmful, harmless,
                                                    RefusalDirectionConfig(pick_layer=pick_layer))
    out = CKPT / f"{family}-steer"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction.detach().cpu(), "layer": int(layer)}, out / "refusal_direction.pt")
    (out / "meta.json").write_text(json.dumps({"layer": int(layer), "n_harmful": len(harmful),
                                               "n_harmless": len(harmless)}), encoding="utf-8")
    Ledger.default().append("steer-extract", {"family": family, "layer": int(layer)})
    return out


def steered(model, steer_dir: Path, strength: float):
    """Context manager: inside it, model.generate runs with the refusal direction added at the chosen layer."""
    import torch
    from safetune.steer import RefusalDirectionModel

    blob = torch.load(steer_dir / "refusal_direction.pt", weights_only=True)
    return RefusalDirectionModel(model, blob["direction"].to(model.device), mode="steer", strength=strength,
                                 layers=[blob["layer"]])
