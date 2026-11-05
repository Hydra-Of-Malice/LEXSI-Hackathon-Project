"""Does the circuit map help? Compares each circuit arm with its whole-model twin and with its random-layer controls.

    python scripts/localisation_analysis.py runs/experiments/full/fire [--judges local]

The standard report compares every arm with the original model. The project's main question needs two other contrasts,
paired on the same prompts: circuit vs the same vector on the whole model, and circuit vs the same vector and knob on
random layers. Reads judged labels, capability records and edit statistics; writes aggregates only, to
localisation.json and localisation.md in the run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langsafe.eval.stats import mcnemar_exact, paired_diff_ci          # noqa: E402

TWINS = {"circuit": "resta", "circuit_merge": "merge_back"}


def judged(run: Path, tag: str, judges: str) -> dict:
    name = f"{tag}.jsonl" if judges == "router" else f"{tag}.{judges}.jsonl"
    with open(run / "judged" / name, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {(r["id"], r["preamble"]): r for r in rows}


def capability(run: Path, tag: str) -> dict:
    path = run / "capability" / f"{tag}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("languages", {}) if path.exists() else {}


def edit(descriptor: dict) -> dict:
    """The size and location of an edit, from the provenance saved with its checkpoint."""
    path = Path(descriptor.get("model", "")) / "lexsi_provenance.json"
    if not path.is_file():
        return {}
    params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
    return {key: params.get(key) for key in ("relative_edit_l2", "edited_tensors", "edited_layers", "layer_subset")}


def contrast(a: dict, b: dict, lang: str | None, kind: str, label: str, resamples: int) -> dict | None:
    """Paired change from a to b in the rate of `label`, over prompts both runs answered."""
    keys = sorted(k for k in set(a) & set(b) if a[k]["kind"] == kind and (lang is None or a[k]["lang"] == lang))
    if not keys:
        return None
    x = [int(a[k]["label"] == label) for k in keys]
    y = [int(b[k]["label"] == label) for k in keys]
    delta, low, high = paired_diff_ci(x, y, resamples)
    return {"delta": delta, "ci": [low, high], "p_mcnemar": mcnemar_exact(x, y), "n": len(keys)}


def capability_contrast(a: dict, b: dict, resamples: int) -> dict | None:
    x, y = [], []
    for lang in sorted(set(a) & set(b)):
        first, second = a[lang].get("per_item", []), b[lang].get("per_item", [])
        if first and len(first) == len(second):
            x += first
            y += second
    if not x:
        return None
    delta, low, high = paired_diff_ci(x, y, resamples)
    return {"delta": delta, "ci": [low, high], "p_mcnemar": mcnemar_exact(x, y), "n": len(x)}


def compare(run: Path, baseline: str, candidate: str, target: str, judges: str, resamples: int) -> dict:
    """What moving from `baseline` to `candidate` does on the three axes."""
    a, b = judged(run, baseline, judges), judged(run, candidate, judges)
    return {"baseline": baseline, "candidate": candidate,
            "target_harmful_refusal": contrast(a, b, target, "harmful", "full_refusal", resamples),
            "all_harmful_refusal": contrast(a, b, None, "harmful", "full_refusal", resamples),
            "over_refusal": contrast(a, b, None, "benign", "full_refusal", resamples),
            "unclear": contrast(a, b, None, "harmful", "unclear", resamples),
            "capability": capability_contrast(capability(run, baseline), capability(run, candidate), resamples)}


def analyse(run: Path, judges: str = "local", resamples: int = 10000) -> dict:
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    target = selection["worst_language"]
    arms = {name: arm["selected"] for name, arm in selection.get("arms", {}).items()}
    result = {"target_language": target, "target_reason": selection.get("target_reason"), "judges": judges,
              "status": "provisional until human and router calibration are complete",
              "edits": {name: {"knob": d["knob"], **edit(d)} for name, d in arms.items()},
              "whole_model_twin": {}, "random_control": {}}
    for circuit, twin in TWINS.items():
        if circuit in arms and twin in arms:
            result["whole_model_twin"][circuit] = compare(run, arms[twin]["tag"], arms[circuit]["tag"], target, judges,
                                                          resamples)
    for control in selection.get("controls", []):
        circuit = arms[control["arm"]]
        result["random_control"][control["tag"]] = {"seed": control["random_seed"], "edit": edit(control),
                                                    **compare(run, control["tag"], circuit["tag"], target, judges, resamples)}
    return result


def _pp(cell: dict | None) -> str:
    if not cell:
        return "n/a"
    return (f"{100 * cell['delta']:+.1f} pp ({100 * cell['ci'][0]:+.1f} to {100 * cell['ci'][1]:+.1f}), "
            f"p={cell['p_mcnemar']:.3f}, n={cell['n']}")


def render(result: dict) -> str:
    lines = ["# Does the circuit map help?", "", f"**{result['status'].capitalize()}.** Judges: `{result['judges']}`. "
             f"Target language: `{result['target_language']}` ({result['target_reason']}).", "",
             "Each row is the change from the baseline to the circuit arm, paired on the same prompts. Positive refusal "
             "and capability changes favour the circuit arm; positive over-refusal and unclear changes count against it.",
             "", "## Edit sizes", "", "| Arm | Knob | Relative L2 | Tensors | Layers |", "|---|---|---|---|---|"]
    for name, cell in result["edits"].items():
        l2 = cell.get("relative_edit_l2")
        layers = cell.get("edited_layers")
        lines.append(f"| {name} | {cell['knob']} | {f'{l2:.5f}' if l2 is not None else 'n/a'} | "
                     f"{cell.get('edited_tensors', 'n/a')} | {len(layers) if layers else 'n/a'} |")
    for title, section in (("Circuit vs whole-model twin", result["whole_model_twin"]),
                           ("Circuit vs random layers", result["random_control"])):
        lines += ["", f"## {title}", "",
                  "| Baseline → circuit arm | Target refusal | All harmful refusal | Over-refusal | Unclear | Capability |",
                  "|---|---|---|---|---|---|"]
        for cell in section.values():
            lines.append(f"| {cell['baseline']} → {cell['candidate']} | {_pp(cell['target_harmful_refusal'])} | "
                         f"{_pp(cell['all_harmful_refusal'])} | {_pp(cell['over_refusal'])} | {_pp(cell['unclear'])} | "
                         f"{_pp(cell['capability'])} |")
        if not section:
            lines.append("| none available | | | | | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--judges", default="local")
    args = parser.parse_args()
    result = analyse(args.run, args.judges)
    (args.run / "localisation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.run / "localisation.md").write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
