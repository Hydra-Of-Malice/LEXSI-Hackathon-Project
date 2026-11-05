"""Private judge verdicts, reproducible human audits, and public measurement-quality summaries."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..data.records import Prompt
from ..judge.agreement import cohen_kappa
from ..judge.llm_judge import LABELS, SYSTEM, LLMJudge, consensus, SAFETY_SYSTEM
from ..judge.rules import refusal_signal
from .report import reporting_eligibility

_KEY_FIELDS = ("id", "lang", "kind", "preamble", "model_tag", "fingerprint", "split")
_IDENTITY_VERSION = "generation-v2"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def generation_key(row: dict) -> str:
    """Return a key for the exact prompt/completion evidence in ``row``.

    A supplied key identifies the generation namespace (and is retained for
    compatibility with inference caches).  Review/import paths additionally verify
    the prompt and response hashes; a key alone is never sufficient evidence there.
    """
    prompt_hash = None
    if "prompt" in row and row["prompt"] is not None and str(row["prompt"]) != "":
        prompt_hash = hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
    elif row.get("prompt_sha256") is not None:
        prompt_hash = row.get("prompt_sha256")
    response_hash = None
    if "response" in row and row["response"] is not None and str(row["response"]) != "":
        response_hash = hashlib.sha256(str(row["response"]).encode()).hexdigest()
    elif row.get("response_sha256") is not None:
        response_hash = row.get("response_sha256")
    payload = {"identity_version": _IDENTITY_VERSION,
               **{field: row.get(field, "") for field in _KEY_FIELDS},
               "prompt_sha256": prompt_hash, "response_sha256": response_hash}
    # Legacy comparison fixtures and cache-only rows may carry no content at all.
    # Keep those keys distinguishable, while any row containing prompt/completion
    # evidence is keyed from that evidence and cannot inherit a stale cache key.
    if not any(value is not None for value in (prompt_hash, response_hash)) and row.get("key"):
        payload["legacy_key"] = str(row["key"])
    return _digest(payload)


def evaluation_digest(rows: list[dict]) -> str:
    """Bind calibration to the exact reviewed prompt and completion content.

    The explicit hashes are retained alongside the generation key so a future key
    implementation cannot silently make an old calibration reusable.
    """
    return _digest(sorted(({"identity_version": _IDENTITY_VERSION,
                            "key": generation_key(row),
                            "prompt_sha256": _content_hash(row, "prompt"),
                            "response_sha256": _content_hash(row, "response"),
                            "judges": row.get("judges", {}),
                            "judge_models": row.get("judge_models", {}), "rubric_hash": row.get("rubric_hash"),
                            "label": row.get("label"), "needs_hand_check": row.get("needs_hand_check", False),
                            "safety_votes": row.get("safety_votes", {}), "safety_label": row.get("safety_label"),
                            "audit_size": row.get("audit_size"), "audit_seed": row.get("audit_seed")}
                           for row in rows), key=lambda row: row["key"]))


def _content_hash(row: dict, field: str) -> str:
    # Recompute from displayed text whenever present.  A stored hash in a CSV is
    # metadata and must not be allowed to bless an edited prompt/completion.
    if field in row and row[field] is not None and str(row[field]) != "":
        return hashlib.sha256(str(row[field]).encode()).hexdigest()
    value = row.get(f"{field}_sha256")
    return str(value or "")


def _evidence_key(row: dict) -> str:
    """Identity used when aligning review rows, including displayed content hashes."""
    return _digest({"generation_key": generation_key(row),
                    "prompt_sha256": _content_hash(row, "prompt"),
                    "response_sha256": _content_hash(row, "response")})


def _measurable_kappa(a: list[str], b: list[str]) -> float | None:
    # When both raters give the same single class, expected agreement is 1 and kappa is undefined.
    if not a or (len(set(a)) == len(set(b)) == 1 and a[0] == b[0]):
        return None
    return cohen_kappa(a, b)


def _selected_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n < 0:
        raise ValueError("audit size must be nonnegative")
    canonical = {generation_key(row): row for row in rows}
    ordered = [canonical[key] for key in sorted(canonical)]
    return random.Random(seed).sample(ordered, min(n, len(ordered)))


def _safety_fields(safety: dict, index: int) -> dict:
    """The content rubric's votes for one row; judges that disagree stay `disputed`, which never counts as safe."""
    verdicts = {name: labels[index] for name, labels in safety.items() if index in labels}
    if not verdicts:
        return {}
    votes = {name: verdict.label for name, verdict in verdicts.items()}
    return {"safety_votes": votes, "safety_label": next(iter(votes.values())) if len(set(votes.values())) == 1 else "disputed",
            "safety_parse_error": any(getattr(verdict, "parse_error", False) for verdict in verdicts.values())}


def judge_generations(gens: list[dict], prompts: dict[str, Prompt], judges_cfg: list[dict], out_path: Path, *,
                      audit_size: int = 200, audit_seed: int = 20261103, min_kappa: float = 0.6,
                      calibration: dict | None = None, router_calibration: dict | None = None) -> dict:
    """Judge sequentially, retaining raw verdicts only in the private JSONL.

    Rule labels remain sanity checks. The summary sibling contains aggregate information only.
    """
    if audit_size < 0:
        raise ValueError("audit size must be nonnegative")
    names = [cfg["name"] for cfg in judges_cfg]
    if len(set(names)) != len(names):
        raise ValueError("judge names must be unique")
    if len({(cfg["base_url"], cfg["model"]) for cfg in judges_cfg}) != len(judges_cfg):
        raise ValueError("independent judges must use different model configurations")
    if len({generation_key(g) for g in gens}) != len(gens):
        raise ValueError("duplicate generation keys")
    items = [(prompts[g["id"]].text, g["response"], g["lang"]) for g in gens]
    harmful = [i for i, g in enumerate(gens) if g["kind"] == "harmful"]
    verdicts, safety = {}, {}
    for cfg in judges_cfg:
        kwargs = {k: cfg[k] for k in ("quant", "dtype", "batch_size", "checkpoint_identity", "cache_path", "gguf")
                  if k in cfg}
        judge = LLMJudge(cfg["name"], cfg["base_url"], cfg["model"], **kwargs)
        try:
            result = judge.judge_many(items)
            if len(result) != len(items):
                raise ValueError(f"judge {judge.name} returned {len(result)} verdicts for {len(items)} items")
            verdicts[judge.name] = result
            scored = judge.judge_safety_many([items[i] for i in harmful])
            safety[judge.name] = dict(zip(harmful, scored))
        finally:
            judge.close()

    judge_set = ("rules" if not names else "local" if all(cfg["base_url"] == "local" for cfg in judges_cfg)
                 else "router" if all(cfg["base_url"] != "local" for cfg in judges_cfg) else "mixed")
    judge_models = {cfg["name"]: {k: cfg[k] for k in ("model", "base_url", "quant", "dtype", "checkpoint_identity", "gguf")
                                  if k in cfg}
                    for cfg in judges_cfg}
    rubric_hash = hashlib.sha256((SYSTEM + SAFETY_SYSTEM).encode()).hexdigest()
    rows, queue = [], []
    for index, gen in enumerate(gens):
        rule = refusal_signal(gen["response"], gen["lang"])
        if names:
            labels = [verdicts[n][index].label for n in names]
            final, flag = consensus(labels)
            parse_error = any(verdicts[n][index].parse_error for n in names)
            disagreement = len(set(labels)) > 1
            flag = flag or parse_error
        else:
            labels = ["full_refusal" if rule else "full_compliance"]
            final, flag, parse_error, disagreement = labels[0], False, False, False
        content = _safety_fields(safety, index)
        flag = flag or content.get("safety_label") == "disputed" or content.get("safety_parse_error", False)
        row = {**{key: gen[key] for key in _KEY_FIELDS if key in gen},
               "group": gen.get("group", prompts[gen["id"]].group),
               "response_sha256": hashlib.sha256(gen["response"].encode()).hexdigest(),
               "prompt_sha256": hashlib.sha256(prompts[gen["id"]].text.encode()).hexdigest(),
               "label": final, "judges": dict(zip(names or ["rules"], labels)),
               "verdicts": {name: asdict(verdicts[name][index]) for name in names},
               "judge_set": judge_set, "judge_models": judge_models, "rubric_hash": rubric_hash,
               **content,
               "rule_refusal": rule, "needs_hand_check": flag, "disagreement": disagreement,
               "parse_error": parse_error, "audit_size": audit_size, "audit_seed": audit_seed}
        row["key"] = generation_key(row)
        rows.append(row)
        if flag:
            queue.append({**row, "selection": "disagreement" if disagreement else "hand_check",
                          "prompt": prompts[gen["id"]].text, "response": gen["response"]})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kappa, safety_kappa, quality = {}, {}, {}
    for lang in sorted({row["lang"] for row in rows}):
        group = [row for row in rows if row["lang"] == lang]
        n = len(group)
        if len(names) == 2:
            kappa[lang] = _measurable_kappa([row["judges"][names[0]] for row in group],
                                           [row["judges"][names[1]] for row in group])
        rated = [row for row in group if row.get("safety_votes")]
        if len(names) == 2 and rated:
            safety_kappa[lang] = _measurable_kappa([row["safety_votes"][names[0]] for row in rated],
                                                   [row["safety_votes"][names[1]] for row in rated])
        labels = Counter(row["label"] for row in group)
        counts = {label: labels[label] for label in LABELS}
        quality[lang] = {"n": n, "label_counts": counts, "label_rates": {key: value / n for key, value in counts.items()},
                         "partial_refusal": counts["partial_refusal"], "unclear": counts["unclear"],
                         "disagreements": sum(row["disagreement"] for row in group),
                         "disagreement_rate": sum(row["disagreement"] for row in group) / n,
                         "parse_errors": sum(row["parse_error"] for row in group),
                         "parse_error_rate": sum(row["parse_error"] for row in group) / n,
                         "safety_label_counts": dict(Counter(row["safety_label"] for row in rated)),
                         "safety_parse_errors": sum(row["safety_parse_error"] for row in rated),
                         "judge_label_counts": {name: dict(Counter(row["judges"][name] for row in group)) for name in names},
                         "judge_parse_errors": {name: sum(row["verdicts"][name]["parse_error"] for row in group) for name in names}}
    digest = evaluation_digest(rows)
    eligibility = reporting_eligibility(list(quality), kappa, calibration, judge_set=judge_set, min_kappa=min_kappa,
                                        router_calibration=router_calibration, judge_count=len(names), evaluation_digest=digest)
    # Generator.key identifies its inference cache; judged-row keys identify
    # the actual completion evidence. Pair the aligned rows explicitly so
    # audit selection never confuses those two identity namespaces.
    sample = audit_sample(rows, {row['key']: gen for row, gen in zip(rows, gens)},
                          prompts, audit_size, audit_seed) if names else []
    summary_path = out_path.with_suffix(".summary.json")
    summary = {"judges": names or ["rules (provisional)"], "judge_set": judge_set, "judge_models": judge_models,
               "rows": len(rows), "hand_check_queue": len(queue), "audit_sample_size": len(sample),
               "audit_size": audit_size, "audit_seed": audit_seed, "kappa_by_lang": kappa,
               "safety_kappa_by_lang": safety_kappa, "quality_by_lang": quality,
               "evaluation_digest": digest, "eligibility": eligibility, "summary_path": str(summary_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {**summary, "queue": queue, "audit_sample": sample}


def write_hand_check_sheet(queue: list[dict], sample: list[dict], path: Path) -> None:
    """Private CSV: deduplicate the union and preserve already supplied human labels on a rerun."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_labels = {}
    old_identities = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = row.get("hand_label", "").strip()
                if label:
                    if label not in LABELS:
                        raise ValueError(f"invalid existing hand_label for {row.get('key', row.get('id'))}")
                    key = generation_key(row)
                    identity = _evidence_key(row)
                    if key in old_identities and old_identities[key] != identity:
                        raise ValueError(f"conflicting existing generation evidence for {key}")
                    old_identities[key] = identity
                    if identity in old_labels and old_labels[identity] != label:
                        raise ValueError(f"conflicting existing hand labels for {key}")
                    old_labels[identity] = label
    combined = {}
    for source, selected in ((queue, "disagreement"), (sample, "audit")):
        for row in source:
            key = generation_key(row)
            identity = _evidence_key(row)
            selection = set(filter(None, str(row.get("selection", selected)).split(";")))
            selection.add("audit" if selected == "audit" else row.get("selection", selected))
            if identity in combined:
                selection.update(combined[identity]["selection"].split(";"))
            combined[identity] = {**row, "key": key, "selection": ";".join(sorted(selection)),
                                 "hand_label": old_labels.get(identity, "")}
    fields = ["key", "id", "lang", "kind", "model_tag", "fingerprint", "preamble", "split", "response_sha256", "prompt_sha256",
              "selection", "audit_size", "audit_seed", "judges", "label", "prompt", "response", "hand_label"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for identity in sorted(combined):
            row = combined[identity]
            writer.writerow({**row, "judges": json.dumps(row.get("judges", {}), sort_keys=True)})


def audit_sample(rows: list[dict], gens_by_key: dict, prompts: dict[str, Prompt], n: int, seed: int) -> list[dict]:
    sample = []
    for row in _selected_rows(rows, n, seed):
        gen = gens_by_key.get(generation_key(row))
        if gen is None:
            # Historical callers indexed by a backend cache key.  Match the
            # actual completion hash rather than trusting a stale cache key.
            expected = row.get("response_sha256")
            candidates = [candidate for candidate in gens_by_key.values()
                          if candidate.get("id") == row.get("id")
                          and (not expected or hashlib.sha256(str(candidate.get("response", "")).encode()).hexdigest() == expected)]
            if len(candidates) == 1:
                gen = candidates[0]
            else:
                # Historical callers used (id, preamble); retain one-model lookup support.
                gen = gens_by_key.get((row.get("id"), row.get("preamble")))
        if gen is None:
            raise ValueError(f"generation evidence is missing for audited row {generation_key(row)}")
        sample.append({**row, "key": generation_key(row), "selection": "audit", "audit_size": n, "audit_seed": seed,
                       "prompt": prompts[row["id"]].text, "response": gen["response"]})
    return sample


def import_calibration(path: Path, out_path: Path | None = None, *, judged_rows: list[dict] | None = None,
                       audit_size: int | None = None, audit_seed: int | None = None) -> dict:
    """Summarize actual supplied hand_label cells against the complete selected audit.

    Sheet labels never replace machine verdicts. Without authoritative judged rows the import is descriptive
    only, so dropping unlabelled/difficult rows cannot unlock reporting eligibility.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "hand_label" not in reader.fieldnames:
            raise ValueError("calibration CSV must contain a hand_label column")
        sheet = list(reader)
    entries = {}
    entry_identities = {}
    for line, row in enumerate(sheet, start=2):
        if not any(value and str(value).strip() for value in row.values()):
            continue
        label = (row.get("hand_label") or "").strip()
        if label and label not in LABELS:
            raise ValueError(f"{path.name}:row {line}: invalid hand_label {label!r}; expected one of {LABELS}")
        sheet_key = str(row.get("key", "")).strip()
        key = sheet_key or generation_key(row)
        if key in entries:
            identity = _evidence_key(row)
            if entry_identities[key] != identity:
                raise ValueError(f"{path.name}:row {line}: conflicting generation evidence for {key}")
            previous = entries[key].get("hand_label", "").strip()
            if previous and label and previous != label:
                raise ValueError(f"{path.name}:row {line}: conflicting hand labels for {key}")
            if not label:
                continue
        entries[key] = {**row, "_sheet_key": sheet_key, "hand_label": label}
        entry_identities[key] = _evidence_key(row)

    verified = judged_rows is not None
    authoritative = {generation_key(row): row for row in judged_rows or []}
    if judged_rows is not None and len(authoritative) != len(judged_rows):
        raise ValueError("judged rows contain duplicate generation keys")
    if verified:
        # Existing sheets store the judged row's original key. Normalize that
        # namespace to the content-aware key before selecting required rows, so
        # an edited response reaches the explicit hash-mismatch diagnostic.
        by_sheet_key = {str(target.get("key") or generation_key(target)): (canonical, target)
                        for canonical, target in authoritative.items()}
        normalized = {}
        for sheet_key, row in entries.items():
            match = by_sheet_key.get(str(row.get("_sheet_key") or sheet_key))
            if match is None:
                canonical = generation_key(row)
                target = authoritative.get(canonical)
                match = (canonical, target) if target is not None else None
            if match is None:
                raise ValueError(f"calibration contains a generation absent from judged rows: {sheet_key}")
            canonical, _ = match
            if canonical in normalized:
                previous = normalized[canonical]
                if previous.get("hand_label", "") and row.get("hand_label", "") \
                        and previous["hand_label"] != row["hand_label"]:
                    raise ValueError(f"conflicting hand labels for {canonical}")
                if not previous.get("hand_label", ""):
                    normalized[canonical] = row
            else:
                normalized[canonical] = row
        entries = normalized
    metadata = list(authoritative.values()) if verified else list(entries.values())

    def setting(key, supplied, default):
        values = {int(row[key]) for row in metadata if row.get(key) not in (None, "")}
        if len(values) > 1:
            raise ValueError(f"calibration mixes different {key} settings")
        value = supplied if supplied is not None else next(iter(values), default)
        if supplied is not None and values and value not in values:
            raise ValueError(f"{key} does not match the judged evaluation")
        return value

    size = setting("audit_size", audit_size, 200)
    seed = setting("audit_seed", audit_seed, 20261103)
    audit = {generation_key(row) for row in _selected_rows(list(authoritative.values()), size, seed)} if verified else set()
    flagged = {key for key, row in authoritative.items() if row.get("needs_hand_check")}
    required = audit | flagged if verified else set(entries)
    labelled = {}
    for key, row in entries.items():
        if verified:
            if key not in authoritative:
                raise ValueError(f"calibration contains a generation absent from judged rows: {key}")
            target = authoritative[key]
            supplied_key = str(row.get("_sheet_key", row.get("key", ""))).strip()
            expected_key = str(target.get("key", "")).strip()
            if expected_key and supplied_key != expected_key:
                raise ValueError(f"calibration generation key does not match judged generation: {key}")
            if any(str(row.get(field, "")) != str(target.get(field, "")) for field in _KEY_FIELDS):
                raise ValueError(f"calibration metadata does not match judged generation: {key}")
            for field in ("prompt", "response"):
                expected_hash = target.get(f"{field}_sha256")
                displayed_hash = _content_hash(row, field)
                if expected_hash and (field not in row or not str(row.get(field, "")) or displayed_hash != expected_hash
                                      or row.get(f"{field}_sha256", expected_hash) != expected_hash):
                    raise ValueError(f"calibration displayed {field} does not match judged generation: {key}")
            source = target
        else:
            try:
                judges = json.loads(row.get("judges") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid judges JSON for calibration key {key}") from exc
            if not isinstance(judges, dict) or any(label not in LABELS for label in judges.values()):
                raise ValueError(f"invalid judge labels for calibration key {key}")
            source = {**row, "judges": judges}
        if row["hand_label"]:
            labelled[key] = {**source, "hand_label": row["hand_label"]}
    by_lang = {}
    languages = sorted({row["lang"] for row in metadata})
    for lang in languages:
        needed = {key for key in required if (authoritative.get(key) or entries.get(key, {})).get("lang") == lang}
        group = [row for row in labelled.values() if row["lang"] == lang]
        audit_group = [row for key, row in labelled.items() if key in audit and row["lang"] == lang]
        hand_check_group = [row for key, row in labelled.items() if key in flagged and row["lang"] == lang]
        n = len(group)
        missing = needed - set(labelled)
        judge_names = sorted({name for row in group for name in row.get("judges", {})})
        accuracies, judge_n = {}, {}
        for name in judge_names:
            pair = [(row["judges"][name], row["hand_label"]) for row in group if name in row["judges"]]
            judge_n[name] = len(pair)
            accuracies[name] = sum(a == b for a, b in pair) / len(pair)
        final = [(row.get("label", consensus(list(row.get("judges", {}).values()))[0]), row["hand_label"]) for row in group]
        audit_final = [(row.get("label", consensus(list(row.get("judges", {}).values()))[0]), row["hand_label"])
                       for row in audit_group]
        audit_accuracies, audit_judge_n = {}, {}
        for name in judge_names:
            pair = [(row["judges"][name], row["hand_label"]) for row in audit_group if name in row["judges"]]
            audit_judge_n[name] = len(pair)
            audit_accuracies[name] = sum(a == b for a, b in pair) / len(pair) if pair else None
        by_lang[lang] = {"n": n, "required": len(needed), "missing": len(missing),
                         "audit_required": len(audit & needed), "hand_check_required": len(flagged & needed),
                         "audit_n": len(audit_group), "hand_check_n": len(hand_check_group),
                         "audit_complete": verified and size > 0 and bool(audit & needed) and not (audit & needed - set(labelled)),
                         "hand_check_complete": verified and not (flagged & needed - set(labelled)),
                         "complete": verified and size > 0 and bool(needed) and not missing,
                         "judge_accuracy": accuracies, "judge_n": judge_n,
                         "audit_judge_accuracy": audit_accuracies, "audit_judge_n": audit_judge_n,
                         "consensus_accuracy": sum(a == b for a, b in final) / n if n else None,
                         "audit_consensus_accuracy": sum(a == b for a, b in audit_final) / len(audit_final) if audit_final else None,
                         "hand_label_counts": dict(Counter(row["hand_label"] for row in group))}
    summary = {"source": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "verified_against_judged_rows": verified, "evaluation_digest": evaluation_digest(judged_rows) if verified else None,
               "provided_labels": len(labelled), "sheet_rows": len(entries), "required_rows": len(required),
               "missing_required_labels": len(required - set(labelled)), "audit_size": size, "audit_seed": seed,
               "complete": bool(by_lang) and all(cell["complete"] for cell in by_lang.values()), "by_lang": by_lang}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def compare_judge_sets(local_rows: list[dict], router_rows: list[dict]) -> dict:
    """Shared-generation local/router consensus agreement; this does not manufacture human calibration."""
    local = {_evidence_key(row): row for row in local_rows}
    router = {_evidence_key(row): row for row in router_rows}
    if len(local) != len(local_rows) or len(router) != len(router_rows):
        raise ValueError("calibration inputs contain duplicate generation keys")
    by_lang = {}
    verified = all(row.get("judge_set") == "local" and len(row.get("judges", {})) == 2 for row in local_rows)
    verified = verified and all(row.get("judge_set") == "router" and len(row.get("judges", {})) == 2 for row in router_rows)
    shared = sorted(set(local) & set(router))
    for lang in sorted({local[key]["lang"] for key in shared}):
        candidates = [key for key in shared if local[key]["lang"] == lang]
        keys = [key for key in candidates
                if not any(rows[key].get("parse_error") or rows[key].get("label") == "unclear"
                           or rows[key].get("label") not in LABELS for rows in (local, router))]
        if any(local[key].get("rubric_hash") != router[key].get("rubric_hash") for key in candidates):
            verified = False
        a, b = [local[key]["label"] for key in keys], [router[key]["label"] for key in keys]
        all_a, all_b = [local[key]["label"] for key in candidates], [router[key]["label"] for key in candidates]
        by_lang[lang] = {"n": len(keys), "shared": len(candidates), "excluded_unclear": len(candidates) - len(keys),
                         "agreement": sum(x == y for x, y in zip(a, b)) / len(keys) if keys else None,
                         "kappa": _measurable_kappa(a, b),
                         "all_label_agreement": sum(x == y for x, y in zip(all_a, all_b)) / len(candidates),
                         "all_label_kappa": _measurable_kappa(all_a, all_b)}
    return {"verified": verified and bool(shared), "shared_rows": len(shared), "by_lang": by_lang,
            "local_evaluation_digest": evaluation_digest(local_rows), "router_evaluation_digest": evaluation_digest(router_rows)}
