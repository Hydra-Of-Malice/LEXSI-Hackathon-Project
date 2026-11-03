"""Load the pre-registered spec (configs/spec.yaml)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs" / "spec.yaml"


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_digest(spec: dict[str, Any]) -> str:
    """Stable hash of the spec, stamped on every run so results can be traced to the exact settings used."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
