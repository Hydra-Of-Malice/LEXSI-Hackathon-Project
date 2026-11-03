"""Inter-judge agreement (Cohen's kappa) and accuracy against hand labels."""
from __future__ import annotations

from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else float("nan")
