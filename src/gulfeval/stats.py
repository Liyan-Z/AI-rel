"""Statistics: accuracy with CIs, paired differences, McNemar, answer-flip rates and a sampling-noise baseline."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (only A right) and c (only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def paired_bootstrap(diff: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def first_repeat(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds[preds.repeat == 0].copy()
    p["correct"] = (p.pred == p.gold).astype(int)
    p["parsed"] = p.pred.notna().astype(int)
    return p


def accuracy_table(preds: pd.DataFrame) -> pd.DataFrame:
    p = first_repeat(preds)
    rows = []
    for v, g in p.groupby("variant"):
        k, n = int(g.correct.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(variant=v, n=n, accuracy=k / n, ci_low=lo, ci_high=hi,
                         parse_failure_rate=1 - g.parsed.mean()))
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def paired_vs_reference(preds: pd.DataFrame, ref: str, n_boot: int = 10000) -> pd.DataFrame:
    """For each variant: accuracy gap vs the reference, CI, McNemar p, and answer-flip rate."""
    p = first_repeat(preds)
    wide_c = p.pivot_table(index="qid", columns="variant", values="correct", aggfunc="first")
    wide_p = p.pivot_table(index="qid", columns="variant", values="pred", aggfunc="first")
    rows = []
    for v in wide_c.columns:
        if v == ref:
            continue
        ok = wide_c[[ref, v]].dropna().astype(int)
        diff = (ok[v] - ok[ref]).to_numpy()
        b = int(((ok[v] == 1) & (ok[ref] == 0)).sum())  # variant right, reference wrong
        c = int(((ok[v] == 0) & (ok[ref] == 1)).sum())  # reference right, variant wrong
        lo, hi = paired_bootstrap(diff, n_boot)
        both = wide_p[[ref, v]].dropna()
        flip = float((both[ref] != both[v]).mean())
        rows.append(dict(variant=v, n=len(ok), gap_vs_ref=float(diff.mean()), ci_low=lo, ci_high=hi,
                         mcnemar_p=mcnemar_exact(b, c), only_ref_right=c, only_variant_right=b,
                         answer_flip_rate=flip))
    return pd.DataFrame(rows).sort_values("gap_vs_ref").reset_index(drop=True)


def noise_baseline(preds: pd.DataFrame) -> pd.DataFrame | None:
    """Flip rate between repeat 0 and repeat 1 of the SAME variant: the floor for 'answers changed'."""
    if preds.repeat.max() < 1:
        return None
    a = preds[preds.repeat == 0].set_index(["qid", "variant"]).pred
    b = preds[preds.repeat == 1].set_index(["qid", "variant"]).pred
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["flip"] = j.a != j.b
    return j.groupby(level="variant").flip.mean().rename("repeat_flip_rate").reset_index()
