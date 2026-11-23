"""Synthetic aligned dataset for tests and dry runs. Not real data; never report results from it."""
from __future__ import annotations

import random

import pandas as pd


def make_synthetic(cfg: dict, n: int = 120) -> pd.DataFrame:
    rnd = random.Random(0)
    subjects = ["anatomy", "virology", "philosophy", "nutrition"]
    rows = []
    for i in range(n):
        gold = rnd.randrange(4)
        subj = subjects[i % len(subjects)]
        for v in cfg["variants"]:
            tag = " ##DIALECT" if v not in ("ENG", "MSA") else ""
            rows.append(dict(qid=f"q{i}", variant=v, subject=subj,
                             question=f"synthetic question {i} [{v}]{tag} ##ANS={'ABCD'[gold]}",
                             choices=["opt1", "opt2", "opt3", "opt4"], answer=gold))
    return pd.DataFrame(rows)
