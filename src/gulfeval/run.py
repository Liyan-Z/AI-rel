"""Run a model over every (question, variant, repeat) and cache each prediction so runs can resume."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .backends import make_backend, with_retries
from .parse import parse_choice
from .prompts import build_prompt


def run_dir(cfg: dict) -> Path:
    p = Path(cfg.get("results_root", "results")) / cfg["run_name"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(r: dict) -> tuple:
    return (r["qid"], r["variant"], r["repeat"])


def run_experiment(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = run_dir(cfg) / "predictions.jsonl"
    done: dict[tuple, dict] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[_key(r)] = r
    backend = make_backend(cfg)
    labels = cfg["prompt"]["option_labels"]
    lang = cfg["prompt"]["instruction_language"]
    jobs = []
    for rep in range(cfg["run"]["repeats"]):
        for row in df.itertuples():
            k = (row.qid, row.variant, rep)
            if k not in done:
                jobs.append((row, rep))

    def work(job):
        row, rep = job
        lab = labels[: len(row.choices)]
        prompt = build_prompt(row.question, row.choices, lab, lang)
        raw = with_retries(lambda: backend.complete(prompt))
        pred = parse_choice(raw, lab)
        gold = lab[row.answer]
        return dict(qid=row.qid, variant=row.variant, repeat=rep, subject=row.subject, gold=gold,
                    pred=pred, raw=raw[:200], model=cfg["model"]["name"])

    with open(out, "a") as f, ThreadPoolExecutor(cfg["run"]["workers"]) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[_key(r)] = r
            f.flush()  # every prediction is on disk immediately: a crash loses at most one
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} new predictions")
    return pd.DataFrame(done.values())
