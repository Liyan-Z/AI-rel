"""Load DialectalArabicMMLU into one normalised table and validate cross-variant alignment."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

LETTERS = "ABCDEFGH"


def norm_subject(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s).strip().lower())


def _answer_to_index(a, n_choices: int) -> int:
    if isinstance(a, str):
        a = a.strip()
        if a.isdigit():
            return int(a)
        if len(a) == 1 and a.upper() in LETTERS:
            return LETTERS.index(a.upper())
        raise ValueError(f"Cannot interpret answer {a!r}")
    return int(a)


def _rows_from_table(df: pd.DataFrame, variant: str, cols: dict) -> list[dict]:
    rows = []
    for rec in df.to_dict("records"):
        if cols.get("choices"):
            choices = list(rec[cols["choices"]])
        else:
            choices = [rec[c] for c in cols["choices_columns"] if c in rec and rec[c] is not None
                       and str(rec[c]) != "nan"]
        idc = cols["id"] if isinstance(cols["id"], list) else [cols["id"]]
        rows.append(
            dict(
                qid="|".join(str(rec[c]) for c in idc),
                variant=variant,
                subject=str(rec.get(cols["subject"], "")),
                question=str(rec[cols["question"]]),
                choices=choices,
                answer=_answer_to_index(rec[cols["answer"]], len(choices)),
            )
        )
    return rows


def load_hf(cfg: dict) -> pd.DataFrame:
    """Download from Hugging Face and return the normalised table. Run `gulfeval inspect` first."""
    from datasets import load_dataset

    d, cols = cfg["dataset"], cfg["dataset"]["columns"]
    frames: list[pd.DataFrame] = []
    if d.get("variant_column"):
        ds = load_dataset(d["hf_name"], split=d["split"]).to_pandas()
        found = sorted(ds[d["variant_column"]].unique())
        print(f"variant values in dataset: {found}")
        missing = [v for v in cfg["variants"] if v not in found]
        if missing:
            raise SystemExit(f"Variants {missing} not found. Edit `variants` in the config to match: {found}")
        for v in cfg["variants"]:
            sub = ds[ds[d["variant_column"]] == v]
            frames.append(pd.DataFrame(_rows_from_table(sub, v, cols)))
    else:
        for v in cfg["variants"]:
            conf = d["variant_configs"][v]
            ds = load_dataset(d["hf_name"], conf, split=d["split"]).to_pandas()
            frames.append(pd.DataFrame(_rows_from_table(ds, v, cols)))
    return pd.concat(frames, ignore_index=True)


def inspect_hf(hf_name: str) -> str:
    """Print the dataset's configs, splits, columns and a sample row so config/default.yaml can be filled in."""
    from datasets import get_dataset_config_names, load_dataset

    out = []
    try:
        configs = get_dataset_config_names(hf_name)
    except Exception as e:  # noqa: BLE001
        configs = []
        out.append(f"Could not list configs: {e}")
    out.append(f"configs: {configs}")
    for c in (configs or [None])[:10]:
        try:
            ds = load_dataset(hf_name, c) if c else load_dataset(hf_name)
        except Exception as e:  # noqa: BLE001
            out.append(f"[{c}] load failed: {e}")
            continue
        for split, d in ds.items():
            out.append(f"[{c}] split={split} rows={len(d)} columns={d.column_names}")
            out.append(f"    sample: {d[0]}")
            break
    return "\n".join(out)


@dataclass
class AlignmentReport:
    n_questions: int
    missing: dict[str, int]
    answer_mismatch: int
    ok: bool

    def __str__(self) -> str:
        return (f"questions in common: {self.n_questions}; missing per variant: {self.missing}; "
                f"gold-answer disagreements across variants: {self.answer_mismatch}; ok={self.ok}")


def validate_alignment(df: pd.DataFrame) -> AlignmentReport:
    """Check every question id appears in every variant and that gold answers agree.

    If gold answers disagree for the same id, the ids probably do NOT identify the same question
    (or options were shuffled per variant) and paired comparisons would be invalid.
    """
    variants = sorted(df.variant.unique())
    ids = {v: set(df[df.variant == v].qid) for v in variants}
    common = set.intersection(*ids.values()) if ids else set()
    missing = {v: len(set.union(*ids.values()) - ids[v]) for v in variants}
    piv = df[df.qid.isin(common)].pivot_table(index="qid", columns="variant", values="answer", aggfunc="first")
    mismatch = int((piv.nunique(axis=1) > 1).sum())
    return AlignmentReport(len(common), missing, mismatch, ok=(mismatch == 0 and len(common) > 0))


def select_subset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only questions present in every variant, then apply the health/all filter and optional limit."""
    variants = cfg["variants"]
    common = set.intersection(*[set(df[df.variant == v].qid) for v in variants])
    df = df[df.qid.isin(common)]
    if cfg["run"]["subset"] == "health":
        wanted = {norm_subject(s) for s in cfg["health_subjects"]}
        df = df[df.subject.map(norm_subject).isin(wanted)]
        if df.empty:
            raise ValueError("Health filter matched nothing; check subject names with `gulfeval inspect`.")
    lim = cfg["run"].get("limit_per_variant")
    if lim:
        keep = sorted(df.qid.unique())[: int(lim)]
        df = df[df.qid.isin(keep)]
    return df.reset_index(drop=True)
