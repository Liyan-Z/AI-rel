import pandas as pd
import pytest
import yaml

from gulfeval import data, stats
from gulfeval.parse import parse_choice
from gulfeval.synthetic import make_synthetic

CFG = yaml.safe_load(open("configs/default.yaml"))


@pytest.mark.parametrize("text,exp", [
    ("B", "B"), ("b.", "b".upper()), ("(C)", "C"), ("Answer: D", "D"), ("The answer is A", "A"),
    ("ج", "C"), ("أ", "A"), ("", None), ("I am not sure", None), ("A or B", None),
])
def test_parse(text, exp):
    assert parse_choice(text, list("ABCD")) == exp


def test_mcnemar_and_wilson():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(10, 10) == pytest.approx(1.0)
    assert stats.mcnemar_exact(0, 12) < 0.001
    lo, hi = stats.wilson(50, 100)
    assert lo < 0.5 < hi


def test_alignment_detects_mismatch():
    df = make_synthetic(CFG, n=10)
    assert data.validate_alignment(df).ok
    bad = df.copy()
    bad.loc[(bad.qid == "q0") & (bad.variant == "UAE"), "answer"] = 3
    bad.loc[(bad.qid == "q0") & (bad.variant == "MSA"), "answer"] = 0
    assert not data.validate_alignment(bad).ok


def test_health_subset():
    df = make_synthetic(CFG, n=40)
    sub = data.select_subset(df, CFG)
    assert set(sub.subject) == {"anatomy", "virology", "nutrition"}


def test_end_to_end_mock(tmp_path, monkeypatch):
    from gulfeval import report, run

    monkeypatch.chdir(tmp_path)
    cfg = yaml.safe_load(open(__file__.replace("tests/test_core.py", "configs/default.yaml")))
    cfg.update(run_name="t")
    cfg["run"].update(subset="all", repeats=2, workers=2)
    cfg["model"].update(temperature=0.7)
    df = make_synthetic(cfg, n=60)
    preds = run.run_experiment(cfg, df)
    assert len(preds) == 60 * len(cfg["variants"]) * 2
    # resume: a second call should add nothing
    again = run.run_experiment(cfg, df)
    assert len(again) == len(preds)
    text = report.make_report(preds, cfg, run.run_dir(cfg))
    assert "Gap vs MSA" in text
    gaps = stats.paired_vs_reference(preds, "MSA", 500)
    # the mock is built to do worse on dialects than on MSA
    assert (gaps[gaps.variant == "UAE"].gap_vs_ref.iloc[0]) < 0
