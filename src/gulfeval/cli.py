from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import data, report, run


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="gulfeval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "validate", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.yaml")
        if name == "run":
            p.add_argument("--synthetic", action="store_true", help="use a fake dataset + mock model (no downloads)")
    a = ap.parse_args(argv)
    cfg = _cfg(a.config)

    if a.cmd == "inspect":
        print(data.inspect_hf(cfg["dataset"]["hf_name"]))
        return
    if a.cmd == "run" and a.synthetic:
        from .synthetic import make_synthetic

        df = make_synthetic(cfg)
        cfg["model"].update(backend="mock", name="mock-model")
    else:
        df = data.load_hf(cfg)
    rep = data.validate_alignment(df)
    print("alignment:", rep)
    if a.cmd == "validate":
        return
    if not rep.ok:
        raise SystemExit("Alignment check failed; fix dataset.columns.id in the config before running.")
    out = run.run_dir(cfg)
    if a.cmd == "run":
        sub_df = data.select_subset(df, cfg)
        print("subjects included:", sorted(sub_df.subject.unique()))
        print(f"running {sub_df.qid.nunique()} questions x {sub_df.variant.nunique()} variants "
              f"x {cfg['run']['repeats']} repeat(s)")
        preds = run.run_experiment(cfg, sub_df)
    else:
        import pandas as pd

        preds = pd.read_json(out / "predictions.jsonl", lines=True)
    print(report.make_report(preds, cfg, out))


if __name__ == "__main__":
    main()
