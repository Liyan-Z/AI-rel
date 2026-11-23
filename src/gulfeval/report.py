"""Turn predictions into tables (CSV + Markdown) and one summary figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats  # noqa: E402


def make_report(preds: pd.DataFrame, cfg: dict, out: Path) -> str:
    ref = cfg["reference_variant"]
    acc = stats.accuracy_table(preds)
    gaps = stats.paired_vs_reference(preds, ref, cfg["stats"]["bootstrap_samples"])
    noise = stats.noise_baseline(preds)
    acc.to_csv(out / "accuracy.csv", index=False)
    gaps.to_csv(out / "paired_vs_reference.csv", index=False)
    if noise is not None:
        noise.to_csv(out / "noise_baseline.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    a = acc.sort_values("accuracy")
    ax.errorbar(a.accuracy, range(len(a)), xerr=[a.accuracy - a.ci_low, a.ci_high - a.accuracy],
                fmt="o", color="#2a6f97", capsize=3)
    ax.set_yticks(range(len(a)), a.variant)
    ax.set_xlabel(f"Accuracy (95% Wilson CI) | model: {cfg['model']['name']} | subset: {cfg['run']['subset']}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_variant.png", dpi=160)
    plt.close(fig)

    md = [f"# Results: {cfg['model']['name']} ({cfg['run']['subset']} subset)\n",
          "## Accuracy by variant\n", acc.round(4).to_markdown(index=False), "",
          f"## Gap vs {ref} (paired)\n", gaps.round(4).to_markdown(index=False), ""]
    if noise is not None:
        md += ["## Sampling-noise baseline (repeat 0 vs repeat 1, same variant)\n",
               noise.round(4).to_markdown(index=False), ""]
    text = "\n".join(md)
    (out / "report.md").write_text(text)
    return text
