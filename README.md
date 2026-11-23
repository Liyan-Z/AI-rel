# GulfEval

**Does an AI model answer health-science questions less accurately, or less consistently, when the same question is written in an Arabic dialect instead of Modern Standard Arabic (MSA) or English?**

GulfEval is a small, reproducible evaluation pipeline built on
[DialectalArabicMMLU](https://arxiv.org/abs/2510.27543) (Altakrori et al., LREC 2026). That benchmark has
~3,135 MMLU-Redux questions, each translated by hand into five dialects (Egyptian, Moroccan, Saudi, Syrian,
Emirati) plus MSA and English — 21,945 question variants in total.

This project is a **replication-and-extension**, not a new benchmark. The benchmark authors already compare models across
dialects. What this repo adds:

| Extension | Why it matters |
|---|---|
| **Paired analysis** on the *same* question across variants (McNemar, paired bootstrap CIs) | Removes question-difficulty noise from the dialect comparison |
| **Answer-flip rate** vs MSA, not just accuracy | Accuracy can be equal while many individual answers change |
| **Sampling-noise baseline** (repeat runs of the same variant) | Separates dialect-caused flips from ordinary run-to-run randomness |
| **Health & biology slice** as the headline | Whether a learner can trust an AI tutor in their own dialect on health topics |
| **Gulf (UAE, KSA) vs other dialects as controls** | Distinguishes a Gulf-specific effect from a general "non-MSA" penalty |
| **Parse-failure tracking** per variant | Stops a formatting artifact being mistaken for a dialect effect |

## Status

- Pipeline, statistics and tests are done and pass on synthetic data (`pytest`).
- Config matches the dataset card for
  [`MBZUAI/Dialectal-Arabic-MMLU`](https://huggingface.co/datasets/MBZUAI/Dialectal-Arabic-MMLU) (CC BY 4.0): columns
  `dialect`, `qid`, `domain`, `question`, `choices`, `answer`. `qid` repeats across domains, so questions are keyed by
  `domain|qid`.
- **Not yet run on the real dataset.** No results are reported here, and none should be until a real run exists.

## Quick start

```bash
pip install -e ".[anthropic]"        # or ".[openai]" / ".[hf]" for a local open model
gulfeval run --synthetic             # fake data + fake model: checks the plumbing, costs nothing

gulfeval validate                    # downloads the data; confirms each question lines up across all 7 variants
export ANTHROPIC_API_KEY=...         # (or OPENAI_API_KEY)
gulfeval run                         # resumable; predictions cached in results/<run_name>/
gulfeval report                      # rebuild tables + figure from cached predictions
```

Free option: run it in Google Colab with `backend: hf` and a small open Arabic-capable model.

## Design choices (and why)

- **Same instruction in one language for every variant** — only the question text changes, so the instruction can't
  cause a gap.
- **Latin option letters (A–D) for every variant** — isolates dialect from answer-label script.
- **Alignment is validated, not assumed** — if the gold answer for the same question id differs between variants,
  the run stops.
- **Unparseable replies are counted as failures**, never guessed, and reported per variant.

## Limitations to state in any write-up

- Dialect text here is *written* Emirati/Saudi/etc. produced by translating MMLU, not natural speech. Say
  "performance on dialect renditions of MMLU-Redux", not "the model understands Emirati".
- Per-subject samples are small (~100 questions); the health slice supports group-level claims, not per-subject ones.
- MMLU questions may appear in model training data. This affects all variants of a question alike, so paired
  comparisons remain valid, but absolute accuracy may be inflated.
- Results are for one model and one prompt format until more are added.
- An independent audit (IFM, on Hugging Face) removed ~6% of this benchmark's samples as problematic, mostly gold-label
  issues. Treat the pipeline's results as provisional until re-checked on that verified subset.
- Very strong models score near ceiling on this benchmark, which shrinks any dialect gap. Smaller or open models will
  show the effect more clearly.

## Roadmap

1. First real run, one cheap model, health slice, 3 repeats → write-up.
2. Add UAE-built Arabic models and a general frontier model for comparison.
3. Mechanism checks: question length, tokenisation cost, English loanwords vs the gap.
4. Short practitioner note for educators and EdTech teams: when is an AI tutor safe to use in dialect?

## Citation

Altakrori, M. H., Habash, N., Lynn, T., Samih, Y., Freihat, A. A., Chirkunov, K., AbuOdeh, M., Florian, R.,
Nakov, P., Aji, A. F. (2026). *DialectalArabicMMLU: Benchmarking Dialectal Capabilities in Arabic and Multilingual
Language Models.* LREC 2026. The dataset's license terms apply to the data; check the dataset card before redistributing.
