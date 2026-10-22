# FME Project: Time-Aware Graph Learning for Corporate Distress

Research software for testing whether reliable, time-varying peer relationships improve corporate distress prediction. The central model combines a five-year LSTM with relation-specific graph message passing over financial similarity and, when available, industry peer links.

The original project synopsis is preserved at [`synopsis.pdf`](synopsis.pdf).

## Research status

The initial public U.S. bankruptcy dataset has been audited, and its `status_label` is constant for every company across all observed years. Its CSV also reports 8,971 firms while the paper reports 8,262. It therefore does **not** provide the point-in-time next-year labels required by this project as released. The training command rejects it for forecasting. No performance claims or results are reported.

The project accepts a prepared, licensed `panel.csv` with:

| Column | Meaning |
|---|---|
| `company_id` | Stable firm identifier |
| `fiscal_year` | Financial reporting year |
| `label` | Binary event label for the documented future horizon |
| numeric columns | Financial predictors available by the prediction date |
| optional `available_date` | Date those predictors became public |
| optional `industry` | Licensed industry group for a separate graph relation |

Labels must reflect a known future horizon and survive the label-transition audit. Do not use company name, filing date, or future-derived status as predictors.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the prepared panel at `data/raw/panel.csv`, or pass another path with `--data_file`.

## Audit and run

```bash
python scripts/audit_dataset.py --data_file data/raw/american_bankruptcy_dataset.csv
python train.py --data_dir data/raw --output_dir results/run-001
python train.py --data_file data/raw/panel.csv --output_dir results/run-001
python train.py --smoke_test --output_dir results/smoke
```

The audit command is read-only. Training is chronological: it selects the checkpoint and operating threshold on validation data, then evaluates the untouched test period. Outputs are written only to `--output_dir`. Smoke mode uses a deterministic synthetic panel to exercise the pipeline; its metrics are software checks, not research results.

## Data and license

The original candidate benchmark is available at [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset) and carries its own CC BY 4.0 license. The exact audited upstream commit and file checksums are recorded in `docs/data-audit.md`. The raw CSV is ignored by Git and must not be redistributed without preserving its attribution and license. The synopsis PDF is not covered by this repository's MIT license; see `docs/NOTICE.md`.

## Reproducibility and reporting

Default split years, model settings, and seed are in `configs/default.json`. The output includes `metrics.json`, `predictions.csv`, `loss_curve.png`, and `model.pt`. For paper results, run the declared seed set, report confidence intervals and all ablations, and retain the complete output manifests. Do not treat feature-similarity edges as causal links or evidence of contagion.

## Paper

See `paper/paper.md` for the claim-safe manuscript scaffold and `experiment-plan.md` for the planned comparisons. The workshop venue remains unselected. All results sections are intentionally marked pending until experiments run on a valid, documented event-horizon dataset.
