# FME Project: Time-Aware Graph Learning for Corporate Distress

Research software for testing whether reliable, time-varying peer relationships improve corporate distress prediction. The central model combines a five-year LSTM with relation-specific graph message passing over financial similarity and, when available, industry peer links.

The original project synopsis is preserved at [`synopsis.pdf`](synopsis.pdf).

## Research status

The first candidate, the American Stock Market bankruptcy CSV, has a firm-constant label and cannot support next-year forecasting as released; see [`docs/data-audit.md`](docs/data-audit.md). We then audited a public SEC-derived numeric dataset with next-year labels. Its filing dates are absent and 103 company-years contain repeated filings, including one conflicting label group. The local experiment excludes those repeated company-years and uses year-only chronological splits. This permits an exploratory benchmark, but it is **not** a verified filing-time study.

One seed was run on the 2019--2021 holdout (9,032 filings, 64 positives). Tuned XGBoost reached AP 0.1141 and ROC-AUC 0.8868, compared with LightGBM at 0.1116 and 0.8790. The reliability-aware temporal GNN reached AP 0.0383 and ROC-AUC 0.7515. Validation-selected XGBoost--GNN fusion reached AP 0.1133 and ROC-AUC 0.8806, below XGBoost on both metrics; its paired company-clustered AP interval includes zero and ROC-AUC declines. **The GNN did not beat the strong baseline on both required metrics, so the success criterion is not met.** Alanis et al. report XGBoost ROC-AUC 0.92 on a different cohort and walk-forward protocol; it is listed as source-reported context, not a directly comparable score or exact reproduction. Recent temporal relational GNN work also overlaps the tested architecture, so no novelty claim is made for LSTM--GNN integration. These are preliminary fiscal-year-only results, not an SOTA claim. See the [IEEE-style paper PDF](paper/main.pdf), [source-specific data audit](docs/uop-data-audit.md), and [`paper/experiment_results.json`](paper/experiment_results.json).

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

The SEC-derived numeric source is available from the [University of Peloponnese repository record](https://amitos.library.uop.gr/items/7bc5792c-2982-4928-8f75-42950fc2ef25). Download `numeric-dataset-variants.zip`, then place `companyfacts_dense.csv` at `data/raw/uop-bankruptcy/companyfacts_dense.csv`. The source's attached terms require attribution and do not grant this project redistribution rights; raw data are therefore ignored by Git. Read [`docs/uop-data-audit.md`](docs/uop-data-audit.md) before using the prepared panel.

## Audit and run

```bash
python scripts/prepare_uop_numeric_panel.py
python scripts/run_lightgbm_sweep.py
python scripts/run_xgboost_sweep.py
python train.py --data_file data/processed/uop_numeric_panel.csv --output_dir results/uop-seed42-corrected --train_end_year 2016 --validation_end_year 2018 --epochs 12 --patience 3 --hidden_dim 32 --neighbors 10 --seed 42 --allow-year-only-source
python scripts/evaluate_score_fusion.py
python scripts/build_paper_results.py
```

An optional engineered-ratio check is available with `python scripts/derive_financial_ratios.py`, then rerun the LightGBM sweep with `--data-file data/processed/uop_numeric_ratio_panel.csv --output-dir results/uop-ratio-lgbm-sweep` and `train.py` with that panel. In the recorded single-seed check, adding 12 ratios reduced LightGBM AP to 0.0827 and ROC-AUC to 0.8670.

The first script audits and prepares a local panel, excluding all repeated company-years because exact filing dates are unavailable. LightGBM/XGBoost hyperparameters and score-fusion weights are chosen on validation only. The graph training command requires the explicit `--allow-year-only-source` flag to mark the result exploratory. Outputs are written under ignored `results/`; the paper tables, summary, and local experiment ledger are generated from those measured outputs. Smoke mode uses synthetic data and is never research evidence.

## Data and license

The original candidate benchmark is available at [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset) and carries its own CC BY 4.0 license. The exact audited upstream commit and file checksums are recorded in `docs/data-audit.md`. The raw CSV is ignored by Git and must not be redistributed without preserving its attribution and license. The synopsis PDF is not covered by this repository's MIT license; see `docs/NOTICE.md`.

## Reproducibility and reporting

Default split years, model settings, and seed are in `configs/default.json`. The output includes `metrics.json`, `predictions.csv`, `loss_curve.png`, and `model.pt`. The runnable comparison suite includes logistic regression, decision tree, random forest, histogram gradient boosting, and MLP controls on both current-year and flattened five-year features, as well as LSTM, GCN, GraphSAGE, and the reliability-aware temporal GNN. Conventional graph baselines omit persistence weighting to isolate the proposed mechanism.

The project's hard success criterion remains improvement over strong tabular and temporal/graph baselines on both AP and ROC-AUC, with paired company-clustered confidence intervals above zero and stability across multiple seeds and future windows. The current one-seed, one-window result fails this gate. Exact filing dates, survival comparisons, the full predeclared tuning budget, and multi-window replication remain outstanding. Feature-similarity edges are statistical associations, not evidence of contagion.

## Paper

The complete current manuscript is [`paper/main.tex`](paper/main.tex), compiled as [`paper/main.pdf`](paper/main.pdf) with IEEEtran. `paper/paper.md` summarizes its claims and limitations. The workshop venue remains unselected; this exploratory report is not submission-ready evidence for an A* venue.
