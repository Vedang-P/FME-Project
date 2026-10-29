# FME Project: Time-Aware Graph Learning for Corporate Distress

Research software for evaluating corporate distress models. The proposed graph model, Delta, combines a five-year LSTM with graph convolution over financial-similarity peers. The project also evaluates Charlie, a validation-selected fusion of multi-head LSTM and boosted-tree scores. The reliability-aware temporal GNN and other tabular, neural, graph, and score-fusion methods remain in the benchmark comparisons.

The original project synopsis is preserved at [`synopsis.pdf`](synopsis.pdf).

Research website: [fme-research-atlas.vercel.app](https://fme-research-atlas.vercel.app)

## Research status

The first candidate, the American Stock Market bankruptcy CSV, has a firm-constant label and cannot support next-year forecasting as released; see [`docs/data-audit.md`](docs/data-audit.md). We then audited a public SEC-derived numeric dataset with next-year labels. Its filing dates are absent and 103 company-years contain repeated filings, including one conflicting label group. The local experiment excludes those repeated company-years and uses year-only chronological splits. This permits an exploratory benchmark, but it is **not** a verified filing-time study.

On the 2019--2021 holdout (9,032 rows, 64 positives), Delta (GCN+LSTM) reached AP 0.0794 and ROC-AUC 0.7708, compared with the standalone LSTM at 0.0708 and 0.7740. XGBoost remains the strongest individual model at AP 0.1141 and ROC-AUC 0.8868. Separately, Charlie with XGBoost reached AP 0.1375 and ROC-AUC 0.8895. Its paired company-clustered AP difference was 0.0233 (95% CI 0.0021 to 0.0518), while the ROC-AUC difference was 0.0027 (95% CI -0.0021 to 0.0078). In the earlier 2015--2016 window, Charlie's AP gains were uncertain and the XGBoost fusion's ROC-AUC fell by point estimate. Charlie therefore does not establish consistent gains on both required metrics. All other tabular, neural, graph, and score-fusion results remain in the [main results table](paper/results_table.tex); the separate [Charlie results table](paper/charlie_results.tex) and [aggregate results ledger](paper/charlie_results.json) preserve the two-window comparison. This fiscal-year evaluation does not establish filing-time performance or state of the art. See the [IEEE-format paper PDF](paper/main.pdf), [source-specific data audit](docs/uop-data-audit.md), and [`paper/experiment_results.json`](paper/experiment_results.json).

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

### Reproduce Charlie

After preparing the raw and ratio panels as described above, run the validation screening, freeze the selected LSTM settings, and score both fiscal-year windows. The raw panels and experiment outputs remain local and are not committed.

```bash
python scripts/run_xgboost_sweep.py
python scripts/run_lightgbm_sweep.py
python scripts/run_gbdt_fiscal_fold.py --fold early --output-dir results/gbdt-early
python scripts/run_multihead_lstm_sweep.py --folds early late --top-configs 1 --validation-only --output-dir results/multihead-lstm-validation-only
python scripts/run_multihead_lstm_sweep.py --folds early late --top-configs 1 --screening-file results/multihead-lstm-validation-only/experiment.json --output-dir results/multihead-lstm-final
python scripts/evaluate_lstm_fusion.py --fold late --output-dir results/multihead-lstm-fusion
python scripts/evaluate_lstm_fusion.py --fold early --xgboost-results results/gbdt-early --lightgbm-results results/gbdt-early --output-dir results/multihead-lstm-fusion-early
python scripts/build_charlie_results.py
python paper/figures/plot_charlie_results.py
```

The XGBoost/LightGBM tuning commands produce the late-window score files. The folded GBDT command produces both tree score sets for the early window. The tracked aggregate ledger includes metrics and confidence intervals only, not company-level score files.

## Data and license

The original candidate benchmark is available at [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset) and carries its own CC BY 4.0 license. The exact audited upstream commit and file checksums are recorded in `docs/data-audit.md`. The raw CSV is ignored by Git and must not be redistributed without preserving its attribution and license. The synopsis PDF is not covered by this repository's MIT license; see `docs/NOTICE.md`.

## Reproducibility and reporting

Default split years, model settings, and seed are in `configs/default.json`. The output includes `metrics.json`, `predictions.csv`, `loss_curve.png`, and `model.pt`. The runnable comparison suite includes logistic regression, decision tree, random forest, histogram gradient boosting, and MLP controls on both current-year and flattened five-year features, as well as LSTM, GCN+LSTM (Delta), GraphSAGE+LSTM, and the reliability-aware temporal GNN. Delta uses conventional similarity-weighted graph convolution; the reliability-aware comparator separately applies peer-set persistence weighting.

The project's hard success criterion remains improvement over strong tabular and temporal/graph baselines on both AP and ROC-AUC, with paired company-clustered confidence intervals above zero and stability across multiple seeds and future windows. Charlie shows an AP gain for XGBoost on the late holdout, but its ROC-AUC interval includes zero and the earlier window is mixed, so it does not pass this gate. Exact filing dates, survival comparisons, a fresh confirmatory period, and broader multi-window replication remain outstanding. Feature-similarity edges are statistical associations, not evidence of contagion.

Charlie is implemented by [`scripts/evaluate_lstm_fusion.py`](scripts/evaluate_lstm_fusion.py) and its multi-head temporal encoder by [`scripts/run_multihead_lstm_sweep.py`](scripts/run_multihead_lstm_sweep.py) and [`src/fme/model.py`](src/fme/model.py). The aggregate paper ledger can be rebuilt from local ignored experiment outputs with `python scripts/build_charlie_results.py`; the original metric files and predictions are not committed.

## Paper

The complete current manuscript is [`paper/main.tex`](paper/main.tex), compiled as [`paper/main.pdf`](paper/main.pdf) with IEEEtran. `paper/paper.md` summarizes its claims and limitations. The workshop venue remains unselected; this exploratory report is not submission-ready evidence for an A* venue.
