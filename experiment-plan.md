# Experiment Plan

## Objective

Test whether time-varying, reliability-weighted financial peer graphs improve one-year-ahead corporate bankruptcy prediction beyond tabular and temporal baselines under chronological evaluation.

## Dataset and data gate

The initial candidate is the public American Stock Market company-year dataset (1999–2018). The official CSV is retained locally under `data/raw/` and excluded from Git. Before any model is trained, the audit checks firm-year uniqueness, field completeness, target timing, and within-company label transitions. A usable target must mean a future event at a defined horizon; a company-level label copied to every year is rejected for forecasting.

The first audit found that the candidate release has 78,682 rows, 8,971 firms, 18 numeric variables, broad industry fields, and no duplicate company-years. However, all 609 firms labelled `failed` have that label in every observed year, and no company's label changes over time. This does not support the planned one-year-ahead task. No performance results may be produced from this release unless an independently documented event year is recovered. See `docs/data-audit.md`.

## Proposed model

An LSTM encodes up to five observed fiscal years. For each prediction year, relation-specific peer graphs are built from lagged financial feature groups and, when available, SIC/industry fields. Each edge weight combines current similarity with neighbor-set persistence from the prior graph snapshot. Relation-specific normalized message passing is combined with the temporal state and a self-only path for binary prediction.

## Training protocol

Fit feature imputation, clipping, and scaling on training years only. Use chronological train/validation/test periods and report a second firm-disjoint evaluation. Train with binary cross-entropy and training-set class weighting; select checkpoints and thresholds using validation data only. Fixed seed 42 is the smoke-test default; the paper protocol will use multiple declared seeds.

## Baselines

Logistic regression, random forest, gradient boosting, feature-only MLP, LSTM, GCN, GraphSAGE, static peer graph, and the proposed reliability-aware temporal graph model.

## Metrics and ablations

Primary metric: PR-AUC. Secondary metrics: ROC-AUC, recall at a validation-selected precision/recall operating point, Brier score, and confusion matrix. Ablations remove temporal history, each graph relation, edge persistence weighting, and graph message passing. Stress tests perturb edges and vary k, history length, and graph density.

## Compute and outputs

The project is designed for a single local GPU but runs on CPU for smoke checks. No cloud experiment is configured or launched. Successful runs save `metrics.json`, `predictions.csv`, the selected checkpoint, and a loss plot in the requested output directory. Results are not included until measured.
