# Experiment Plan

## Objective and success definition

Test whether time-varying, reliability-weighted financial peer graphs improve one-year-ahead corporate bankruptcy prediction beyond strong traditional, temporal, and graph baselines under chronological evaluation. Success is deliberately demanding: the proposed model must beat the strongest traditional baseline **and** the strongest established neural/graph baseline on held-out PR-AUC and ROC-AUC, with paired company-clustered 95% confidence intervals above zero, consistent gains across declared seeds and future test windows, and no failure of the fixed operating-point or calibration guardrails. Novel architecture or a negative-result analysis by itself does not meet this project's success criterion. The full gate is specified in `docs/comparison-framework.md` and must not be relaxed after test evaluation.

## Dataset and data gate

The initial candidate is the public American Stock Market company-year dataset (1999–2018). The official CSV is retained locally under `data/raw/` and excluded from Git. Before any model is trained, the audit checks firm-year uniqueness, field completeness, target timing, and within-company label transitions. A usable target must mean a future event at a defined horizon; a company-level label copied to every year is rejected for forecasting.

The first audit found that the candidate release has 78,682 rows, 8,971 firms, 18 numeric variables, broad industry fields, and no duplicate company-years. However, all 609 firms labelled `failed` have that label in every observed year, and no company's label changes over time. This does not support the planned one-year-ahead task. No performance results may be produced from this release unless an independently documented event year is recovered. See `docs/data-audit.md`.

## Proposed model

An LSTM encodes up to five observed fiscal years. For each prediction year, relation-specific peer graphs are built from lagged financial feature groups and, when available, SIC/industry fields. Each edge weight combines current similarity with neighbor-set persistence from the prior graph snapshot. Relation-specific normalized message passing is combined with the temporal state and a self-only path for binary prediction.

### Why temporal and peer information could help

Distress is a developing process, so five-year sequences may expose changes in financial condition that a single-year feature vector misses. Firms with similar financial profiles may share exposures and provide context when an individual history is short or noisy. Since feature-derived neighborhoods can be unstable, persistence weighting may reduce the influence of transient links. These are hypotheses only: strong tabular models may capture the signal already, peers may add noise, and the GNN may not outperform simpler methods. Graph edges represent statistical similarity, not causal contagion.

## Training protocol

Fit feature imputation, clipping, and scaling on training years only. Use chronological train/validation/test periods and report a second firm-disjoint evaluation. Train with binary cross-entropy and training-set class weighting; select checkpoints and thresholds using validation data only. Fixed seed 42 is the smoke-test default; the paper protocol will use multiple declared seeds. Declare the hyperparameter search budget for each model family before the final test run. Select the strongest comparator on validation data; never choose baselines after seeing test outcomes.

## Baselines and fair comparison

Run regularized logistic regression, decision tree, random forest, and histogram gradient boosting on both current-year features and a flattened trailing five-year history. Run MLP on both feature views, then LSTM, GCN, and GraphSAGE with the same five-year history. Conventional GCN/GraphSAGE use peer snapshots without persistence weights. Compare against the proposed relation-specific persistence-weighted temporal GNN, plus no-persistence, leave-one-relation-out, and no-message-passing controls. An industry-only graph is a separate baseline only if industry identifiers and use rights are verified.

All methods use the same eligible observations, label horizon, point-in-time cutoff, chronological splits, training-only transformations, and validation-only selection. The current-year versus flattened-history controls show whether any gain comes from access to multiyear history alone. The detailed comparison matrix is in `docs/comparison-framework.md`.

## Metrics, uncertainty, and ablations

Primary ranking metric: PR-AUC / average precision. Co-primary success metric: ROC-AUC. Also report precision at a validation-selected threshold targeting 80% recall, achieved precision and recall, Brier score, calibration, class counts, and confusion matrix. Use paired company-cluster bootstrap confidence intervals and repeated declared seeds. Add a second future test window when the event count supports it. Accuracy is not a headline metric.

Ablations remove temporal history, each graph relation, edge persistence weighting, and graph message passing. Stress tests perturb edges and vary k, history length, and graph density. Detailed gate definitions and reporting rules are in `docs/comparison-framework.md`.

## Compute and outputs

The project is designed for a single local GPU but runs on CPU for smoke checks. No cloud experiment is configured or launched. Successful runs save `metrics.json`, `predictions.csv`, the selected checkpoint, and a loss plot in the requested output directory. Results are not included until measured. Passing the full hard success gate—not merely running the code—is required before calling the paper successful under this project's criterion.
