# Manuscript summary

The IEEE conference manuscript is available in [`main.tex`](main.tex) and [`main.pdf`](main.pdf). It proposes Delta, a GCN+LSTM model, and evaluates Charlie, a validation-selected multi-head LSTM and boosted-tree score fusion, alongside the full set of tabular, temporal, graph, and score-fusion baselines on an SEC-derived bankruptcy panel.

## Test results

| Model | AP | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.1141 | 0.8868 |
| LightGBM | 0.1116 | 0.8790 |
| Logistic regression | 0.0504 | 0.8325 |
| Histogram gradient boosting | 0.0546 | 0.8028 |
| MLP | 0.0589 | 0.8154 |
| LSTM | 0.0708 | 0.7740 |
| (Ours) Delta (GCN + LSTM) | 0.0794 | 0.7708 |
| GraphSAGE + LSTM | 0.0698 | 0.7649 |
| Reliability-aware temporal GNN | 0.0383 | 0.7515 |
| LightGBM + GraphSAGE fusion | 0.1181 | 0.8705 |
| LightGBM + reliability-GNN fusion | 0.1127 | 0.8688 |
| XGBoost + reliability-GNN fusion | 0.1133 | 0.8806 |

The LightGBM and XGBoost fusion weights were selected on validation data. The paired company-clustered 95% interval for the LightGBM-GraphSAGE AP difference is [-0.00285, 0.01913]. For XGBoost-reliability-GNN fusion, the AP interval is [-0.01363, 0.01493] and the ROC-AUC difference is -0.00626 with interval [-0.01558, -0.00037].

## Charlie score fusion

Charlie centers the tree and LSTM scores using validation medians and scales by each validation 95th-to-5th percentile range. It selects the LSTM weight on validation to maximize AP subject to no validation ROC-AUC decrease.

| Test window | Tree model | Tree AP / ROC-AUC | Charlie AP / ROC-AUC | Paired ΔAP, 95% CI | Paired ΔROC-AUC, 95% CI |
|---|---|---:|---:|---:|---:|
| 2015–2016 | XGBoost | 0.0826 / 0.8601 | 0.1036 / 0.8497 | 0.0234 [-0.0038, 0.0578] | -0.0102 [-0.0460, 0.0193] |
| 2015–2016 | LightGBM | 0.0697 / 0.8536 | 0.0928 / 0.8580 | 0.0254 [-0.0552, 0.0974] | 0.0045 [-0.0405, 0.0475] |
| 2019–2021 | XGBoost | 0.1141 / 0.8868 | 0.1375 / 0.8895 | 0.0233 [0.0021, 0.0518] | 0.0027 [-0.0021, 0.0078] |
| 2019–2021 | LightGBM | 0.1116 / 0.8790 | 0.1300 / 0.8875 | 0.0175 [-0.0006, 0.0410] | 0.0085 [-0.0004, 0.0193] |

The later XGBoost fusion has a positive paired AP interval, but its ROC-AUC interval includes zero. Earlier-window AP intervals include zero; the XGBoost fusion lowers ROC-AUC by point estimate. These results are exploratory and do not meet the dual-metric success criterion. Full aggregates are in [`charlie_results.json`](charlie_results.json).

## Evaluation scope

The filtered cohort contains 42,742 company-year observations from 5,612 firms, with 293 positive labels and 34 numeric features. Training, validation, and test periods are 2008--2016, 2017--2018, and 2019--2021. Exact filing-availability dates are unavailable, so the evaluation uses fiscal-year ordering. Results use one neural seed and one future test window.

Alanis, Chava, and Shah report XGBoost ROC-AUC 0.92 on a different U.S. public-firm cohort and walk-forward protocol. The manuscript lists that score separately from local results. [Source paper](https://arxiv.org/abs/2212.12051)
