# Manuscript summary

The IEEE conference manuscript is available in [`main.tex`](main.tex) and [`main.pdf`](main.pdf). It evaluates financial-peer temporal graph models against boosted-tree and neural baselines on a SEC-derived bankruptcy panel.

## Test results

| Model | AP | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.1141 | 0.8868 |
| LightGBM | 0.1116 | 0.8790 |
| Reliability-aware temporal GNN | 0.0383 | 0.7515 |
| XGBoost with reliability-GNN score fusion | 0.1133 | 0.8806 |
| LightGBM with GraphSAGE score fusion | 0.1181 | 0.8705 |

The LightGBM and XGBoost fusion weights were selected on validation data. The paired company-clustered 95% interval for the LightGBM-GraphSAGE AP difference is [-0.00285, 0.01913]. For XGBoost-reliability-GNN fusion, the AP interval is [-0.01363, 0.01493] and the ROC-AUC difference is -0.00626 with interval [-0.01558, -0.00037].

## Evaluation scope

The filtered cohort contains 42,742 company-year observations from 5,612 firms, with 293 positive labels and 34 numeric features. Training, validation, and test periods are 2008--2016, 2017--2018, and 2019--2021. Exact filing-availability dates are unavailable, so the evaluation uses fiscal-year ordering. Results use one neural seed and one future test window.

Alanis, Chava, and Shah report XGBoost ROC-AUC 0.92 on a different U.S. public-firm cohort and walk-forward protocol. The manuscript lists that score separately from local results. [Source paper](https://arxiv.org/abs/2212.12051)
