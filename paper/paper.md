# Paper status

The IEEE conference-format manuscript is in [`main.tex`](main.tex) and compiled to [`main.pdf`](main.pdf). It reports a fiscal-year-only exploratory comparison on a 2019--2021 holdout. It does not claim an exact reproduction, SOTA result, or architectural novelty.

## Main measured result

| Model | Test AP | Test ROC-AUC |
|---|---:|---:|
| XGBoost, validation-tuned | 0.1141 | 0.8868 |
| LightGBM, validation-tuned | 0.1116 | 0.8790 |
| Reliability-aware temporal GNN | 0.0383 | 0.7515 |
| XGBoost + reliability-GNN fusion | 0.1133 | 0.8806 |

The GNN model and its XGBoost fusion fail to improve on XGBoost. The fusion's paired company-clustered AP interval includes zero, and its ROC-AUC interval indicates a decline. The project's success gate is not met.

Adding 12 engineered financial ratios reduced LightGBM AP to 0.0827 and ROC-AUC to 0.8670 in the single-seed diagnostic.

Alanis, Chava, and Shah report XGBoost ROC-AUC 0.92 on their own U.S. public-firm dataset and walk-forward protocol. The manuscript preserves that source-reported value in a separate table. It is not directly comparable to our local result, and this work reproduces the algorithm family rather than their exact data/model pipeline.

## Limits

The SEC-derived panel omits exact filing availability dates. Repeated company-years were excluded rather than ordered by accession. The reported split is exploratory, has one seed and one future window, and cannot establish point-in-time forecasting performance. A recent published relational temporal GNN also overlaps the tested architecture, so the LSTM--GNN composition is not a novelty claim. Further work needs filing dates, a source-aligned replication, equal-budget tuning, multiple seeds/windows, and a passed dual-metric gate before a positive paper claim.
