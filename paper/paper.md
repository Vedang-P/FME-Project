# Reliability-Aware Temporal Peer Graphs for Corporate Distress Prediction

**Authors:** Atharv Shanker, Vedang Pandey, Shoumik Mitra
**Status:** Research scaffold; no validated forecasting dataset or experimental results yet.

## Abstract

Corporate distress prediction combines a firm's financial history with information about related firms, but similarity graphs can change over time and may contain unreliable links. We study whether relation-specific, reliability-weighted temporal peer graphs improve future-period bankruptcy prediction beyond strong tabular and temporal baselines. A recurrent encoder summarizes up to five years of financial data. Graph snapshots are constructed only from information available at each prediction date; edge weights combine contemporaneous feature similarity with persistence across adjacent snapshots. We evaluate this design using chronological and company-disjoint splits, graph perturbations, ablations, and probability calibration. **The public candidate dataset currently fails the point-in-time target audit, so no performance results or empirical claims are reported.**

## 1. Introduction

Financial distress develops over time and may co-vary among firms with related financial profiles or industry exposure. The project asks whether those peer relationships add measurable predictive value under realistic out-of-time evaluation. The contribution, if supported after a literature audit, is the explicit modeling and stress-testing of peer-edge reliability over temporal graph snapshots—not the general combination of recurrent and graph neural networks.

## 2. Data and prediction task

The candidate U.S. public-company dataset spans 1999–2018 and includes 18 financial variables and broad industry fields. Its released target is constant within each firm across all years, and its CSV firm count differs from the paper's reported count. It cannot support a one-year-ahead forecasting evaluation as provided. The data section and all experimental claims remain pending a point-in-time dataset with verified event dates, feature availability dates, and complete negative follow-up.

## 3. Method

For each firm and prediction year, an LSTM encodes the preceding five fiscal years. Snapshot graphs represent financial-peer relations by feature group and optional industry membership. Edge confidence combines feature similarity with neighborhood persistence between consecutive snapshots. Relation-specific message passing is fused with the firm's temporal state and a self-only representation for binary classification.

## 4. Evaluation protocol

Use chronological train, validation, and test periods, plus a company-disjoint generalization analysis. Fit every transformation on training data only. Select checkpoints, calibration, and operating thresholds using validation data. The rationale is that distress may be better characterized by multiyear changes and that financially similar peers may provide context; persistence weighting is intended to reduce reliance on transient links. These are hypotheses, not established benefits.

Compare regularized logistic regression, decision tree, random forest, and histogram gradient boosting on current-year features and flattened five-year histories; compare MLP on both views; and include LSTM, unweighted-for-persistence GCN/GraphSAGE, and static or industry graph variants when supported by the licensed fields. Report PR-AUC and ROC-AUC, precision at a validation-selected 80%-recall operating point, Brier score, calibration, and confusion matrices. Use paired company-clustered confidence intervals and repeated-seed results. Ablate temporal history, relation channels, persistence weighting, and message passing; stress-test edge corruption and graph density.

The project defines success in advance: the proposed method must outperform both the strongest traditional baseline and the strongest neural/graph baseline on held-out PR-AUC and ROC-AUC, with paired company-clustered 95% confidence intervals above zero, consistent direction across declared seeds and evaluable future windows, and no failure of operating-point or calibration guardrails. If this gate fails, the method has not met the project's success criterion; the gate will not be relaxed after seeing test results. See `docs/comparison-framework.md` for the complete protocol.

## 5. Results and limitations

**Pending a valid dataset and measured experiments.** The released candidate's firm-level labels cannot be interpreted as next-year events. Similarity relationships are observational and do not establish risk contagion or causal influence. The A* workshop target is not yet selected.

## References

References and venue-specific formatting will be completed after the novelty audit and dataset choice are finalized.
