# Baseline comparison and paper success gate

## What the experiment must establish

The paper's central empirical claim is that reliability-aware, time-varying financial-peer graphs improve future-period distress prediction over strong non-graph and graph methods. It is a hypothesis, not an expected result. Under the project's success criterion, an implementation, a novel architecture, or a negative-result analysis alone does not count as a successful paper: the proposed method must clear the held-out performance gate below.

## Why test temporal and peer information

Distress can emerge through changes in liquidity, leverage, profitability, and cash flow, so a five-year history may contain warning patterns that a single-year snapshot misses. Firms with similar financial profiles may also share macroeconomic or sector exposures, making contemporaneous peers a possible source of context. A feature-similarity graph can be noisy and can change as firms' condition changes; weighting links by their persistence is intended to discount unstable neighbors.

These mechanisms could improve ranking and early identification of future distress, especially when an individual firm's history is short or noisy. They can also fail: historical features may already contain the useful signal, peer links may add noise, and a temporal GNN may overfit. Every explanation above is a testable rationale, not a result. Similarity links are statistical associations and do not establish contagion or causality.

## Comparison ladder

All models use the same event horizon, eligible firm-years, point-in-time feature availability, chronological split, and train-only preprocessing. Thresholds are selected on validation data only. Hyperparameter search budgets must be declared and kept comparable; test data is opened once after model and protocol selection.

| Group | Methods | Input | Question answered |
|---|---|---|---|
| Established classical baselines | Regularized logistic regression, decision tree, random forest, histogram gradient boosting | Current-year features and the same trailing five years flattened into one vector | Does the proposed model beat standard credit-risk classifiers, and is any gain explained by access to history alone? |
| Neural tabular controls | MLP on current-year features and flattened five-year histories | Same information as the corresponding classical feature view | Does graph structure help beyond a flexible feature-only neural predictor? |
| Temporal control | LSTM without graph messages | Five-year sequence | Does peer context add value beyond temporal encoding? |
| Conventional graph controls | GCN and GraphSAGE over financial peer snapshots without persistence weighting | Same five-year LSTM states and current peer snapshots | Does reliability weighting add value beyond ordinary graph message passing? |
| Proposed model | Relation-specific temporal GNN with financial-similarity edges weighted by similarity and neighborhood persistence | Same five-year histories; graph built only from information available by each forecast date | Does the proposed relation and edge-reliability mechanism improve prediction? |
| Mechanism controls | Proposed model without each relation, without persistence, and without graph messages; edge perturbation and density variants | Matched to proposed model | Which component drives any measured gain, and is it robust to plausible graph errors? |

The `last_year` versus `flattened_history` classical and MLP results explicitly control for the five-year information available to the proposed model. GCN/GraphSAGE receive snapshots without persistence weighting so their scores do not include the proposed mechanism by accident. If a validated industry field is available and licensed, an industry-only graph is reported as a separate comparator; missing or anonymized industry information is not reconstructed or guessed.

## Metrics and statistical comparison

- **Primary ranking metric: PR-AUC / average precision.** It is the preregistered primary metric because distress events are rare and precision-recall tradeoffs matter.
- **Co-primary success metric: ROC-AUC.** It measures ranking across operating points and prevents a PR-AUC-only win from hiding a broader ranking loss.
- **Operating-point metric: precision at a validation-selected threshold targeting 80% recall.** Report achieved precision and recall; if validation cannot attain the target, say so and report the closest attainable point. The threshold is frozen before test evaluation.
- **Probability quality: Brier score**, with calibration plot and calibration slope/intercept where estimable. Calibration is a guardrail, not a substitute for the ranking gate.
- Report class prevalence, positive and negative counts, confusion matrix, and all test metrics. Do not use accuracy as a headline metric.

Compare predictions on identical test observations. Report paired 95% confidence intervals for metric differences using a company-cluster bootstrap so repeated firm-years are resampled together. Repeat the declared training seeds and report per-seed results and aggregate uncertainty. Include at least two distinct future test windows if the audited panel has enough event-positive firms; otherwise state the limitation and do not imply multi-period robustness.

## Hard success gate

The method passes the project's paper-success criterion only if all of these hold on a valid, audited point-in-time event-horizon dataset:

1. On the untouched chronological test set, the proposed method's PR-AUC is higher than **both** the strongest traditional baseline (selected among logistic regression, decision tree, random forest, and histogram gradient boosting) and the strongest established neural/graph baseline (MLP, LSTM, GCN, or GraphSAGE).
2. The paired company-clustered 95% confidence interval for each PR-AUC difference is entirely above zero, and the proposed model also beats those same two comparators on ROC-AUC with paired 95% intervals entirely above zero.
3. The PR-AUC and ROC-AUC gains have the same direction across the predeclared seeds and each evaluable future test window; a win isolated to one seed or one window does not pass.
4. At the validation-selected operating point, the proposed model does not reduce precision at the target recall relative to either comparator, and its Brier score is not materially worse. Any failure of these guardrails is reported and disqualifies a success claim.
5. No target leakage, test-set tuning, or selective reporting is found; the strongest comparator is selected by validation performance under the declared tuning budget, not by choosing whichever baseline makes the proposed method look best after test results are known.

If any item fails, the method has **not** met the user's success criterion. Do not describe a negative or mixed result as a successful paper or relax this gate after seeing test results. Preserve the result, diagnose the failure, and seek a better justified data source or method before making a success claim. If no valid label panel is available, no performance claim can be evaluated.

## Current status

The current candidate American Stock Market release fails the target audit because its firm-level status is constant across all observed years and does not provide the verified forecast horizon required here. Consequently, the hard gate cannot currently be evaluated, and no baseline comparison scores are research evidence. The expanded runnable baseline suite is ready for an eligible panel; the empirical work remains blocked on valid labels and availability dates.
