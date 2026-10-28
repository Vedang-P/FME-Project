#!/usr/bin/env python3
"""Build paper tables and a compact ledger from measured local artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main() -> int:
    neural = load("results/uop-seed42-corrected/metrics.json")
    lightgbm = load("results/uop-lgbm-sweep/metrics.json")
    xgboost = load("results/uop-xgb-sweep/metrics.json")
    ratio_lightgbm = load("results/uop-ratio-lgbm-sweep/metrics.json")
    ratio_neural = load("results/uop-ratio-seed42/metrics.json")
    fusion_lgbm = load("results/uop-score-fusion/metrics.json")["models"]
    fusion_xgb = load("results/uop-score-fusion-xgb/metrics.json")["models"]

    rows = []
    definitions = [
        ("XGBoost, validation-tuned", xgboost["test_metrics"]),
        ("LightGBM, validation-tuned", lightgbm["test_metrics"]),
        ("Logistic regression, last year", neural["models"]["logistic_regression_last_year"]),
        ("Histogram gradient boosting, last year", neural["models"]["hist_gradient_boosting_last_year"]),
        ("MLP, flattened five-year history", neural["models"]["mlp_flattened_history"]),
        ("LSTM", neural["models"]["lstm"]),
        ("GCN + LSTM", neural["models"]["gcn"]),
        ("GraphSAGE + LSTM", neural["models"]["graphsage"]),
        ("Reliability-aware temporal GNN", neural["models"]["reliability_temporal_gnn"]),
    ]
    for name, metrics in definitions:
        rows.append((name, metrics["pr_auc"], metrics["roc_auc"], metrics["brier"], metrics["precision"], metrics["recall"]))
    for key, label, source in [
        ("lightgbm__graphsage", "LightGBM + GraphSAGE score fusion", {"lightgbm__graphsage": fusion_lgbm["graphsage"]}),
        ("lightgbm__reliability_temporal_gnn", "LightGBM + reliability-aware GNN fusion", {"lightgbm__reliability_temporal_gnn": fusion_lgbm["reliability_temporal_gnn"]}),
        ("xgboost__reliability_temporal_gnn", "XGBoost + reliability-aware GNN fusion", fusion_xgb),
    ]:
        metrics = source[key]
        rows.append((label, metrics["test_blend_pr_auc"], metrics["test_blend_roc_auc"], None,
                     metrics["test_precision_at_validation_threshold"], metrics["test_recall_at_validation_threshold"]))

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Held-out fiscal-year test results (2019--2021, $n=9{,}032$, 64 positives). Operating thresholds target 80\% recall on validation; final columns show test precision and recall.}",
        r"\label{tab:main}", r"\footnotesize",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Model & AP $\uparrow$ & ROC-AUC $\uparrow$ & Brier $\downarrow$ & Precision (val 80\%R) & Test recall \\",
        r"\midrule",
    ]
    for name, ap, auc, brier, precision, recall in rows:
        lines.append(f"{name} & {fmt(ap)} & {fmt(auc)} & {fmt(brier)} & {fmt(precision)} & {fmt(recall)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "paper/results_table.tex").write_text("\n".join(lines) + "\n")

    published = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Published benchmark context from Alanis et al. The source reports ROC-AUC for XGBoost; AP is not reported. Different cohort, predictors, and walk-forward protocol make this non-comparable to our holdout.}",
        r"\label{tab:published}", r"\footnotesize",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Evaluation & AP & ROC-AUC \\", r"\midrule",
        r"Alanis et al. [1] & n/r & 0.9200 \\",
        r"This study, XGBoost & 0.1141 & 0.8868 \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (ROOT / "paper/published_benchmark_table.tex").write_text("\n".join(published) + "\n")

    summary = {
        "source": "local metrics.json artifacts; see per-run metadata for split and availability status",
        "split": neural["splits"],
        "availability_date_status": neural["availability_date_status"],
        "lightgbm": lightgbm,
        "xgboost": xgboost,
        "derived_ratio_diagnostic": {"lightgbm": ratio_lightgbm, "neural_and_graph_models": ratio_neural["models"]},
        "models": neural["models"],
        "score_fusion_lightgbm": fusion_lgbm,
        "score_fusion_xgboost": fusion_xgb,
        "literature_context": {
            "source": "Alanis, Chava, and Shah (2022), arXiv:2212.12051",
            "xgboost_roc_auc_source_reported": 0.92,
            "average_precision_reported": False,
            "directly_comparable": False,
            "reason": "different cohort, predictors, and walk-forward evaluation protocol",
        },
    }
    (ROOT / "paper/experiment_results.json").write_text(json.dumps(summary, indent=2) + "\n")

    ledger = [
        {"run": "uop-seed42-corrected", "kind": "temporal_graph_models", "metrics_path": "results/uop-seed42-corrected/metrics.json", "split": neural["splits"], "availability_date_status": neural["availability_date_status"], "seed": 42, "models": neural["models"]},
        {"run": "uop-lgbm-sweep", "kind": "lightgbm_validation_sweep", "metrics_path": "results/uop-lgbm-sweep/metrics.json", "selected_config": lightgbm.get("selected_config"), "selected_view": lightgbm.get("selected_view"), "validation_trials": len(lightgbm.get("validation_trials", [])), "test_metrics": lightgbm["test_metrics"]},
        {"run": "uop-xgb-sweep", "kind": "xgboost_validation_sweep", "metrics_path": "results/uop-xgb-sweep/metrics.json", "selected_config": xgboost.get("selected_config"), "selected_view": xgboost.get("selected_view"), "validation_trials": len(xgboost.get("validation_trials", [])), "test_metrics": xgboost["test_metrics"]},
        {"run": "uop-ratio-lgbm-sweep", "kind": "derived_ratio_diagnostic", "metrics_path": "results/uop-ratio-lgbm-sweep/metrics.json", "feature_count": 46, "selected_view": ratio_lightgbm.get("selected_view"), "test_metrics": ratio_lightgbm["test_metrics"]},
        {"run": "uop-ratio-seed42", "kind": "derived_ratio_neural_diagnostic", "metrics_path": "results/uop-ratio-seed42/metrics.json", "feature_count": 46, "models": ratio_neural["models"]},
        {"run": "uop-score-fusion-xgb", "kind": "validation_selected_fusion", "metrics_path": "results/uop-score-fusion-xgb/metrics.json", "models": fusion_xgb},
    ]
    (ROOT / ".rstack").mkdir(exist_ok=True)
    (ROOT / ".rstack/experiments.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in ledger))
    print(f"Wrote paper tables, result summary, and experiment ledger under {ROOT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
