#!/usr/bin/env python3
"""Validation-tune a rank-normalized graph residual blend and bootstrap the holdout."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.metrics import choose_recall_threshold


def validation_scale(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Strictly monotone robust scaling fitted on validation data only."""
    median = float(np.median(reference))
    q05, q95 = np.quantile(reference, [0.05, 0.95])
    scale = max(float(q95 - q05), 1e-12)
    return (values - median) / scale


def cluster_bootstrap(y, incumbent, candidate, cluster_ids, *, n_boot=2000, seed=2606):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    incumbent = np.asarray(incumbent)
    candidate = np.asarray(candidate)
    clusters = np.unique(cluster_ids)
    grouped = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    diffs_ap, diffs_auc = [], []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        index = np.concatenate([grouped[c] for c in sampled])
        labels = y[index]
        if labels.min() == labels.max():
            continue
        diffs_ap.append(average_precision_score(labels, candidate[index]) - average_precision_score(labels, incumbent[index]))
        diffs_auc.append(roc_auc_score(labels, candidate[index]) - roc_auc_score(labels, incumbent[index]))
    def summary(values):
        return {"mean": float(np.mean(values)), "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])], "valid_replicates": len(values)}
    return {"pr_auc_difference": summary(diffs_ap), "roc_auc_difference": summary(diffs_auc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-results", default="results/uop-lgbm-sweep")
    parser.add_argument("--xgboost-results", default="results/uop-xgb-sweep")
    parser.add_argument("--graph-results", default="results/uop-seed42-corrected")
    parser.add_argument("--output-dir", default="results/uop-score-fusion")
    args = parser.parse_args()
    root = ROOT / "results"
    base_models = {
        "lightgbm": (ROOT / args.lightgbm_results, "score"),
        "xgboost": (ROOT / args.xgboost_results, "score"),
    }
    gnn_val = pd.read_csv(ROOT / args.graph_results / "validation_predictions.csv")
    gnn_test = pd.read_csv(ROOT / args.graph_results / "predictions.csv")
    gnn_val = gnn_val.rename(columns={"year": "fiscal_year"})
    gnn_test = gnn_test.rename(columns={"year": "fiscal_year"})
    weights = np.linspace(0.0, 1.0, 11)
    reports = {}
    test_predictions = []
    for base_name, (base_directory, _) in base_models.items():
      base_val = pd.read_csv(base_directory / "validation_predictions.csv")
      base_test = pd.read_csv(base_directory / "predictions.csv")
      base_val = base_val.rename(columns={"score": "base_score"})
      base_test = base_test.rename(columns={"score": "base_score"})
      for graph_model in ["gcn", "graphsage", "reliability_temporal_gnn"]:
        graph_val = gnn_val.loc[gnn_val.model.eq(graph_model), ["company_id", "fiscal_year", "label", "score"]].rename(columns={"score": "graph_score"})
        graph_test = gnn_test.loc[gnn_test.model.eq(graph_model), ["company_id", "fiscal_year", "label", "score"]].rename(columns={"score": "graph_score"})
        valid = base_val.merge(graph_val, on=["company_id", "fiscal_year", "label"], validate="one_to_one")
        test = base_test.merge(graph_test, on=["company_id", "fiscal_year", "label"], validate="one_to_one")
        valid_base_raw = valid.base_score.to_numpy()
        valid_graph_raw = valid.graph_score.to_numpy()
        valid_base = validation_scale(valid_base_raw, valid_base_raw)
        valid_graph = validation_scale(valid_graph_raw, valid_graph_raw)
        validation_base_auc = float(roc_auc_score(valid.label, valid_base))
        best = None
        validation_grid = []
        for w in weights:
            blend = (1 - w) * valid_base + w * valid_graph
            ap = float(average_precision_score(valid.label, blend))
            auc = float(roc_auc_score(valid.label, blend))
            candidate = {"graph_weight": float(w), "validation_pr_auc": ap, "validation_roc_auc": auc}
            validation_grid.append(candidate)
            if auc >= validation_base_auc and (best is None or ap > best["validation_pr_auc"]):
                best = candidate
        if best is None:
            best = {"graph_weight": 0.0, "validation_pr_auc": float(average_precision_score(valid.label, valid_base)), "validation_roc_auc": validation_base_auc}
        test_base = validation_scale(valid_base_raw, test.base_score.to_numpy())
        test_graph = validation_scale(valid_graph_raw, test.graph_score.to_numpy())
        test["blend_score"] = (1 - best["graph_weight"]) * test_base + best["graph_weight"] * test_graph
        validation_blend = (1 - best["graph_weight"]) * valid_base + best["graph_weight"] * valid_graph
        operating_threshold = choose_recall_threshold(valid.label.to_numpy(), validation_blend, 0.8)
        test_prediction = (test.blend_score.to_numpy() >= operating_threshold).astype(int)
        model_key = f"{base_name}__{graph_model}"
        reports[model_key] = {
            "base_model": base_name,
            "graph_model": graph_model,
            **best,
            "validation_grid": validation_grid,
            "validation_base_pr_auc": float(average_precision_score(valid.label, valid_base)),
            "validation_base_roc_auc": validation_base_auc,
            "validation_graph_pr_auc": float(average_precision_score(valid.label, valid_graph)),
            "test_base_pr_auc": float(average_precision_score(test.label, test_base)),
            "test_base_roc_auc": float(roc_auc_score(test.label, test_base)),
            "test_blend_pr_auc": float(average_precision_score(test.label, test.blend_score)),
            "test_blend_roc_auc": float(roc_auc_score(test.label, test.blend_score)),
            "validation_selected_threshold_for_recall_0.8": operating_threshold,
            "test_precision_at_validation_threshold": float(precision_score(test.label, test_prediction, zero_division=0)),
            "test_recall_at_validation_threshold": float(recall_score(test.label, test_prediction, zero_division=0)),
            "test_paired_company_cluster_bootstrap": cluster_bootstrap(
                test.label.to_numpy(), test_base, test.blend_score, test.company_id.astype(str).to_numpy()
            ),
        }
        test["base_model"] = base_name
        test["graph_model"] = graph_model
        test_predictions.append(test[["company_id", "fiscal_year", "label", "base_model", "graph_model", "blend_score"]])
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(test_predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({
        "split": {"train": "2008-2016", "validation": "2017-2018", "test": "2019-2021"},
        "availability_date_status": "not supplied; fiscal-year-only exploratory evaluation",
        "fusion": "validation-selected convex average of validation-scaled XGBoost or LightGBM scores and temporal graph scores",
        "validation_grid": [float(w) for w in weights],
        "cluster_bootstrap_replicates": 2000,
        "models": reports,
    }, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
