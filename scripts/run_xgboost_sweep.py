#!/usr/bin/env python3
"""Tune XGBoost on validation years and evaluate once on the future holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.features import RobustFeatureTransform, build_sequences
from fme.metrics import binary_metrics, choose_recall_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--output-dir", default="results/uop-xgb-sweep")
    args = parser.parse_args()
    data = pd.read_csv(ROOT / args.data_file)
    data["company_id"] = data.company_id.astype(str)
    data = data.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    columns = [c for c in data.select_dtypes(include=[np.number]).columns if c not in {"fiscal_year", "label"}]
    train = data.fiscal_year.le(2016).to_numpy()
    valid = data.fiscal_year.between(2017, 2018).to_numpy()
    test = data.fiscal_year.ge(2019).to_numpy()
    transform = RobustFeatureTransform(columns).fit(data.loc[train])
    x = transform.transform(data)
    sequences, _, _ = build_sequences(data, x, history_length=5)
    views = {"last_year": x, "flattened_history": sequences.reshape(len(sequences), -1)}
    y = data.label.to_numpy(dtype=int)
    pos = max(int(y[train].sum()), 1)
    neg = max(int(train.sum()) - pos, 1)
    configs = [
        {"max_depth": depth, "learning_rate": rate, "n_estimators": rounds, "min_child_weight": child, "reg_lambda": reg, "subsample": 0.8, "colsample_bytree": 0.8}
        for depth, rate, rounds, child, reg in [
            (2, .03, 300, 10, 1.0), (3, .03, 300, 10, 1.0),
            (4, .03, 300, 10, 1.0), (2, .05, 500, 20, 5.0),
            (3, .05, 500, 20, 5.0), (4, .05, 500, 20, 5.0),
            (3, .03, 500, 20, 5.0), (4, .03, 500, 20, 1.0),
        ]
    ]
    trials = []
    best = None
    for view_name, matrix in views.items():
        for config_id, config in enumerate(configs):
            model = XGBClassifier(
                **config, objective="binary:logistic", eval_metric="aucpr",
                scale_pos_weight=neg / pos, tree_method="hist", n_jobs=4,
                random_state=42,
            )
            model.fit(matrix[train], y[train], verbose=False)
            val_scores = model.predict_proba(matrix[valid])[:, 1]
            ap = float(average_precision_score(y[valid], val_scores))
            trial = {"view": view_name, "config_id": config_id, "validation_pr_auc": ap, "config": config}
            trials.append(trial)
            if best is None or ap > best["validation_pr_auc"]:
                best = {**trial, "validation_scores": val_scores}
    assert best is not None
    threshold = choose_recall_threshold(y[valid], best["validation_scores"], 0.8)
    final = XGBClassifier(
        **best["config"], objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=neg / pos, tree_method="hist", n_jobs=4,
        random_state=42,
    )
    final.fit(views[best["view"]][train], y[train], verbose=False)
    val_scores = final.predict_proba(views[best["view"]][valid])[:, 1]
    test_scores = final.predict_proba(views[best["view"]][test])[:, 1]
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    def predictions(mask, scores):
        result = data.loc[mask, ["company_id", "fiscal_year", "label"]].copy()
        result["score"] = scores
        result["threshold"] = threshold
        return result
    predictions(valid, val_scores).to_csv(out / "validation_predictions.csv", index=False)
    predictions(test, test_scores).to_csv(out / "predictions.csv", index=False)
    report = {
        "seed": 42,
        "split": {"train": "2008-2016", "validation": "2017-2018", "test": "2019-2021"},
        "availability_date_status": "not supplied; fiscal-year-only exploratory evaluation",
        "validation_trials": trials,
        "selected_view": best["view"], "selected_config": best["config"],
        "validation_pr_auc": float(average_precision_score(y[valid], val_scores)),
        "threshold_for_validation_recall_0.8": threshold,
        "test_metrics": binary_metrics(y[test], test_scores, threshold),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_view": best["view"], "selected_config": best["config"], "validation_pr_auc": report["validation_pr_auc"], "test_metrics": report["test_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
