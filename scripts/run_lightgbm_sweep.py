#!/usr/bin/env python3
"""Tune LightGBM on the validation years and evaluate once on the future holdout."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.features import RobustFeatureTransform, build_sequences
from fme.metrics import binary_metrics, choose_recall_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--output-dir", default="results/uop-lgbm-sweep")
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
    configs = [
        {"num_leaves": leaves, "max_depth": depth, "min_child_samples": child, "learning_rate": rate, "n_estimators": rounds, "reg_lambda": reg}
        for leaves, depth, child, rate, rounds, reg in [
            (7, 4, 20, .03, 300, 1.0), (15, 5, 20, .03, 300, 1.0),
            (31, 6, 20, .03, 300, 1.0), (15, -1, 50, .03, 500, 5.0),
            (7, 4, 50, .06, 500, 5.0), (15, 5, 50, .06, 500, 5.0),
            (31, 6, 50, .06, 500, 5.0), (15, -1, 20, .06, 300, 1.0),
        ]
    ]
    trials = []
    best = None
    for view_name, matrix in views.items():
        for config_id, config in enumerate(configs):
            model = LGBMClassifier(
                **config, objective="binary", class_weight="balanced", verbosity=-1,
                n_jobs=2, random_state=42,
            )
            model.fit(matrix[train], y[train])
            pred = model.predict_proba(matrix[valid])[:, 1]
            ap = float(average_precision_score(y[valid], pred))
            trial = {"view": view_name, "config_id": config_id, "validation_pr_auc": ap, "config": config}
            trials.append(trial)
            if best is None or ap > best["validation_pr_auc"]:
                best = {**trial, "model": model, "validation_scores": pred}
    assert best is not None
    val_threshold = choose_recall_threshold(y[valid], best["validation_scores"], 0.8)
    selected = LGBMClassifier(
        **best["config"], objective="binary", class_weight="balanced", verbosity=-1,
        n_jobs=2, random_state=42,
    )
    selected.fit(views[best["view"]][train], y[train])
    test_scores = selected.predict_proba(views[best["view"]][test])[:, 1]
    metrics = binary_metrics(y[test], test_scores, val_threshold)
    test_frame = data.loc[test, ["company_id", "fiscal_year", "label"]].copy()
    test_frame["score"] = test_scores
    test_frame["threshold"] = val_threshold
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(out / "predictions.csv", index=False)
    val_frame = data.loc[valid, ["company_id", "fiscal_year", "label"]].copy()
    val_frame["score"] = best["validation_scores"]
    val_frame["threshold"] = val_threshold
    val_frame.to_csv(out / "validation_predictions.csv", index=False)
    report = {
        "seed": 42, "split": {"train": "2008-2016", "validation": "2017-2018", "test": "2019-2021"},
        "availability_date_status": "not supplied; fiscal-year-only exploratory evaluation",
        "validation_trials": trials,
        "selected_view": best["view"], "selected_config": best["config"],
        "validation_pr_auc": best["validation_pr_auc"], "validation_threshold_at_recall_0.8": val_threshold,
        "test_metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_view": best["view"], "selected_config": best["config"], "validation_pr_auc": best["validation_pr_auc"], "test_metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
