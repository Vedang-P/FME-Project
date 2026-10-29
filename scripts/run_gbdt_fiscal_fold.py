#!/usr/bin/env python3
"""Tune XGBoost and LightGBM on a declared fiscal-year validation fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.features import RobustFeatureTransform, build_sequences
from fme.metrics import binary_metrics, choose_recall_threshold


FOLDS = {
    "early": {"train_end": 2012, "val_start": 2013, "val_end": 2014, "test_start": 2015, "test_end": 2016},
    "late": {"train_end": 2016, "val_start": 2017, "val_end": 2018, "test_start": 2019, "test_end": 2021},
}
XGB_CONFIGS = [
    {"max_depth": d, "learning_rate": r, "n_estimators": n, "min_child_weight": c, "reg_lambda": l, "subsample": 0.8, "colsample_bytree": 0.8}
    for d, r, n, c, l in [(2,.03,300,10,1.0),(3,.03,300,10,1.0),(4,.03,300,10,1.0),(2,.05,500,20,5.0),(3,.05,500,20,5.0),(4,.05,500,20,5.0),(3,.03,500,20,5.0),(4,.03,500,20,1.0)]
]
LGB_CONFIGS = [
    {"num_leaves": l, "max_depth": d, "min_child_samples": c, "learning_rate": r, "n_estimators": n, "reg_lambda": reg}
    for l, d, c, r, n, reg in [(7,4,20,.03,300,1.0),(15,5,20,.03,300,1.0),(31,6,20,.03,300,1.0),(15,-1,50,.03,500,5.0),(7,4,50,.06,500,5.0),(15,5,50,.06,500,5.0),(31,6,50,.06,500,5.0),(15,-1,20,.06,300,1.0)]
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=list(FOLDS), default="early")
    parser.add_argument("--data-file", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    spec = FOLDS[args.fold]
    data = pd.read_csv(ROOT / args.data_file)
    data["company_id"] = data.company_id.astype(str)
    data = data.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    feature_columns = [c for c in data.select_dtypes(include=[np.number]).columns if c not in {"fiscal_year", "label"}]
    years = data.fiscal_year.astype(int)
    train = years.le(spec["train_end"]).to_numpy()
    validation = years.between(spec["val_start"], spec["val_end"]).to_numpy()
    test = years.between(spec["test_start"], spec["test_end"]).to_numpy()
    transform = RobustFeatureTransform(feature_columns).fit(data.loc[train])
    x = transform.transform(data)
    sequences, _, _ = build_sequences(data, x, history_length=5)
    views = {"last_year": x, "flattened_history": sequences.reshape(len(sequences), -1)}
    y = data.label.to_numpy(dtype=int)
    positives = max(int(y[train].sum()), 1)
    negatives = max(int(train.sum()) - int(y[train].sum()), 1)
    output_dir = ROOT / (args.output_dir or f"results/gbdt-{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = {}

    for family, configs in (("xgboost", XGB_CONFIGS), ("lightgbm", LGB_CONFIGS)):
        trials = []
        best = None
        for view_name, matrix in views.items():
            for index, config in enumerate(configs):
                if family == "xgboost":
                    model = XGBClassifier(**config, objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=negatives / positives, tree_method="hist", n_jobs=4, random_state=42)
                else:
                    model = LGBMClassifier(**config, objective="binary", class_weight="balanced", verbosity=-1, n_jobs=2, random_state=42)
                model.fit(matrix[train], y[train])
                scores = model.predict_proba(matrix[validation])[:, 1]
                ap = float(average_precision_score(y[validation], scores))
                trial = {"view": view_name, "config_index": index, "validation_ap": ap, "config": config}
                trials.append(trial)
                if best is None or ap > best["validation_ap"]:
                    best = {**trial, "validation_scores": scores}
        assert best is not None
        threshold = choose_recall_threshold(y[validation], best["validation_scores"], 0.8)
        if family == "xgboost":
            final = XGBClassifier(**best["config"], objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=negatives / positives, tree_method="hist", n_jobs=4, random_state=42)
        else:
            final = LGBMClassifier(**best["config"], objective="binary", class_weight="balanced", verbosity=-1, n_jobs=2, random_state=42)
        final.fit(views[best["view"]][train], y[train])
        val_scores = final.predict_proba(views[best["view"]][validation])[:, 1]
        test_scores = final.predict_proba(views[best["view"]][test])[:, 1]

        def frame_for(mask, scores):
            frame = data.loc[mask, ["company_id", "fiscal_year", "label"]].copy()
            frame["score"] = scores
            frame["threshold"] = threshold
            return frame

        frame_for(validation, val_scores).to_csv(output_dir / f"{family}-validation.csv", index=False)
        frame_for(test, test_scores).to_csv(output_dir / f"{family}-test.csv", index=False)
        all_reports[family] = {
            "selected_view": best["view"], "selected_config": best["config"], "validation_ap": best["validation_ap"],
            "validation_metrics": binary_metrics(y[validation], val_scores, threshold),
            "test_metrics": binary_metrics(y[test], test_scores, threshold),
            "validation_trials": trials,
        }

    report = {
        "fold": args.fold, "split": spec,
        "availability_date_status": "fiscal-year-only exploratory; exact filing-availability dates unavailable",
        "metric": "average precision selected on validation; ROC-AUC also reported",
        "models": all_reports,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"selected_view": values["selected_view"], "validation": values["validation_metrics"], "test": values["test_metrics"]} for name, values in all_reports.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
