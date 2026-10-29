#!/usr/bin/env python3
"""Validation-select an LSTM/GBDT blend and evaluate the frozen score on later years."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.metrics import choose_recall_threshold


def robust_scale(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    center = float(np.median(reference))
    lower, upper = np.quantile(reference, [0.05, 0.95])
    return (values - center) / max(float(upper - lower), 1e-12)


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def cluster_bootstrap(y, reference, candidate, company_ids, *, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    company_ids = np.asarray(company_ids).astype(str)
    companies = np.unique(company_ids)
    groups = {company: np.flatnonzero(company_ids == company) for company in companies}
    differences = {"average_precision": [], "roc_auc": []}
    for _ in range(replicates):
        sampled = rng.choice(companies, size=len(companies), replace=True)
        ids = np.concatenate([groups[c] for c in sampled])
        labels = y[ids]
        if labels.min() == labels.max():
            continue
        differences["average_precision"].append(
            average_precision_score(labels, candidate[ids]) - average_precision_score(labels, reference[ids])
        )
        differences["roc_auc"].append(
            roc_auc_score(labels, candidate[ids]) - roc_auc_score(labels, reference[ids])
        )
    return {
        metric_name: {
            "mean_difference": float(np.mean(values)),
            "ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
            "valid_replicates": len(values),
        }
        for metric_name, values in differences.items()
    }


def predictions_from_experiment(experiment_path: Path, split: str, panel: pd.DataFrame, fold: str) -> pd.DataFrame:
    document = json.loads(experiment_path.read_text())
    key = "selected_validation_runs" if split == "validation" and document.get("selected_validation_runs") else "frozen_test_runs"
    runs = [run for run in document.get(key, []) if run["fold"] == fold]
    if not runs:
        raise ValueError(f"No {fold}-fold {split} runs in {experiment_path}.")
    parts = []
    for run in runs:
        data = run["validation_predictions"] if split == "validation" else run["test_predictions"]
        row_ids = np.asarray(data["row_ids"], dtype=int)
        part = panel.iloc[row_ids][["company_id", "fiscal_year"]].copy()
        part["label"] = np.asarray(data["labels"], dtype=int)
        part[f"score_seed_{run['seed']}"] = np.asarray(data["scores"], dtype=float)
        parts.append(part)
    merged = parts[0]
    for part in parts[1:]:
        score_col = [c for c in part.columns if c.startswith("score_seed_")][0]
        merged = merged.merge(part[["company_id", "fiscal_year", "label", score_col]], on=["company_id", "fiscal_year", "label"], validate="one_to_one")
    score_columns = [c for c in merged if c.startswith("score_seed_")]
    merged["score"] = merged[score_columns].mean(axis=1)
    return merged[["company_id", "fiscal_year", "label", "score"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-validation", default="results/multihead-lstm-validation-only/experiment.json")
    parser.add_argument("--lstm-test", default="results/multihead-lstm-final/experiment.json")
    parser.add_argument("--fold", choices=("early", "late"), default="late")
    parser.add_argument("--panel", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--xgboost-results", default="results/uop-xgb-sweep")
    parser.add_argument("--lightgbm-results", default="results/uop-lgbm-sweep")
    parser.add_argument("--output-dir", default="results/multihead-lstm-fusion")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2606)
    args = parser.parse_args()

    panel = pd.read_csv(ROOT / args.panel, usecols=["company_id", "fiscal_year", "label"])
    panel["company_id"] = panel.company_id.astype(str)
    panel = panel.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    val_lstm = predictions_from_experiment(ROOT / args.lstm_validation, "validation", panel, args.fold)
    test_lstm = predictions_from_experiment(ROOT / args.lstm_test, "test", panel, args.fold)
    weight_grid = np.linspace(0.0, 1.0, 11)
    reports = {}
    test_rows = []

    for base_name, directory in (("xgboost", args.xgboost_results), ("lightgbm", args.lightgbm_results)):
        result_dir = ROOT / directory
        if args.fold == "early":
            base_val = pd.read_csv(result_dir / f"{base_name}-validation.csv")
            base_test = pd.read_csv(result_dir / f"{base_name}-test.csv")
        else:
            base_val = pd.read_csv(result_dir / "validation_predictions.csv")
            base_test = pd.read_csv(result_dir / "predictions.csv")
        base_val = base_val.rename(columns={"score": "base_score"})
        base_test = base_test.rename(columns={"score": "base_score"})
        base_val["company_id"] = base_val.company_id.astype(str)
        base_test["company_id"] = base_test.company_id.astype(str)
        val = base_val.merge(val_lstm, on=["company_id", "fiscal_year", "label"], validate="one_to_one")
        test = base_test.merge(test_lstm, on=["company_id", "fiscal_year", "label"], validate="one_to_one")
        y_val = val.label.to_numpy(dtype=int)
        y_test = test.label.to_numpy(dtype=int)
        val_base = robust_scale(val.base_score.to_numpy(), val.base_score.to_numpy())
        val_lstm_scaled = robust_scale(val.score.to_numpy(), val.score.to_numpy())
        test_base = robust_scale(val.base_score.to_numpy(), test.base_score.to_numpy())
        test_lstm_scaled = robust_scale(val.score.to_numpy(), test.score.to_numpy())
        base_val_metrics = metrics(y_val, val_base)
        selected = None
        validation_trials = []
        for weight in weight_grid:
            combined = (1 - weight) * val_base + weight * val_lstm_scaled
            result = {"lstm_weight": float(weight), **metrics(y_val, combined)}
            validation_trials.append(result)
            if result["roc_auc"] >= base_val_metrics["roc_auc"] and (selected is None or result["average_precision"] > selected["average_precision"]):
                selected = result
        if selected is None:
            selected = {"lstm_weight": 0.0, **base_val_metrics}

        weight = selected["lstm_weight"]
        blended_val = (1 - weight) * val_base + weight * val_lstm_scaled
        blended_test = (1 - weight) * test_base + weight * test_lstm_scaled
        threshold = choose_recall_threshold(y_val, blended_val, 0.8)
        predicted = blended_test >= threshold
        base_test_scaled = test_base
        reports[base_name] = {
            "selected_lstm_weight": weight,
            "validation_selected_subject_to_no_roc_auc_drop": True,
            "validation_base": base_val_metrics,
            "validation_lstm": metrics(y_val, val_lstm_scaled),
            "validation_blend": metrics(y_val, blended_val),
            "validation_grid": validation_trials,
            "test_base": metrics(y_test, base_test_scaled),
            "test_lstm_ensemble": metrics(y_test, test_lstm_scaled),
            "test_blend": metrics(y_test, blended_test),
            "test_threshold_for_validation_recall_0.8": float(threshold),
            "test_precision_at_threshold": float(precision_score(y_test, predicted, zero_division=0)),
            "test_recall_at_threshold": float(recall_score(y_test, predicted, zero_division=0)),
            "paired_company_clustered_bootstrap": cluster_bootstrap(
                y_test, base_test_scaled, blended_test, test.company_id.to_numpy(),
                replicates=args.bootstrap_replicates, seed=args.seed,
            ),
        }
        save = test[["company_id", "fiscal_year", "label"]].copy()
        save["base_model"] = base_name
        save["blend_score"] = blended_test
        save["base_score_scaled"] = base_test_scaled
        save["lstm_score_scaled"] = test_lstm_scaled
        test_rows.append(save)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(test_rows, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    split_definitions = {
        "early": {"train": "2008-2012", "validation": "2013-2014", "test": "2015-2016"},
        "late": {"train": "2008-2016", "validation": "2017-2018", "test": "2019-2021"},
    }
    (output / "metrics.json").write_text(json.dumps({
        "fold": args.fold,
        "split": split_definitions[args.fold],
        "availability_date_status": "fiscal-year-only exploratory; exact filing-availability dates are absent",
        "lstm": "validation-selected four-year raw-feature multi-head LSTM with missingness channels; five-seed mean score",
        "weight_grid": [float(w) for w in weight_grid],
        "bootstrap_replicates": args.bootstrap_replicates,
        "models": reports,
    }, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
