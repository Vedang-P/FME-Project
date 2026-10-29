#!/usr/bin/env python3
"""Create a compact, tracked Charlie results ledger from local metric files.

Raw datasets, per-firm predictions, and full experiment outputs remain local.
This script copies only aggregate metrics and paired confidence intervals into
paper/charlie_results.json for paper and figure provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_window(name: str, path: str) -> dict:
    report = json.loads((ROOT / path).read_text())
    models = {}
    for base_name, item in report["models"].items():
        base, blend = item["test_base"], item["test_blend"]
        paired = item["paired_company_clustered_bootstrap"]
        models[base_name] = {
            "selected_lstm_weight": item["selected_lstm_weight"],
            "test_n": base["n"],
            "test_positives": base["positives"],
            "base": {"average_precision": base["average_precision"], "roc_auc": base["roc_auc"]},
            "charlie": {"average_precision": blend["average_precision"], "roc_auc": blend["roc_auc"]},
            "paired_difference": {
                "average_precision": paired["average_precision"],
                "roc_auc": paired["roc_auc"],
            },
        }
    return {"split": report["split"], "models": models}


def main() -> None:
    result = {
        "method": "Charlie",
        "definition": "Validation-selected convex score fusion of a multi-head temporal LSTM and a boosted-tree model.",
        "selection": {
            "score_scaling": "Each model's score is centered by its validation median and divided by its validation 95th-minus-5th percentile range.",
            "lstm_weight_grid": [round(i / 10, 1) for i in range(11)],
            "rule": "Maximize validation average precision subject to ROC-AUC not falling below the tree model.",
            "test_used_for_selection": False,
        },
        "lstm": "UOP: four-year raw-feature multi-head LSTM with missingness channels for the late window; five-seed mean score. The early window uses its validation-selected five-year multi-head attention LSTM; five-seed mean score.",
        "availability_date_status": "Fiscal-year exploratory. Exact filing-availability dates are absent.",
        "windows": {
            "early": load_window("early", "results/multihead-lstm-fusion-early/metrics.json"),
            "late": load_window("late", "results/multihead-lstm-fusion/metrics.json"),
        },
        "provenance": {
            "aggregate_sources": [
                "results/multihead-lstm-fusion-early/metrics.json",
                "results/multihead-lstm-fusion/metrics.json",
            ],
            "evaluation_script": "scripts/evaluate_lstm_fusion.py",
            "lstm_training_script": "scripts/run_multihead_lstm_sweep.py",
            "raw_results_and_predictions_are_not_committed": True,
        },
        "interpretation": "Late-window XGBoost fusion increases AP with a paired company-clustered 95% interval above zero; its ROC-AUC interval includes zero. Early-window AP intervals include zero, and XGBoost-fusion ROC-AUC is lower by point estimate. These two windows do not establish repeatable gains on both required metrics.",
    }
    destination = ROOT / "paper/charlie_results.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
