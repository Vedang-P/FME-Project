"""Generate the manuscript's plots from the recorded result ledger and local scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent
LEDGER = json.loads((ROOT / "paper/experiment_results.json").read_text())

COLORS = {
    "XGBoost, validation-tuned": "#b23a32",
    "LightGBM, validation-tuned": "#287b8e",
    "Logistic regression, last year": "#737b85",
    "Histogram gradient boosting, last year": "#ad7e2c",
    "MLP, flattened five-year history": "#92978f",
    "LSTM": "#3e806e",
    "GCN + LSTM": "#a14d35",
    "Delta (GCN + LSTM, proposed)": "#a14d35",
    "GraphSAGE + LSTM": "#74a99d",
    "Reliability-aware temporal GNN": "#8b6f9f",
    "LightGBM + GraphSAGE fusion (highest AP)": "#d3813f",
    "LightGBM + reliability-aware GNN fusion": "#9a6f8c",
    "XGBoost + reliability-aware GNN fusion": "#536c9b",
}

def fusion_row(record: dict) -> dict:
    return {
        "pr_auc": record["test_blend_pr_auc"],
        "roc_auc": record["test_blend_roc_auc"],
        "precision": record["test_precision_at_validation_threshold"],
        "recall": record["test_recall_at_validation_threshold"],
        # Score fusions are not calibrated probabilities, so Brier is undefined.
    }


ROWS = [
    ("XGBoost, validation-tuned", LEDGER["xgboost"]["test_metrics"]),
    ("LightGBM, validation-tuned", LEDGER["lightgbm"]["test_metrics"]),
    ("Logistic regression, last year", LEDGER["models"]["logistic_regression_last_year"]),
    ("Histogram gradient boosting, last year", LEDGER["models"]["hist_gradient_boosting_last_year"]),
    ("MLP, flattened five-year history", LEDGER["models"]["mlp_flattened_history"]),
    ("LSTM", LEDGER["models"]["lstm"]),
    ("Delta (GCN + LSTM, proposed)", LEDGER["models"]["gcn"]),
    ("GraphSAGE + LSTM", LEDGER["models"]["graphsage"]),
    ("Reliability-aware temporal GNN", LEDGER["models"]["reliability_temporal_gnn"]),
    ("LightGBM + GraphSAGE fusion (highest AP)", fusion_row(LEDGER["score_fusion_lightgbm"]["graphsage"])),
    ("LightGBM + reliability-aware GNN fusion", fusion_row(LEDGER["score_fusion_lightgbm"]["reliability_temporal_gnn"])),
    ("XGBoost + reliability-aware GNN fusion", fusion_row(LEDGER["score_fusion_xgboost"]["xgboost__reliability_temporal_gnn"])),
]


def export(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metric_summary() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": "#293541"})
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 5.2), sharey=True, gridspec_kw={"wspace": 0.24})
    metrics = [
        ("pr_auc", "Average precision", (0, 0.13), [0, 0.05, 0.10]),
        ("roc_auc", "ROC-AUC", (0.70, 0.92), [0.70, 0.80, 0.90]),
        ("brier", "Brier score", (0, 0.24), [0, 0.10, 0.20]),
        ("precision", "Test precision\n(val-targeted threshold)", (0, 0.045), [0, 0.02, 0.04]),
        ("recall", "Test recall", (0.72, 0.96), [0.75, 0.85, 0.95]),
    ]
    y = np.arange(len(ROWS))
    labels = [name for name, _ in ROWS]
    for ax, (key, title, limits, ticks) in zip(axes, metrics):
        for index, (name, values) in enumerate(ROWS):
            value = values.get(key)
            if value is None:
                continue
            ax.scatter(value, index, s=34, color=COLORS[name], zorder=3, edgecolor="white", linewidth=0.45)
        ax.set_title(title, fontsize=8.5, weight="semibold", pad=10, loc="left")
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=7.5)
        ax.grid(axis="x", color="#d8d9d5", linewidth=0.7)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#aab0b0")
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.6)
    axes[0].tick_params(axis="y", pad=7)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].get_yticklabels()[6].set_fontweight("bold")
    axes[0].get_yticklabels()[9].set_fontweight("bold")
    axes[0].set_ylim(len(ROWS) - 0.35, -0.65)
    fig.suptitle("Held-out performance across the complete results table", x=0.205, y=0.985, ha="left", fontsize=12, weight="bold", color="#1c2a34")
    fig.text(0.205, 0.935, "One dot per model and metric. Higher is better except for Brier; fused scores are not calibrated probabilities.", ha="left", fontsize=8.2, color="#57636a")
    fig.text(0.205, 0.015, "Bold rows: strongest individual baseline, highest AP point estimate, and proposed model. Brier is not defined for score fusions.", ha="left", fontsize=8, color="#57636a")
    fig.subplots_adjust(left=0.205, right=0.99, top=0.82, bottom=0.08)
    export(fig, "heldout_metrics")


def load_curve_scores() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    neural_path = ROOT / "results/uop-seed42-corrected/predictions.csv"
    neural = pd.read_csv(neural_path)
    column_map = {
        "LSTM": "lstm",
        "Delta (GCN + LSTM, proposed)": "gcn",
        "Reliability-aware temporal GNN": "reliability_temporal_gnn",
        "GraphSAGE + LSTM": "graphsage",
    }
    for label, model in column_map.items():
        rows = neural.loc[neural["model"] == model].sort_values(["company_id", "year"])
        scores[label] = (rows["label"].to_numpy(), rows["score"].to_numpy())
    for label, path in [
        ("XGBoost, validation-tuned", ROOT / "results/uop-xgb-sweep/predictions.csv"),
        ("LightGBM, validation-tuned", ROOT / "results/uop-lgbm-sweep/predictions.csv"),
    ]:
        frame = pd.read_csv(path).sort_values(["company_id", "fiscal_year"])
        scores[label] = (frame["label"].to_numpy(), frame["score"].to_numpy())
    return scores


def ranking_curves() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9.5})
    scores = load_curve_scores()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    for label, (truth, prediction) in scores.items():
        expected = LEDGER["models"].get({"LSTM": "lstm", "Delta (GCN + LSTM, proposed)": "gcn", "Reliability-aware temporal GNN": "reliability_temporal_gnn", "GraphSAGE + LSTM": "graphsage"}.get(label, ""), None)
        if label == "XGBoost, validation-tuned":
            expected = LEDGER["xgboost"]["test_metrics"]
        elif label == "LightGBM, validation-tuned":
            expected = LEDGER["lightgbm"]["test_metrics"]
        if expected and (abs(average_precision_score(truth, prediction) - expected["pr_auc"]) > 1e-4 or abs(roc_auc_score(truth, prediction) - expected["roc_auc"]) > 1e-4):
            raise ValueError(f"Prediction artifact for {label} does not match the recorded metric ledger")
        short = {
            "XGBoost, validation-tuned": "XGBoost",
            "LightGBM, validation-tuned": "LightGBM",
            "Delta (GCN + LSTM, proposed)": "Delta (GCN + LSTM)",
        }.get(label, label)
        precision, recall, _ = precision_recall_curve(truth, prediction)
        false_positive_rate, true_positive_rate, _ = roc_curve(truth, prediction)
        axes[0].plot(recall, precision, label=short, color=COLORS[label], linewidth=1.35)
        axes[1].plot(false_positive_rate, true_positive_rate, label=short, color=COLORS[label], linewidth=1.35)
    prevalence = 64 / 9032
    axes[0].axhline(prevalence, color="#92999a", linestyle="--", linewidth=0.9, label=f"Prevalence ({prevalence:.3f})")
    axes[0].set_title("Precision–recall", loc="left", weight="semibold")
    axes[1].set_title("ROC", loc="left", weight="semibold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    for ax in axes:
        ax.grid(color="#e1e2df", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#9da6a8")
        ax.spines["left"].set_color("#9da6a8")
        ax.legend(frameon=False, fontsize=6.5, loc="best", handlelength=1.5, labelspacing=0.35)
    fig.suptitle("Ranking curves on the 2019–2021 holdout", x=0.08, y=1.03, ha="left", fontsize=11, weight="bold", color="#1c2a34")
    fig.text(0.08, -0.035, "Curves use saved test-set scores (n = 9,032; 64 positives); AP values are reported in the metric figure.", fontsize=7.5, color="#57636a")
    fig.tight_layout()
    export(fig, "ranking_curves")


def method_diagram() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.25, 1.45, 2.2, 1.15, "Firm history", "Up to five annual\nfinancial vectors", "#e8efe9", "#3e806e"),
        (3.0, 1.45, 2.05, 1.15, "LSTM encoder", "Firm state at year t", "#e8efe9", "#3e806e"),
        (5.65, 2.85, 2.25, 0.9, "Peer graph at year t", "Financial-similarity links", "#e9edf1", "#287b8e"),
        (5.65, 1.15, 2.25, 1.15, "GCN aggregation", "Self state + weighted\npeer-state mean", "#eee9f3", "#7651a1"),
        (10.75, 1.45, 1.05, 1.15, "Score", "Distress\nrisk", "#f1e9e7", "#b23a32"),
    ]
    centers = {}
    for x, y, w, h, title, body, fill, edge in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.12", facecolor=fill, edgecolor=edge, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=9, weight="bold", color="#23313b")
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=7.5, color="#45535c", linespacing=1.25)
        centers[title] = (x, y, w, h)
    def arrow(a, b, color="#728087", rad=0.0):
        x, y, w, h = centers[a]
        X, Y, W, H = centers[b]
        start = (x + w, y + h / 2) if X > x else (x + w / 2, y)
        end = (X, Y + H / 2) if X > x else (X + W / 2, Y + H)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1.1, connectionstyle=f"arc3,rad={rad}"))
    arrow("Firm history", "LSTM encoder")
    arrow("LSTM encoder", "GCN aggregation")
    ax.add_patch(FancyArrowPatch((6.78, 2.85), (6.78, 2.30), arrowstyle="-|>", mutation_scale=10, color="#287b8e", linewidth=1.1))
    arrow("GCN aggregation", "Score", "#7651a1")
    ax.text(6.77, 0.53, "Peer links are predictive associations, not causal or contagion edges.", ha="center", fontsize=7.2, color="#57636a")
    fig.tight_layout(pad=0.4)
    export(fig, "method_overview")


def split_timeline() -> None:
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    years = np.arange(2008, 2022)
    colors = ["#63877a" if year <= 2016 else "#d39a48" if year <= 2018 else "#b23a32" for year in years]
    ax.bar(years, np.ones_like(years), width=0.88, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlim(2007.4, 2021.6)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=0, fontsize=7.2)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#9da6a8")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_title("Fiscal-year evaluation windows", loc="left", fontsize=10, weight="bold", color="#1c2a34", pad=8)
    handles = [
        Patch(facecolor="#63877a", label="Train · 2008–2016 · 27,676 rows · 200 positives"),
        Patch(facecolor="#d39a48", label="Validation · 2017–2018 · 6,034 rows · 29 positives"),
        Patch(facecolor="#b23a32", label="Test · 2019–2021 · 9,032 rows · 64 positives"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.02, 0.10), fontsize=7.1, handlelength=0.9, handleheight=0.8, labelspacing=0.35)
    fig.text(0.02, 0.015, "Exact filing availability dates are not released; fiscal years do not guarantee point-in-time availability.", fontsize=7.1, color="#57636a")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.53)
    export(fig, "evaluation_split")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    metric_summary()
    ranking_curves()
    method_diagram()
    split_timeline()
    print("Generated: heldout_metrics, ranking_curves, method_overview, evaluation_split (PDF and PNG)")
