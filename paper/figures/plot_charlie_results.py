"""Plot paired Charlie score-fusion changes from the tracked result ledger."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEDGER = json.loads((ROOT / "paper/charlie_results.json").read_text())
COLORS = {"xgboost": "#287b8e", "lightgbm": "#b77a3b"}
NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True, gridspec_kw={"wspace": 0.24})
    groups = [
        ("early", "2015–2016", "xgboost"),
        ("early", "2015–2016", "lightgbm"),
        ("late", "2019–2021", "xgboost"),
        ("late", "2019–2021", "lightgbm"),
    ]
    y_positions = np.arange(len(groups))
    for ax, metric, title in zip(
        axes,
        ["average_precision", "roc_auc"],
        ["Change in average precision", "Change in ROC-AUC"],
    ):
        for row, (window, years, model) in enumerate(groups):
            outcome = LEDGER["windows"][window]["models"][model]["paired_difference"][metric]
            mean = outcome["mean_difference"]
            lo, hi = outcome["ci95"]
            ax.errorbar(
                mean,
                row,
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=COLORS[model],
                ecolor=COLORS[model],
                capsize=2.5,
                markersize=5,
                linewidth=1.1,
                zorder=3,
            )
        ax.axvline(0, color="#6c7478", linewidth=0.9, linestyle="--")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Charlie minus tree baseline")
        ax.grid(axis="x", color="#e1e2df", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9da6a8")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions, [f"{years} · {NAMES[model]}" for _, years, model in groups])
    axes[0].invert_yaxis()
    fig.suptitle("Paired change after adding the LSTM score", x=0.08, y=1.04, ha="left", fontsize=11, weight="bold", color="#1c2a34")
    fig.text(0.08, -0.02, "Points show paired mean differences; bars show 95% company-clustered bootstrap intervals. Zero indicates no change.", fontsize=7.5, color="#57636a")
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.18, top=0.81, wspace=0.24)
    figure_dir = Path(__file__).resolve().parent
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 240})]:
        fig.savefig(figure_dir / f"charlie_comparison.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
