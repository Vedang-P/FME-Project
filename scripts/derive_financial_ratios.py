#!/usr/bin/env python3
"""Create a local ratio-enriched feature variant for controlled ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.where(denominator.abs() > 1e-12)
    ratio = numerator / den
    return ratio.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--output", default="data/processed/uop_numeric_ratio_panel.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    f = frame
    frame["current_ratio"] = safe_divide(f.current_assets, f.current_liabilities)
    frame["liabilities_to_assets"] = safe_divide(f.liabilities, f.assets)
    frame["long_term_debt_to_assets"] = safe_divide(f.long_term_debt, f.assets)
    frame["debt_to_equity"] = safe_divide(f.long_term_debt, f.stockholders_equity)
    frame["return_on_assets"] = safe_divide(f.net_income, f.assets)
    frame["operating_margin"] = safe_divide(f.operating_income, f.revenue)
    frame["net_margin"] = safe_divide(f.net_income, f.revenue)
    frame["gross_margin"] = safe_divide(f.gross_profit, f.revenue)
    frame["cash_to_current_liabilities"] = safe_divide(f.cash_and_cash_equivalents, f.current_liabilities)
    frame["operating_cash_flow_to_liabilities"] = safe_divide(f.net_cash_from_operating_activities, f.liabilities)
    frame["working_capital_to_assets"] = safe_divide(f.current_assets - f.current_liabilities, f.assets)
    frame["capex_to_operating_cash_flow"] = safe_divide(f.capital_expenditures.abs(), f.net_cash_from_operating_activities.abs())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} rows and {frame.shape[1] - 3} numeric feature columns to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
