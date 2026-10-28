#!/usr/bin/env python3
"""Prepare an ignored local panel from the University of Peloponnese numeric release.

This adapter intentionally keeps source data local and does not infer filing dates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/uop-bankruptcy/companyfacts_dense.csv")
    parser.add_argument("--output", default="data/processed/uop_numeric_panel.csv")
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output)
    frame = pd.read_csv(source, low_memory=False)
    required = {"cik_adsh", "label", "year"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    frame["company_id"] = frame["cik_adsh"].astype(str).str.split("_").str[0]
    frame["fiscal_year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    labels = frame["label"].astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    if not labels.isin(mapping).all():
        raise ValueError(f"Unexpected label values: {sorted(labels.unique())}")
    frame["label"] = labels.map(mapping).astype(int)
    excluded = {"cik_adsh", "label", "year", "company_id", "fiscal_year"}
    feature_cols = [c for c in frame.columns if c not in excluded]
    for name in feature_cols:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    duplicate_mask = frame.duplicated(["company_id", "fiscal_year"], keep=False)
    duplicate_groups = frame.loc[duplicate_mask].groupby(["company_id", "fiscal_year"], sort=False)
    conflicting_label_groups = int((duplicate_groups.label.nunique() > 1).sum())
    duplicate_group_count = int(duplicate_groups.ngroups)
    duplicate_rows_removed = int(duplicate_mask.sum())
    # Multiple filings for one CIK and year cannot be safely ordered because
    # filing dates are absent; discard every such company-year instead of choosing
    # a convenient accession or risking duplicate observations across splits.
    frame = frame.loc[~duplicate_mask].copy()
    split_summary = {}
    for name, lower, upper in [("train", 2008, 2016), ("validation", 2017, 2018), ("test", 2019, 2021)]:
        part = frame.loc[frame.fiscal_year.between(lower, upper)]
        split_summary[name] = {
            "years": [lower, upper],
            "rows": int(len(part)),
            "positive_rows": int(part.label.sum()),
            "positive_rate": float(part.label.mean()),
            "companies": int(part.company_id.nunique()),
        }
    result = frame[["company_id", "fiscal_year", "label", *feature_cols]].sort_values(
        ["company_id", "fiscal_year"], kind="stable"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    audit = {
        "source_title": "An Open Financial Dataset for Bankruptcy Prediction",
        "source_doi": "10.26263/amitos-3539",
        "source_file": source.name,
        "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "rows": int(len(result)),
        "companies": int(result.company_id.nunique()),
        "year_min": int(result.fiscal_year.min()),
        "year_max": int(result.fiscal_year.max()),
        "positive_rows": int(result.label.sum()),
        "positive_rate": float(result.label.mean()),
        "positive_companies": int(result.loc[result.label.eq(1), "company_id"].nunique()),
        "companies_with_label_transitions": int((result.groupby("company_id").label.nunique() > 1).sum()),
        "duplicate_company_year_groups_removed": duplicate_group_count,
        "duplicate_company_year_rows_removed": duplicate_rows_removed,
        "duplicate_company_year_label_conflicts": conflicting_label_groups,
        "feature_count": len(feature_cols),
        "available_date_present": False,
        "duplicate_resolution": "excluded all repeated company-years because filing dates are absent",
        "feature_missing_cells": int(result[feature_cols].isna().sum().sum()),
        "split_summary": split_summary,
        "prepared_file": str(target),
    }
    audit_path = target.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
