# SEC-derived bankruptcy panel audit

## Source and terms

- Record: [An Open Financial Dataset for Bankruptcy Prediction](https://amitos.library.uop.gr/items/7bc5792c-2982-4928-8f75-42950fc2ef25)
- Author: Angelos Potiriadis, University of Peloponnese, 2026
- DOI: `10.26263/amitos-3539`
- Downloaded variant: `numeric-dataset-variants.zip`, file `variants/companyfacts_dense.csv`
- Local file SHA-256: `a6e516ffd7e444d7c6c8f3b4e8ee74203884440cc7a752a864ba5e57d108a60b`
- The attached repository license notice retains copyright and requires attribution. This project does not include the raw data or filing-level prediction files in version control.

The thesis describes its inherited label as a next-fiscal-year bankruptcy outcome. Its released numeric CSV has the composite filing key, label, year, and financial facts, but does not include exact SEC filing/publication dates. Consequently, a strict as-of-filing split cannot be verified from the release alone.

## Local panel audit

| Check | Observed |
|---|---:|
| Source rows | 42,956 (thesis-reported) |
| Source years | 2008--2021 |
| Financial features | 34 |
| Repeated company-year groups | 103 |
| Rows excluded from repeated groups | 214 |
| Conflicting-label repeated groups | 1 |
| Audited rows | 42,742 |
| Audited companies | 5,612 |
| Positive rows | 293 (0.6855%) |
| Firms with positive rows | 285 |
| Firms with label transitions | 242 |
| Missing feature cells | 591,647 |
| Exact filing dates available | No |

Every row in a repeated company-year group is excluded. No filing is chosen by accession ordering because the data do not include filing dates; one of those groups has inconsistent target values. The resulting panel intentionally differs from the source thesis's 42,956-row cohort.

## Chronological cohorts used in this report

| Split | Years | Rows | Positive labels | Rate |
|---|---|---:|---:|---:|
| Train | 2008--2016 | 27,676 | 200 | 0.7226% |
| Validation | 2017--2018 | 6,034 | 29 | 0.4806% |
| Test | 2019--2021 | 9,032 | 64 | 0.7086% |

The test split shares issuers with earlier splits. It represents a future-year holdout, not a company-disjoint cohort. As the filing dates are missing, results are exploratory and cannot certify the predeclared filing-time success criterion.

## Preparation

After downloading the source file to `data/raw/uop-bankruptcy/companyfacts_dense.csv`, run:

```bash
python scripts/prepare_uop_numeric_panel.py
```

This writes the ignored `data/processed/uop_numeric_panel.csv` and its audit JSON. The adapter verifies the required fields, binary label encoding, feature types, repeated company-years, label transitions, source hash, and chronological split counts. Do not upload the source data or derived filing-level scores to the public repository.
