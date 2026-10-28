# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js App Router, selected by the user. Use static generation where practical and deploy with the Vercel CLI.

## Users

Research readers and reviewers who want to understand the paper's task, dataset, model, evaluation, and results without first reading the full manuscript.

## Product Purpose

Explain the corporate bankruptcy prediction study section by section and make its paper, code, data provenance, results, and evaluation limits easy to find.

## Positioning

The site presents a reliability-aware temporal GNN study alongside its measured comparisons with XGBoost and LightGBM. It distinguishes local results from published results on other datasets.

## Operating Context

The site accompanies the IEEE-format manuscript in `paper/main.pdf` and the public source repository. Visitors arrive to scan the research overview, inspect the method and metrics, then open the paper or code.

## Capabilities and Constraints

- Explain the prediction target, audited cohort, chronological split, model architecture, baselines, metrics, and results.
- Include original diagrams for the panel timeline, LSTM and peer graph method, evaluation protocol, and results.
- Link to the paper PDF, source repository, and dataset record.
- Keep source-reported benchmark values separate from local experiment results.
- State that exact filing-availability dates are missing and the current results use fiscal-year ordering.
- Do not present the model as state of the art or imply that score differences on different datasets are directly comparable.
- Deploy the static site to Vercel using its CLI.

## Evidence on Hand

- Manuscript and local experiment summary: `paper/main.tex`, `paper/main.pdf`, and `paper/experiment_results.json`.
- Dataset audit: `docs/uop-data-audit.md`.
- Public repository: `https://github.com/Vedang-P/FME-Project`.
- Dataset record: `https://amitos.library.uop.gr/items/7bc5792c-2982-4928-8f75-42950fc2ef25`.

## Product Principles

- Lead with the research question and measured result.
- Make the data and method understandable through faithful diagrams.
- Preserve the difference between source-reported and locally measured values.
- Present evaluation limits with the results they qualify.

## Direction contract

**THESIS:** The site is an economic atlas of model evidence, with a legend that explains each comparison and boundary.

**OWN-WORLD:** Survey-ink blue-black, warm map-stock ground, restrained rust for the active test window, and contour-like lines that only encode actual years, model families, or graph relations. A slim atlas legend carries section wayfinding. The article column remains calm and readable.

**STORY:** Readers first see the question and the measured outcome, then move through the data, model, evaluation, results, and limitations. Visuals map the 2008–2021 year split, the LSTM and financial-peer flow, and the local model scores. Published figures on another cohort have their own labeled comparison.

**FIRST VIEWPORT:** A compact atlas masthead spans the top, a narrow contents legend anchors the left on desktop, and the main column opens with the paper title, one-sentence question, concise abstract, and an evidence plot with XGBoost, LightGBM, and temporal GNN values. The primary action opens the IEEE PDF.

**FORM:** Own-ranked candidate 1, Economic Atlas, selected by the user. Seed reference: `seed:0872e4be207844a8`.

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
