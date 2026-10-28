---
name: Economic Atlas — Financial Distress Research
description: Editorial atlas for explaining financial distress research, evaluation evidence, and limits.
colors:
  survey-ink: "#202d32"
  map-stock: "#f4f1e9"
  bright-stock: "#fbfaf6"
  quiet-ink: "#59645f"
  atlas-rule: "#d5d3c8"
  peer-green: "#3e806e"
  peer-green-wash: "#e4eee7"
  signal-blue: "#287b8e"
  signal-blue-wash: "#e5edf0"
  test-rust: "#a14d35"
  test-rust-wash: "#f1e6df"
  muted-blue-gray: "#4e6683"
  muted-lilac-wash: "#eee8f2"
  train-sage: "#63877a"
  validation-ochre: "#d39a48"
  map-stock-shade: "#f5f4ee"
  map-ink-wash: "#ecebe5"
  ours-row-wash: "#f4eae4"
  ours-row-ink: "#733a27"
  chart-fusion: "#b77a3b"
typography:
  display:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "clamp(42px, 5.8vw, 71px)"
    fontWeight: 400
    lineHeight: 0.98
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "clamp(30px, 3.6vw, 42px)"
    fontWeight: 400
    lineHeight: 1.08
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "22px"
    fontWeight: 500
  body:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.75
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "12px"
    fontWeight: 500
rounded:
  key-mark: "1px"
  scrollbar: "8px"
  diagram-node: "11px"
  marker: "50%"
spacing:
  compact: "8px"
  label: "12px"
  inline: "18px"
  reading: "24px"
  section: "58px"
components:
  nav-link:
    textColor: "{colors.quiet-ink}"
    typography: "{typography.label}"
    padding: "9px 0"
  paper-link:
    textColor: "{colors.test-rust}"
    typography: "{typography.label}"
    padding: "4px 0"
---

# Design System: Economic Atlas — Financial Distress Research

## Overview

**Creative North Star: “Economic Atlas”**

Economic Atlas treats a research page like a field guide to evidence: survey-ink blue-black type sits on warm map-stock, with fine rules and carefully labeled marks giving structure to the reading path. The desktop composition pairs a slim contents legend with a calm article column. On small screens, the legend yields to a compact section menu and wide evidence regions remain scrollable so their labels and comparisons stay intact.

The visual language is factual and restrained. Green, blue, ochre, and rust distinguish actual data splits, model families, or the proposed model; every mark is explained by a nearby label or legend. The page leads with a plain account of the measured outcome: XGBoost is the best individual baseline, and the proposed temporal GNN marked “(Ours)” scores lower in this local fiscal-year evaluation. Source-reported results from another cohort remain separately labeled from local measurements.

**Key Characteristics:**
- Warm paper surfaces, survey-ink type, and thin map-like dividers.
- A narrow legend and readable article measure organize a long-form research narrative.
- Restrained color encodes real years, model families, and comparison status.
- Results and limitations appear together, with source-reported and locally measured evidence kept distinct.

## Colors

The palette is warm and low-glare, with survey-ink for primary text and muted data colors reserved for marks that carry meaning.

### Primary
- **Survey Ink** (`#202d32`): Main text, diagram titles, and the site mark.
- **Test Rust** (`#a14d35`): The active test period, “(Ours)” result emphasis, section markers, text links, focus outlines, and scrollbar thumb.
- **Signal Blue** (`#287b8e`): Blue-coded comparison marks and financial-peer connections.

### Secondary
- **Peer Green** (`#3e806e`): Green-coded model and method marks; paired with its pale wash for diagram nodes.
- **Muted Blue-Gray** (`#4e6683`): The blue-gray gated-message node; its pale lilac wash is `#eee8f2`.
- **Validation Ochre** (`#d39a48`): The validation period in the year split.

### Neutral
- **Map Stock** (`#f4f1e9`): Page background.
- **Bright Stock** (`#fbfaf6`): Article surface and compact navigation menu.
- **Quiet Ink** (`#59645f`): Supporting copy and captions.
- **Atlas Rule** (`#d5d3c8`): Dividers and section boundaries.
- **Map Stock Shade** (`#f5f4ee`): Method and evaluation ribbons.
- **Ours Row Wash** (`#f4eae4`) and **Ours Row Ink** (`#733a27`): The proposed-model row in the complete results table.
- **Train Sage** (`#63877a`): Training-year marks.
- **Chart Fusion** (`#b77a3b`): Fusion-model chart marks.

**The Evidence Color Rule.** Color identifies an actual data split, model family, or proposed-model result. Keep the slim legend and nearby labels so color never carries the comparison alone.

## Typography

**Display Font:** Newsreader (with Georgia, serif fallback)

**Body Font:** DM Sans (with sans-serif fallback)
**Label/Mono Font:** IBM Plex Mono (with ui-monospace, monospace fallback)

**Character:** Newsreader gives headlines and short section leads an editorial, report-like voice. DM Sans carries reading copy; IBM Plex Mono sets compact navigation, metrics, dates, and chart labels with a measured instrument-panel quality.

### Hierarchy
- **Display** (400, `clamp(42px, 5.8vw, 71px)`, `0.98`): The main research question, with tight `-0.035em` tracking.
- **Headline** (400, `clamp(30px, 3.6vw, 42px)`, `1.08`): Major article section headings, with `-0.025em` tracking.
- **Title** (500, `22px`): Smaller figure and results subsection headings.
- **Body** (400, `14px`, `1.75`): Main article copy, generally constrained to `65ch`; shorter interpretations use `58ch`.
- **Label** (500, `12px`): IBM Plex Mono for navigation and chart/table headers. Most labels use normal tracking; the brand uses `.04em`, and the contents-rail label uses `.08em`.

**The Measured Type Rule.** Keep scores, table values, and axis ticks in tabular or monospaced forms; reserve Newsreader for the editorial reading hierarchy.

## Layout

At desktop widths, a centered grid places a `174px` sticky contents rail beside an article column capped at `860px`, with a fluid gap from `46px` to `128px`. The sticky masthead is `74px` tall. The main article uses a bright stock surface with responsive inline padding from `26px` to `76px`. Body sections use generous vertical spacing (`58px` top and `61px` bottom); labels and metric rows are denser, usually `12px` or `14px` apart.

At `1050px`, the rail narrows to `140px`, the article column to `760px`, and the gap to `42px`. At `760px`, the side rail and desktop navigation disappear, the header becomes `64px`, the mobile section menu appears, and the article uses `23px` side padding. Data and limits layouts stack. The chart, method map, and full metrics remain in horizontal scroll regions on narrow screens instead of compressing their plotted labels or columns.

## Elevation & Depth

The page is flat by default. Fine borders, warm surface changes, and aligned rules define sections and tables; the only structural shadow is the compact mobile navigation menu (`0 9px 24px rgb(32 45 50 / 10%)`). A small rust marker in the lead answer has a restrained four-pixel wash behind it. There is no general card-shadow system.

## Shapes

The form language is mostly square and ruled: article sections, evidence plots, and tables use thin borders rather than rounded cards. Tiny legend keys use a `1px` radius, scrollbar thumbs use `8px`, and SVG method-map nodes use `11px` corners. Small circular marks use a full `50%` radius. Avoid rounding the reading column or inventing card silhouettes.

## Components

### Header and section navigation
The navigation is quiet and editorial, with a clear route through the research sections.
- Desktop: a sticky bright-stock masthead, compact mono brand, inline section links, and an underlined “Read the paper” link.
- Mobile: native disclosure navigation labeled “Sections”; the desktop contents rail is hidden.
- Link hover changes to Test Rust. Keyboard focus uses a `2px` Test Rust outline with `4px` offset.

### Paper and dataset links
These are text links, not buttons. They use Test Rust, a thin underline, compact `12px` type, and a small inline arrow; the masthead paper link uses a subdued rule.

### Year split and slim legend
Fourteen years are arranged from 2008 through 2021. Sage, ochre, and rust bottom rules map the training, validation, and test periods, with a compact mono legend spelling out each range. Preserve the key and its labels whenever the year marks are reused.

### Evidence charts and method map
Results are plotted as factual points and fine chart tracks, with metric scales, values, and explanatory notes visible. Blue, green, rust, ochre, and other muted hues distinguish existing model groups. The proposed model is explicitly labeled “(Ours)” and rust-highlighted; the chart and nearby prose state that it is below XGBoost. The method map uses softly tinted, outlined SVG nodes with thin directional lines and an explicit caveat that peer links are statistical associations, not causal relationships. On mobile, both regions scroll horizontally.

### Results table
The complete metrics table uses tabular numeric columns, mono column headings, thin horizontal rules, and an overflow-x scroll region. Its caption names the fiscal-year test window and thresholding context. The proposed-model row has a pale rust wash and dark rust text. Keep it readable without implying that it outranks the stronger baselines.

## Do's and Don'ts

### Do:
- **Do** use Map Stock and Bright Stock as the page and article surfaces, with Survey Ink for the primary text.
- **Do** label each data color with its year split or model-family meaning in a slim legend or adjacent copy.
- **Do** state clearly that XGBoost is the best individual baseline and the proposed “(Ours)” model is lower in this local evaluation.
- **Do** preserve the source-reported versus locally measured distinction in charts, captions, and prose.
- **Do** keep mobile charts and tables in their existing horizontal scroll regions.
- **Do** retain visible keyboard focus, the readable article measure, and the reduced-motion behavior.

### Don't:
- **Don't** use Test Rust as decoration across large areas; it marks the test window, the proposed model, and navigation states.
- **Don't** make the local proposed-model result look like the leading baseline.
- **Don't** merge source-reported comparisons with local holdout measurements.
- **Don't** remove chart scales, table captions, axis labels, or the color legend when reusing evidence visuals.
- **Don't** add button, form, or card patterns; the current interface uses links, navigation, data regions, and an article layout.
