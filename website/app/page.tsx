const repository = "https://github.com/Vedang-P/FME-Project";
const paper = `${repository}/blob/main/paper/main.pdf`;
const dataset = "https://amitos.library.uop.gr/items/7bc5792c-2982-4928-8f75-42950fc2ef25";

const results = [
  { name: "XGBoost", note: "Best individual baseline", ap: 0.1141, auc: 0.8868, group: "tree", featured: true },
  { name: "LightGBM", note: "Tuned baseline", ap: 0.1116, auc: 0.8790, group: "tree" },
  { name: "Logistic regression", note: "Last-year inputs", ap: 0.0504, auc: 0.8325, group: "baseline" },
  { name: "Histogram gradient boosting", note: "Last-year inputs", ap: 0.0546, auc: 0.8028, group: "baseline" },
  { name: "MLP", note: "Flattened five-year history", ap: 0.0589, auc: 0.8154, group: "neural" },
  { name: "LSTM", note: "Five-year sequence", ap: 0.0708, auc: 0.7740, group: "neural" },
  { name: "GCN + LSTM", note: "Financial peer graph", ap: 0.0794, auc: 0.7708, group: "graph" },
  { name: "GraphSAGE + LSTM", note: "Financial peer graph", ap: 0.0698, auc: 0.7649, group: "graph" },
  { name: "Reliability-aware temporal GNN", note: "Our proposed model", ap: 0.0383, auc: 0.7515, group: "ours", featured: true },
  { name: "LightGBM + GraphSAGE", note: "Validation-selected score fusion", ap: 0.1181, auc: 0.8705, group: "fusion", featured: true },
  { name: "LightGBM + reliability GNN", note: "Validation-selected score fusion", ap: 0.1127, auc: 0.8688, group: "fusion" },
  { name: "XGBoost + reliability GNN", note: "Validation-selected score fusion", ap: 0.1133, auc: 0.8806, group: "fusion" },
];

const metrics = [
  { name: "XGBoost", ap: "0.1141", auc: "0.8868", brier: "0.0853", precision: "0.0262", recall: "0.8438", label: "Best baseline" },
  { name: "LightGBM", ap: "0.1116", auc: "0.8790", brier: "0.0527", precision: "0.0239", recall: "0.8125", label: "Baseline" },
  { name: "Logistic regression", ap: "0.0504", auc: "0.8325", brier: "0.1693", precision: "0.0161", recall: "0.8125", label: "Baseline" },
  { name: "Histogram gradient boosting", ap: "0.0546", auc: "0.8028", brier: "0.0418", precision: "0.0204", recall: "0.8594", label: "Baseline" },
  { name: "MLP", ap: "0.0589", auc: "0.8154", brier: "0.1846", precision: "0.0118", recall: "0.8281", label: "Neural" },
  { name: "LSTM", ap: "0.0708", auc: "0.7740", brier: "0.2067", precision: "0.0110", recall: "0.8750", label: "Temporal" },
  { name: "GCN + LSTM", ap: "0.0794", auc: "0.7708", brier: "0.1701", precision: "0.0112", recall: "0.8281", label: "Graph" },
  { name: "GraphSAGE + LSTM", ap: "0.0698", auc: "0.7649", brier: "0.1782", precision: "0.0113", recall: "0.8750", label: "Graph" },
  { name: "Reliability-aware temporal GNN", ap: "0.0383", auc: "0.7515", brier: "0.2162", precision: "0.0120", recall: "0.7812", label: "Ours" },
  { name: "LightGBM + GraphSAGE", ap: "0.1181", auc: "0.8705", brier: "—", precision: "0.0252", recall: "0.8125", label: "Fusion" },
  { name: "LightGBM + reliability GNN", ap: "0.1127", auc: "0.8688", brier: "—", precision: "0.0248", recall: "0.8125", label: "Fusion" },
  { name: "XGBoost + reliability GNN", ap: "0.1133", auc: "0.8806", brier: "—", precision: "0.0256", recall: "0.8438", label: "Fusion" },
];

const heroResults = [
  { name: "XGBoost", note: "Best individual baseline", ap: 0.1141, auc: 0.8868, ours: false },
  { name: "LightGBM", note: "Boosted-tree baseline", ap: 0.1116, auc: 0.8790, ours: false },
  { name: "Reliability-aware temporal GNN", note: "(Ours)", ap: 0.0383, auc: 0.7515, ours: true },
];

function Arrow() {
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3.5 10h12m-4.5-4.5L15.5 10 11 14.5" /></svg>;
}

function HeroEvidence() {
  return (
    <div className="hero-evidence" aria-label="Held-out average precision and ROC-AUC comparison">
      <div className="hero-evidence-grid">
        <span className="hero-evidence-head">Held-out results</span>
        <span className="hero-evidence-head">Average precision</span>
        <span className="hero-evidence-head">ROC-AUC</span>
        {heroResults.map((item) => (
          <div className={`hero-evidence-row ${item.ours ? "hero-evidence-row-ours" : ""}`} key={item.name}>
            <span className="hero-evidence-model">{item.ours ? <><strong>(Ours)</strong> {item.name}</> : <><strong>{item.name}</strong><small>{item.note}</small></>}</span>
            <span className="hero-score" aria-label={`Average precision ${item.ap.toFixed(4)}`}><span className="hero-score-track"><i style={{ width: `${(item.ap / 0.13) * 100}%` }} /></span>{item.ap.toFixed(4)}</span>
            <span className="hero-score" aria-label={`ROC-AUC ${item.auc.toFixed(4)}`}><span className="hero-score-track"><i style={{ width: `${((item.auc - 0.7) / 0.2) * 100}%` }} /></span>{item.auc.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div className="hero-evidence-axis"><span>AP scale: 0–0.13</span><span>ROC-AUC scale: 0.70–0.90</span></div>
      <p className="hero-evidence-note">Fiscal-year test set · higher is better for both metrics</p>
    </div>
  );
}

function AtlasMark() {
  return (
    <svg className="atlas-mark" viewBox="0 0 48 48" role="img" aria-label="FME research mark">
      <path d="M8 9.5h31M8 16.5h31M8 23.5h31M8 30.5h31M8 37.5h31" />
      <path className="atlas-mark-accent" d="M11 31c8-12 15 9 26-7" />
      <circle cx="37" cy="24" r="2.2" />
    </svg>
  );
}

function SplitRibbon() {
  return (
    <div className="split-ribbon" aria-label="Training years 2008 to 2016, validation 2017 to 2018, test 2019 to 2021">
      <div className="split-years">
        {Array.from({ length: 14 }, (_, index) => {
          const year = 2008 + index;
          const kind = year <= 2016 ? "train" : year <= 2018 ? "validation" : "test";
          return <span className={`year year-${kind}`} key={year}>{year}</span>;
        })}
      </div>
      <div className="split-key">
        <span><i className="key-train" />Train · 2008–2016</span>
        <span><i className="key-validation" />Validation · 2017–2018</span>
        <span><i className="key-test" />Test · 2019–2021</span>
      </div>
    </div>
  );
}

function MethodMap() {
  return (
    <div className="method-map">
      <svg viewBox="0 0 920 390" role="img" aria-labelledby="method-title method-desc">
        <title id="method-title">Reliability-aware temporal graph model</title>
        <desc id="method-desc">Up to five years of firm financial features feed an LSTM. At each year, financial similarity graphs provide relation-specific messages, weighted by similarity and peer-set persistence. A gate combines graph messages with the firm's temporal state to estimate distress risk.</desc>
        <defs>
          <marker id="arrow-ink" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#68746c" /></marker>
          <marker id="arrow-rust" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#a14d35" /></marker>
        </defs>
        <path className="map-arrow" d="M180 180H255" markerEnd="url(#arrow-ink)" />
        <path className="map-arrow" d="M420 180H555" markerEnd="url(#arrow-ink)" />
        <path className="map-arrow map-arrow-blue" d="M515 92L590 130" markerEnd="url(#arrow-ink)" />
        <path className="map-arrow map-arrow-rust" d="M515 275L590 222" markerEnd="url(#arrow-rust)" />
        <path className="map-arrow map-arrow-rust" d="M440 244V151" markerEnd="url(#arrow-rust)" />
        <path className="map-arrow" d="M735 180H795" markerEnd="url(#arrow-ink)" />
        <g className="map-node map-node-green">
          <rect x="24" y="125" width="156" height="110" rx="11" />
          <text className="map-title" x="102" y="163">Firm history</text>
          <text className="map-copy" x="102" y="190">Up to five annual</text>
          <text className="map-copy" x="102" y="210">financial feature vectors</text>
        </g>
        <g className="map-node map-node-green">
          <rect x="255" y="125" width="165" height="110" rx="11" />
          <text className="map-title" x="337" y="163">LSTM encoder</text>
          <text className="map-copy" x="337" y="190">Firm state at</text>
          <text className="map-copy" x="337" y="210">prediction year t</text>
        </g>
        <g className="map-node map-node-blue">
          <rect x="335" y="26" width="180" height="100" rx="11" />
          <text className="map-title" x="425" y="62">Financial peers</text>
          <text className="map-copy" x="425" y="88">Liquidity · solvency</text>
          <text className="map-copy" x="425" y="108">profitability · cash flow</text>
        </g>
        <g className="map-node map-node-rust">
          <rect x="335" y="244" width="180" height="100" rx="11" />
          <text className="map-title" x="425" y="280">Link reliability</text>
          <text className="map-copy" x="425" y="306">Similarity weighted by</text>
          <text className="map-copy" x="425" y="326">peer-set persistence</text>
        </g>
        <g className="map-node map-node-violet">
          <rect x="590" y="125" width="145" height="110" rx="11" />
          <text className="map-title" x="662" y="163">Gated messages</text>
          <text className="map-copy" x="662" y="190">Relation-specific</text>
          <text className="map-copy" x="662" y="210">aggregation</text>
        </g>
        <g className="map-node map-node-ink">
          <rect x="795" y="125" width="100" height="110" rx="11" />
          <text className="map-title" x="845" y="163">Score</text>
          <text className="map-copy" x="845" y="190">Distress</text>
          <text className="map-copy" x="845" y="210">risk</text>
        </g>
        <text className="map-foot" x="460" y="379">Peer links are statistical associations, not causal or contagion relationships.</text>
      </svg>
      <p className="figure-note">Conceptual data flow from the implemented model. Peer sets are rebuilt at each fiscal-year snapshot; scroll sideways on small screens to follow the complete chain.</p>
    </div>
  );
}

function ResultsChart() {
  const items = [...results].sort((a, b) => b.ap - a.ap);
  const start = 260;
  const width = 400;
  const aucStart = 760;
  const aucWidth = 400;
  const row = 35;
  return (
    <div className="chart-wrap">
      <svg className="results-chart" viewBox={`0 0 1220 ${items.length * row + 78}`} role="img" aria-labelledby="results-title results-desc">
        <title id="results-title">Average precision and ROC-AUC for all evaluated models</title>
        <desc id="results-desc">XGBoost is the strongest individual baseline on both metrics. The proposed reliability-aware temporal GNN is below the boosted-tree baselines.</desc>
        <text className="chart-head" x={start} y="24">Average precision · higher is better</text>
        <text className="chart-head" x={aucStart} y="24">ROC-AUC · higher is better</text>
        {[0, 0.05, 0.10, 0.15].map((tick) => <g key={`ap-${tick}`}><line className="chart-grid" x1={start + (tick / 0.15) * width} x2={start + (tick / 0.15) * width} y1="46" y2={items.length * row + 50} /><text className="chart-tick" x={start + (tick / 0.15) * width} y="43">{tick.toFixed(2)}</text></g>)}
        {[0.70, 0.75, 0.80, 0.85, 0.90].map((tick) => <g key={`auc-${tick}`}><line className="chart-grid" x1={aucStart + ((tick - 0.70) / 0.20) * aucWidth} x2={aucStart + ((tick - 0.70) / 0.20) * aucWidth} y1="46" y2={items.length * row + 50} /><text className="chart-tick" x={aucStart + ((tick - 0.70) / 0.20) * aucWidth} y="43">{tick.toFixed(2)}</text></g>)}
        {items.map((item, index) => {
          const y = index * row + 70;
          const color = item.group === "tree" ? "#287b8e" : item.group === "ours" ? "#a14d35" : item.group === "graph" ? "#3e806e" : item.group === "fusion" ? "#b77a3b" : item.group === "neural" ? "#688d78" : "#68746c";
          return <g className={`chart-row ${item.featured ? "chart-row-featured" : ""}`} key={item.name}>
            <text className="chart-model" x="0" y={y + 4}>{item.name}{item.name === "XGBoost" ? " · best individual baseline" : item.name === "LightGBM + GraphSAGE" ? " · highest AP" : item.name === "Reliability-aware temporal GNN" ? " · (Ours)" : ""}</text>
            <text className="chart-note" x="0" y={y + 17}>{item.note}</text>
            <line className="chart-track" x1={start} x2={start + width} y1={y} y2={y} />
            <line className="chart-track" x1={aucStart} x2={aucStart + aucWidth} y1={y} y2={y} />
            <circle cx={start + (item.ap / 0.15) * width} cy={y} r={item.featured ? 6 : 4.6} fill={color} />
            <circle cx={aucStart + ((item.auc - 0.70) / 0.20) * aucWidth} cy={y} r={item.featured ? 6 : 4.6} fill={color} />
            <text className="chart-value" x={start + (item.ap / 0.15) * width + 10} y={y + 4}>{item.ap.toFixed(4)}</text>
            <text className="chart-value" x={aucStart + ((item.auc - 0.70) / 0.20) * aucWidth + 10} y={y + 4}>{item.auc.toFixed(4)}</text>
          </g>;
        })}
        <text className="chart-tick" x={start} y={items.length * row + 68}>64 positive test labels · AP baseline prevalence = 0.007</text>
      </svg>
      <p className="figure-note">All 12 reported test comparisons. Fusion scores are not probabilities; see the table for Brier score, precision, and recall. Swipe horizontally on small screens to inspect every row.</p>
    </div>
  );
}

function MetricsTable() {
  return (
    <div className="table-scroll">
      <table>
        <caption>Held-out fiscal-year test metrics, 2019–2021. Thresholds target 80% recall on validation; precision and recall are reported on test.</caption>
        <thead><tr><th scope="col">Model</th><th scope="col">AP ↑</th><th scope="col">ROC-AUC ↑</th><th scope="col">Brier ↓</th><th scope="col">Test precision</th><th scope="col">Test recall</th></tr></thead>
        <tbody>{metrics.map((item) => <tr className={`row-${item.label.toLowerCase().replaceAll(" ", "-")}`} key={item.name}>
          <th scope="row">{item.label === "Ours" ? <><strong>(Ours) {item.name}</strong></> : item.label === "Best baseline" ? <strong>{item.name} · best individual baseline</strong> : item.name === "LightGBM + GraphSAGE" ? <strong>{item.name} · highest AP</strong> : item.name}</th>
          <td>{item.ap}</td><td>{item.auc}</td><td>{item.brier}</td><td>{item.precision}</td><td>{item.recall}</td>
        </tr>)}</tbody>
      </table>
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="FME Research home"><AtlasMark /><span>FME <b>RESEARCH ATLAS</b></span></a>
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#data">Data</a><a href="#model">Model</a><a href="#results">Results</a><a href="#limits">Limits</a>
        </nav>
        <details className="mobile-nav">
          <summary>Sections</summary>
          <nav aria-label="Page sections"><a href="#question">Overview</a><a href="#data">Data</a><a href="#model">Model</a><a href="#evaluation">Evaluation</a><a href="#results">Results</a><a href="#limits">Limits</a></nav>
        </details>
        <a className="header-paper" href={paper}>Read the paper <Arrow /></a>
      </header>

      <div className="page-shell" id="top">
        <aside className="contents-rail">
          <p>IN THIS STUDY</p>
          <a href="#question">Research question</a><a href="#data">Data and task</a><a href="#model">Model</a><a href="#evaluation">Evaluation</a><a href="#results">Results</a><a href="#limits">Limitations</a>
          <span className="rail-rule" />
          <span className="rail-period">2008—2021</span>
          <span className="rail-caption">Fiscal-year panel</span>
        </aside>

        <article className="article-column">
          <section className="hero-section" id="question">
            <h1>Can financial peer graphs improve distress prediction?</h1>
            <p className="hero-deck">We test whether changing peer relationships add signal beyond firm-level financial histories and strong boosted-tree baselines.</p>
            <div className="hero-answer">
              <span className="answer-pin" />
              <p><strong>In this fiscal-year holdout, they did not.</strong> XGBoost remained the strongest individual baseline. Our proposed temporal GNN did not improve on it.</p>
              <a href="#results" aria-label="Go to the measured results"><Arrow /></a>
            </div>
            <HeroEvidence />
            <SplitRibbon />
            <div className="hero-meta"><span>Public SEC-derived financial data</span><span>Next-fiscal-year label</span><span>Single seed · one test window</span></div>
          </section>

          <section className="content-section data-section" id="data">
            <div className="section-heading"><h2>Data and prediction task</h2></div>
            <div className="data-layout">
              <div className="reading-copy">
                <p>Each observation pairs a company-year with 34 numeric financial features and an inherited label for bankruptcy in the next fiscal year. We use a public SEC-derived panel because an initially considered bankruptcy dataset has labels that remain constant within each company over time.</p>
                <p>We remove 214 rows from repeated company-year filing groups because exact filing dates are unavailable. The audited panel contains 42,742 company-years from 5,612 firms, including 293 positive labels.</p>
                <a className="text-link" href={dataset}>Dataset record <Arrow /></a>
              </div>
              <div className="data-index" aria-label="Dataset summary">
                <div><strong>42,742</strong><span>company-year rows</span></div>
                <div><strong>5,612</strong><span>firms</span></div>
                <div><strong>34</strong><span>numeric features</span></div>
                <div><strong>293</strong><span>positive labels</span></div>
              </div>
            </div>
            <aside className="margin-note"><span className="note-rule" /><p>The source does not include exact filing or publication dates. A fiscal-year ordering cannot confirm that every feature was public at the prediction time.</p></aside>
          </section>

          <section className="content-section model-section" id="model">
            <div className="section-heading"><h2>What the model adds</h2></div>
            <p className="section-lede">Our proposed model represents a firm's recent history and its position among financial peers at each year.</p>
            <MethodMap />
            <div className="method-notes">
              <div><h3>Temporal state</h3><p>An LSTM encodes up to five annual feature vectors into a firm representation.</p></div>
              <div><h3>Peer relations</h3><p>Separate similarity graphs capture liquidity, solvency, and profitability or cash-flow structure.</p></div>
              <div><h3>Link reliability</h3><p>Cosine similarity is adjusted using peer-set overlap between successive graph snapshots.</p></div>
            </div>
          </section>

          <section className="content-section evaluation-section" id="evaluation">
            <div className="section-heading"><h2>How we evaluated it</h2></div>
            <p className="section-lede">Models are selected without using the future test labels, then compared on the same chronological holdout.</p>
            <SplitRibbon />
            <div className="evaluation-facts">
              <div><strong>Primary metric</strong><span>Average precision, suited to rare events</span></div>
              <div><strong>Co-required metric</strong><span>ROC-AUC, to compare overall ranking</span></div>
              <div><strong>Operating point</strong><span>Threshold selected on validation for 80% recall</span></div>
              <div><strong>Uncertainty</strong><span>Company-clustered paired intervals for score fusion</span></div>
            </div>
          </section>

          <section className="content-section results-section" id="results">
            <div className="section-heading"><h2>What the results show</h2></div>
            <p className="section-lede">The best individual baseline is XGBoost. The proposed reliability-aware temporal GNN is weaker on both headline ranking metrics.</p>
            <ResultsChart />
            <div className="complete-metrics">
              <h3>Every metric in one view</h3>
              <p>All five test metrics for every model in the results table. Each panel uses its own scale; lower Brier score is better.</p>
              <div className="complete-metrics-scroll"><img src="/heldout_metrics.png" alt="Five aligned dot plots compare average precision, ROC-AUC, Brier score, test precision, and test recall for all twelve evaluated models. XGBoost is the best individual baseline, and the proposed reliability-aware temporal GNN is marked (Ours)." /></div>
            </div>
            <div className="result-callout"><div><span>BEST INDIVIDUAL BASELINE</span><strong>XGBoost</strong></div><p>AP 0.1141 · ROC-AUC 0.8868</p></div>
            <div className="result-callout result-callout-ours"><div><span>PROPOSED MODEL · OURS</span><strong>Reliability-aware temporal GNN</strong></div><p>AP 0.0383 · ROC-AUC 0.7515</p></div>
            <p className="result-interpretation">LightGBM–GraphSAGE has the highest AP estimate, but its confidence interval includes zero and its ROC-AUC is lower. Adding our model to XGBoost does not improve either metric.</p>
            <MetricsTable />
            <div className="external-benchmark"><span>SEPARATE PUBLISHED RESULT</span><p>Alanis et al. report ROC-AUC 0.92 for XGBoost on a different cohort and protocol. It is context, not a matched comparison.</p></div>
          </section>

          <section className="content-section limits-section" id="limits">
            <div className="section-heading"><h2>What this study can support</h2></div>
            <div className="limits-layout">
              <p className="section-lede">The current evidence supports a fiscal-year exploratory comparison, not a filing-time claim or state-of-the-art result.</p>
              <ul>
                <li>Filing availability dates are missing, so point-in-time leakage cannot be ruled out.</li>
                <li>The test years share some firms with earlier years; company-disjoint performance is not established.</li>
                <li>Neural results use one seed and one future test window.</li>
                <li>Financial similarity edges describe predictive association, not causal contagion.</li>
                <li>The proposed model does not beat the best individual baseline on AP and ROC-AUC.</li>
              </ul>
            </div>
          </section>

          <section className="resources-section" id="resources">
            <h2>Read the study</h2>
            <p>Explore the full IEEE-format paper, the experiment code, and the source-specific data audit.</p>
            <div className="resource-links"><a href={paper}>Open paper PDF <Arrow /></a><a href={repository}>Browse source repository <Arrow /></a><a href={dataset}>Dataset documentation <Arrow /></a></div>
          </section>
        </article>
      </div>

      <footer className="site-footer"><span>FME RESEARCH ATLAS</span><span>Financial distress prediction · Fiscal-year evaluation</span><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}
