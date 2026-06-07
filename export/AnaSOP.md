# AnaSOP
Analysis Standard Operating Procedure

This document records the **human-designed analysis procedure** for the project.
It explains the **research objective, conceptual framework, estimands, and analytical workflow**.

AnaSOP serves two purposes:

1. Guide researchers on how to reproduce and extend the analysis.
2. Provide contextual reasoning for automated research systems such as Jiazi.

This document may evolve as the research progresses.

---

## 1. Research Objective

### Research Question

This study addresses one central question with six sub-questions:

1. Has the volume of Energy Price news coverage grown relative to total news coverage between 2021 and 2026?
2. Have Energy Price articles become semantically more focused (concrete) over time, as measured by their mean TMPT relevance score?
3. Which countries are most prominently associated with U.S. Energy Price topics in news coverage, and how does their geographic distribution compare to the United States?
4. Which of the top 30 countries exhibit anomalous news coverage intensity during the study period, and when do these anomalies occur?
5. What are the high-impact international events identified by TF-IDF analysis of anomaly windows for top 30 countries?
6. What is the measured impact of these high-impact events on energy ETF prices, and how do different ETFs respond?

### Scope of Analysis

- **Unit of observation**: Individual news articles, aggregated to daily and monthly time series; ETF closing prices aggregated to 14-day rolling windows.
- **Topics**: Energy Price relevance defined by 10 TMPT keywords (see §4). Country analysis covers top 30 countries by Energy Price article mention rate. Event analysis covers 8 high-impact events identified in `event.txt`.
- **Period**: 2021-09-01 to 2026-05-01 (approximately 56 months).
- **Universe**: U.S. English news articles, deduplicated; five U.S. energy ETFs (ICLN, IXC, VDE, XLE, XOP).

### Study Design Declaration

This is an **observational associational study**. The analysis documents the existence, magnitude, and temporal structure of associations between Energy Price news coverage, international events, and energy ETF price movements. Results describe co-occurrence patterns, temporal anomalies, and regression coefficients; they do not establish causal relationships.

---

## 2. Theoretical Background / Conceptual Framework

### 2.1 Literature Context

Energy prices are jointly determined by supply fundamentals, geopolitical risk, and financial market expectations. News media serve as a real-time transmission mechanism: geopolitical shocks (e.g., conflicts in oil-producing regions, sanctions regimes, trade disruptions) are first absorbed into public discourse before being priced into energy assets. Tracking how Energy Price coverage evolves in mass media — and which international events co-occur with spikes in that coverage — provides a systematic lens on how geopolitical information flows through to energy markets.

Prior work on news-based economic uncertainty indices (Baker, Bloom & Davis 2016) and commodity price forecasting from text (Qin et al. 2022) establishes that news sentiment and topic intensity carry predictive content for asset prices. The present study extends this line of inquiry by combining news-based event detection with direct regression onto ETF price levels.

### 2.2 Conceptual Channels

Two conceptual channels frame this study:

**Channel 1 — Coverage Growth as Attention Signal**
The share of Energy Price articles in total news coverage proxies aggregate media attention to energy pricing. A rising share indicates increasing editorial attention to energy costs, tariffs, and market pricing — consistent with heightened public concern about energy affordability and supply security.

**Channel 2 — Geopolitical Event Transmission**
International events (conflicts, diplomatic disruptions, supply shocks) associated with major oil- and gas-producing or transit countries generate distinctive vocabulary in Energy Price news. TF-IDF identifies these event signatures in anomaly windows. OLS regression then quantifies the association between event news intensity and ETF price levels, capturing the degree to which energy financial markets co-move with geopolitical news flows.

### 2.3 Institutional Context

The study period (2021–2026) spans several structural breaks relevant to energy pricing:

- **Pre-shock baseline (2021–early 2022)**: Energy prices elevated post-COVID recovery; supply chains under stress.
- **Russia-Ukraine conflict onset (Feb 2022)**: Acute natural gas and oil supply disruptions in Europe; energy price volatility surge.
- **Sanctions and energy reconfiguration (2022–2023)**: European LNG imports, Middle East supply realignment, Iranian sanctions enforcement.
- **Clean energy transition acceleration (2023–2024)**: IRA implementation, renewable capacity additions affecting electricity price dynamics.
- **Geopolitical reconfiguration (2024–2026)**: Ongoing U.S.-China competition, Middle East tensions (Iran–Israel), Brazil energy policy shifts, Germany industrial energy cost debate.

These structural episodes provide natural temporal variation for interpreting observed event impacts on ETF prices.

---

## 3. Data Overview

### 3.1 Data Sources

| Dataset | File Path | Content | Dimensions |
|---------|-----------|---------|------------|
| Deduplicated news articles | `data/processed/deduped_aug/YYYY-MM-DD.parquet` | Article text + 768-dim BERT embedding | >21M articles, 1,697 daily files |
| Energy Price TMPT scores | `data/processed/energy_price_scores/YYYY-MM-DD.parquet` | 10 TMPT scores per article | Same date range |
| Deduplicated article stats | `data/processed/dedup_aug_stats.csv` | Daily total article count (`saved`) | 1,697 rows |
| Country daily mentions | `data/processed/country_daily_mentions.csv` | Daily country mention count, proportion, rel_count per country | ~249 countries × 1,697 days |
| Top 30 countries | `data/processed/top30_countries.csv` | Top 30 countries by grand mean Energy Price mention rate | 30 rows |
| Country rolling z-scores | `data/processed/country_rolling_zscore_W{14,21,28}d.csv` | Forward rolling z-score per country per date (3 windows) | dates × countries |
| Country anomaly consensus | `data/processed/country_anomaly_consensus.csv` | Date-country pairs where z-score > 3 across all 3 windows | Variable rows |
| Country TF-IDF terms | `data/processed/country_tfidf/{country_id}.csv` | Top 100 TF-IDF terms per anomaly window per country | 100 rows × 8 cols per country |
| High-impact events | `data/processed/country_tfidf/event.txt` | 8 human-selected event labels with keyword patterns | 8 events |
| Event news daily counts | `data/processed/event_news_daily.csv` | Daily count of Energy Price articles matching each event | 1,697 rows × 10 cols |
| ETF daily prices | `data/raw/ETF_daily_US_2021-09-01_2026-05-22.xlsx` | Daily closing prices for 5 ETFs | ~1,178 trading days × 5 tickers |

### 3.2 Unit of Observation and Time Coverage

- **Primary unit**: individual news article (for scoring); aggregated to day (for counts and z-scores) and month (for visualization).
- **Coverage**: 2021-09-01 to 2026-05-01.
- **Panel structure**: unbalanced daily panel — article counts vary by day.

### 3.3 Sample Size

| Group | Definition | Approximate N |
|-------|-----------|--------------|
| Full corpus | All deduplicated articles | >21M articles |
| Energy Price articles ("rel") | All 10 TMPT scores > 0.5 | See `dedup_aug_stats.csv` |
| Background articles ("irrel") | Not rel | Remainder |
| ETF trading observations | 5 ETFs × ~1,178 trading days | ~5,890 observations |

### 3.4 Focal Variables

The analysis focuses on:

- **10 Energy Price TMPT keywords**: "Energy Price", "Electricity Price", "Energy Cost", "Power Price", "Fuel Price", "Energy Tariff", "Electricity Tariff", "Energy Market Price", "Power Cost", "Energy Pricing".
- **Top 30 countries**: selected by grand mean Energy Price article mention rate; tracked daily in `country_daily_mentions.csv`.
- **8 high-impact events**: ARE_SAU, ESP, BRA, GEO, DEU, IRN, ISR, UKR — selected from TF-IDF anomaly-window analysis and summarized in `event.txt`.
- **5 ETFs**: ICLN (clean energy), IXC (global energy), VDE (diversified energy), XLE (large-cap energy), XOP (oil & gas exploration).

---

## 4. Variable Construction / Key Variables

### 4.1 Energy Price Relevance Flag

```
energy_price_flag_i = 1  if  min(s_{i,1}, ..., s_{i,10}) > 0.5
                      0  otherwise
```

where `s_{i,k}` is the TMPT similarity score between article `i`'s 768-dim embedding and the pre-computed representation of Energy Price keyword `k` (k = 1…10, defined in §3.4).

**Produced by**: `src/preprocessing/energy_price_scoring.py`

### 4.2 Outcome Variables

| Variable | Definition | Units | Source |
|----------|-----------|-------|--------|
| `energy_price_proportion_t` | N_rel_t / N_total_t | proportion | energy_price_scores + dedup_aug_stats |
| `mean_score_t` | grand mean TMPT score across 10 keywords for rel articles on day t | [0, 1] | energy_price_scores |
| `country_rate_{t,c}` | N_mention_{t,c} / N_rel_t | proportion | country_daily_mentions.csv |
| `zscore_{t,c,W}` | forward rolling z-score of country mention proportion, window W | standardized | country_rolling_zscore_W{W}d.csv |
| `X_E_{j,t}` | sum(N_event_j, [t−13, t]) / sum(N_rel, [t−13, t]) | ratio | event_news_daily.csv |
| `Y_{i,t}` | mean(Close_i, [t−12, t+1]) | USD | ETF_daily |

where `t` = date, `c` = country, `W` ∈ {14, 21, 28}, `j` = event label, `i` = ETF ticker.

### 4.3 Main Explanatory Variables

| Variable | Definition | Units | Source |
|----------|-----------|-------|--------|
| `X_AR_{i,t}` | mean(Close_i, [t−13, t]) | USD | ETF_daily (autoregressive smoother) |
| `X_ED_t` | sum(N_rel, [t−13, t]) / sum(N_total, [t−13, t]) | ratio | dedup_aug_stats + event_news_daily |
| `X_E_{j,t}` | sum(N_event_j, [t−13, t]) / sum(N_rel, [t−13, t]) | ratio | event_news_daily.csv |

All rolling windows are 14 calendar days.

### 4.4 Derived Monthly Variables

Monthly aggregates used for visualization (Q1, Q2, Q3):

```
prop_m       = mean_{t ∈ month m} energy_price_proportion_t
s̄_m          = mean_{t ∈ month m} mean_score_t
z_m          = (s̄_m − mean_m s̄_m) / std_m s̄_m
c̄_{m,c}      = mean_{t ∈ month m} country_rate_{t,c}
```

### 4.5 Event Variables

IRN and ISR are merged into a single event variable IRN_ISR due to co-occurrence during the same geopolitical episode:

```
N_IRN_ISR_t = N_IRN_t + N_ISR_t
X_E_IRN_ISR_t = sum(N_IRN_ISR, [t−13, t]) / sum(N_rel, [t−13, t])
```

Final event variables in regression: ARE_SAU, ESP, BRA, GEO, DEU, IRN_ISR, UKR (7 variables).

**Produced by**: `src/preprocessing/event_news_count.py`, `src/preprocessing/etf_regression.py`

---

## 5. Identification Strategy

### 5.1 Design Principle

The study uses two complementary isolation designs:

**Descriptive isolation (Q1–Q5)**: Each analysis is conducted within the Energy Price rel article pool. Country and event measures are computed as proportions of rel articles, removing between-topic confounding from the broader news corpus. Country anomalies are identified by requiring z-score > 3 across all three rolling windows (14, 21, 28 days), providing consensus-based noise reduction.

**Regression isolation (Q6)**: For each ETF independently, a single OLS model is estimated with all event variables entered simultaneously, conditional on autoregressive ETF price levels and aggregate energy news intensity. This controls for baseline price momentum and secular changes in energy news volume, isolating the marginal association of each event's news intensity with ETF price levels.

### 5.2 Model Specification

**ETF Regression Model**

For each ETF `i` (ICLN, IXC, VDE, XLE, XOP):

```
Y_{i,t} = α_i
         + Σ_m β_{im} · X_AR_{m,t}         (m = ICLN, IXC, VDE, XLE, XOP)
         + γ_i · X_ED_t
         + Σ_j δ_{ij} · X_E_{j,t}          (j = ARE_SAU, ESP, BRA, GEO, DEU, IRN_ISR, UKR)
         + ε_{i,t}
```

Estimated by OLS. N = 1,178 trading days.

**Primary estimands**: `δ_{ij}` — the association between a 1-unit increase in event `j`'s 14-day news intensity ratio and the 14-day forward average price of ETF `i`, conditional on autoregressive price levels and aggregate energy news density.

### 5.3 Interpretation Limits

**This study does not claim causal effects.** The analysis is purely associational:

1. The Energy Price article pool is defined by a model-based threshold (all 10 TMPT scores > 0.5). Sensitivity to this threshold is not formally tested.
2. Event news intensity (`X_E_j`) is constructed from article counts matching keyword patterns. It captures co-occurrence of geopolitical terms with Energy Price discourse, not the magnitude or severity of the underlying event.
3. The regression model does not instrument for event intensity; reverse causality (ETF price movements generating news) cannot be ruled out.
4. Autoregressive controls absorb price momentum but do not eliminate omitted variable bias from simultaneous macro shocks.

---

## 6. Main Estimation Framework

### 6.1 Primary Analysis

The primary analysis consists of six analyses organized into three tracks:

| Track | Sub-questions | Measure | Granularity |
|-------|---------------|---------|-------------|
| Coverage dynamics | Q1, Q2 | Proportion + TMPT z-score | Monthly |
| Geographic and event structure | Q3, Q4, Q5 | Country mention rate + z-score heatmap + TF-IDF events | Aggregate + Daily |
| Price impact | Q6 | OLS regression coefficients | 14-day rolling (trading days) |

**Regression grid**: 5 ETFs × 1 model specification = 5 estimated equations.

Each model answers: *After controlling for autoregressive price levels and aggregate energy news density, which international event's news intensity is associated with forward ETF price levels, and in which direction?*

### 6.2 Heterogeneity Analysis

| Analysis | Figure/Table | Grouping variable |
|----------|-------------|------------------|
| Country prominence across geography | fig_country_map | Country identity |
| Country anomaly timing | fig_country_zscore_heatmap | Country × Date |
| Event impact by ETF type | table_regression | ETF identity (clean vs. fossil vs. exploration) |

### 6.3 Robustness Checks

| # | Check | What it perturbs | Why it matters |
|---|-------|-----------------|----------------|
| 1 | IRN and ISR separated vs. merged | Event variable definition | Confirms IRN_ISR merger does not mask opposing effects |
| 2 | Window size sensitivity (W=14 vs. W=21 vs. W=28) | Rolling window in z-score anomaly detection | Confirms country anomaly consensus is not window-specific |

---

## 7. Analytical Workflow

### Step 1 — Data Preparation

- Run `src/preprocessing/dedup.py`: deduplicate raw daily parquets using ±7-day sliding window.
- Output: `data/processed/deduped_aug/YYYY-MM-DD.parquet`, `data/processed/dedup_aug_stats.csv`
- Run `src/preprocessing/energy_price_scoring.py` (conda tf): score all deduplicated articles against 10 Energy Price keywords using TMPTv2; threshold 0.5.
- Output: `data/processed/energy_price_scores/YYYY-MM-DD.parquet`

### Step 2 — Country Mention Counting

- Run `src/preprocessing/country_daily_mentions.py`: count Energy Price rel article mentions per country per day using alias-based regex from `data/raw/country_dict_0.json`.
- Output: `data/processed/country_daily_mentions.csv`
- Run `src/preprocessing/top30_countries.py`: select top 30 countries by grand mean mention rate.
- Output: `data/processed/top30_countries.csv`

### Step 3 — Country Anomaly Detection

- Run `src/preprocessing/country_rolling_zscore.py`: compute forward rolling proportions and z-scores for windows W = 14, 21, 28 days.
- Output: `data/processed/country_rolling_zscore_W{14,21,28}d.csv`
- Run `src/preprocessing/country_anomaly_consensus.py`: find date-country pairs with z-score > 3 in all three windows.
- Output: `data/processed/country_anomaly_consensus.csv`

### Step 4 — Event Identification (TF-IDF)

- Run `src/preprocessing/country_article_index.py`: build per-country article index for anomaly windows.
- Output: `data/processed/country_articles/{country_id}/{date}.parquet`
- Run `src/preprocessing/country_event_tfidf.py`: compute TF-IDF distinctiveness for anomaly vs. control windows per country.
- Output: `data/processed/country_tfidf/{country_id}.csv`
- **Human step**: review TF-IDF top terms per country, identify high-impact events, write `data/processed/country_tfidf/event.txt`.

### Step 5 — Event News Counting

- Run `src/preprocessing/event_news_count.py`: count daily Energy Price rel articles matching each event's keyword patterns.
- Output: `data/processed/event_news_daily.csv`

### Step 6 — ETF Regression

- Run `src/preprocessing/etf_regression.py`: build 14-day rolling features and run OLS for each of 5 ETFs.
- Output: `data/results/etf_regression_results.csv`

### Step 7 — Figures and Tables

- `data/analysis_workflow.svg` → `export/figures/fig_analysis_workflow.png` workflow chart
- `src/analyses/fig_prop_monthly.py` → fig_prop_monthly, table_yearly_stats
- `src/analyses/fig_score_monthly.py` → fig_score_monthly
- `src/analyses/fig_country_map.py` → fig_country_map
- `src/analyses/fig_country_zscore_heatmap.py` → fig_country_zscore_heatmap
- `src/analyses/table_regression.py` → table_regression

---

## 8. Expected Results

Result 1. Energy Price news coverage has grown as a share of total news between 2021 and 2026, with acceleration following the 2022 Russia-Ukraine conflict.
  - fig_prop_monthly: Monthly line chart showing the proportion (%) of Energy Price articles in total news; x-axis = month (2021-09 to 2026-05), y-axis = proportion (%). Expected upward trend with a visible step-change in early 2022.
  - table_yearly_stats: Annual summary table with columns: Year, N_rel, N_total, Energy Price Share (%), YoY Growth (%). Final row shows full-period totals. Expected monotonic increase in N_rel and share.

Result 2. Energy Price articles have become semantically more focused over time, suggesting increasingly concrete coverage of energy pricing topics.
  - fig_score_monthly: Monthly Z-score of the grand mean TMPT relevance score for Energy Price rel articles; x-axis = month, y-axis = z-score. Expected positive trend from 2022 onward.

Result 3. In Energy Price news, the United States is most prominently co-mentioned with a concentrated set of countries, with European, Middle Eastern, and East Asian countries showing the highest association rates.
  - fig_country_map: World map with the United States shown in gray; other countries colored on a white-to-red scale proportional to their grand mean mention rate in Energy Price rel articles. Expected red concentration in Western Europe (DEU, GBR, FRA), Middle East (SAU, IRN), and East Asia (CHN, JPN).

Result 4. Several of the top 30 countries exhibit statistically anomalous news intensity during identifiable geopolitical episodes, with anomalies concentrated in 2022–2024.
  - fig_country_zscore_heatmap: Heatmap with countries on the y-axis and dates on the x-axis; cell color represents the consensus z-score (white = normal, dark red = high anomaly). Expected clusters of high z-scores in 2022 (UKR, RUS, DEU) and 2023–2024 (ISR, IRN, ARE, SAU).

Result 5. TF-IDF analysis of anomaly windows identifies 8 high-impact international events associated with Energy Price news spikes.
  - table_events: Event summary table listing event label, associated countries, representative keywords from TF-IDF, and anomaly window dates. Entries: ARE_SAU, ESP, BRA, GEO, DEU, IRN, ISR, UKR.

Result 6. High-impact international events are significantly associated with Energy Price ETF price levels, with clean energy ETFs (ICLN) and exploration ETFs (XOP) showing the broadest and largest event responses respectively.
  - table_regression: OLS results table with columns: ETF, Variable, Coefficient, Std Error, t-stat, p-value, Significance (stars), R², Adj. R², N. Rows: one per (ETF, variable) pair. Expected: UKR significant and positive across all 5 ETFs; DEU positive for ICLN; BRA negative for ICLN; IRN_ISR positive for XOP.

---

## 9. Relationship with Jiazi

The analysis repository provides artifacts to Jiazi through the `export/` interface.

Exported artifacts include:
- Figures (PNG/SVG)
- Workflow chart (`fig_analysis_workflow.png`)
- Regression / results tables (CSV/LaTeX)
- Reproducible scripts
- actionbrief.yaml

Jiazi uses these artifacts to generate manuscript sections and assemble the research paper.

---

## 10. Human Debug Notes

- **2026-06-07**: Initial AnaSOP written. Research type confirmed as applied (observational associational study). Six sub-questions covering: (1) Energy Price news coverage growth, (2) semantic concreteness trend via TMPT z-score, (3) country co-mention geography, (4) top-30 country anomaly detection, (5) TF-IDF event identification, (6) ETF regression. Pipeline: deduplication → TMPT scoring (threshold 0.5, 10 Energy Price keywords) → country mention counting → rolling z-score anomaly detection (W=14,21,28, consensus threshold z>3) → per-country TF-IDF on anomaly windows → human event selection (event.txt, 8 events) → event news counting → OLS regression (5 ETFs, 14-day rolling windows). IRN and ISR merged into IRN_ISR in regression. UKR added as 8th event. Full study period 2021-09-01 to 2026-05-01. ETF data available to 2026-05-22.
- **2026-06-08**: Added `fig_analysis_workflow.png` as a workflow support figure in `export/figures/`, with SVG source stored at `data/analysis_workflow.svg`. The figure summarizes the analysis sequence from data preparation through geographic event discovery, event counting, ETF regression, and final exported artifacts.
