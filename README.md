# ESG07d — Energy Price News and ETF Impact Analysis

An observational study of U.S. English-language news coverage of Energy Price topics (2021–2026) and its association with energy ETF price movements.

---

## Research Overview

Six sub-questions addressed in sequence:

1. Has the volume of Energy Price news grown relative to total news (2021–2026)?
2. Have Energy Price articles become semantically more focused over time (TMPT score trend)?
3. Which countries are most prominently co-mentioned with Energy Price topics?
4. Which top-30 countries exhibit anomalous coverage intensity, and when?
5. What high-impact international events drove the anomalies (TF-IDF)?
6. What is the measured association between those events and energy ETF prices (OLS)?

**Study type**: Observational, associational. No causal claims.
**Corpus**: >21M deduplicated U.S. English news articles across 1,697 daily files.
**Period**: 2021-09-01 to 2026-05-01.
**ETFs**: ICLN, IXC, VDE, XLE, XOP.

See `docs/AnaSOP.md` for the full research design, variable definitions, and identification strategy.

---

## Structure

```
ESG07d/
├── src/
│   ├── preprocessing/     # Dedup, TMPT scoring, country counts, z-scores, TF-IDF, regression
│   └── analyses/          # Figure and table generation scripts
├── data/
│   ├── raw/               # ETF prices, country alias dictionary (DVC-tracked)
│   ├── processed/         # Intermediate datasets (DVC-tracked)
│   └── results/           # Figures and tables (DVC-tracked)
├── export/                # Final artifacts for Jiazi handoff
│   ├── figures/           # fig_*.png
│   ├── tables/            # table_*.xlsx
│   ├── code/              # Reproducible analysis scripts
│   ├── AnaSOP.md
│   └── actionbrief.yaml
├── docs/
│   ├── AnaSOP.md          # Research design and workflow
│   ├── COMMANDS.md        # Detailed command reference
│   ├── TEAM_RULES.md      # Collaboration conventions
│   └── INTELLECTUAL_PROPERTY.md
├── model/                 # SAPTv2_XgbRegressor.joblib
├── build_export.py        # Assembles export/ for Jiazi handoff
├── pyproject.toml
└── uv.lock
```

---

## Analysis Pipeline

| Step | Script(s) | Output |
|------|-----------|--------|
| 1. Deduplication | `src/preprocessing/dedup.py` | `data/processed/deduped_aug/`, `dedup_aug_stats.csv` |
| 2. TMPT scoring | `src/preprocessing/energy_price_scoring.py` (conda tf) | `data/processed/energy_price_scores/` |
| 3. Country mention counting | `src/preprocessing/country_daily_mentions.py` | `data/processed/country_daily_mentions.csv` |
| 4. Top-30 selection | `src/preprocessing/top30_countries.py` | `data/processed/top30_countries.csv` |
| 5. Rolling z-scores | `src/preprocessing/country_rolling_zscore.py` | `data/processed/country_rolling_zscore_W{14,21,28}d.csv` |
| 6. Anomaly consensus | `src/preprocessing/country_anomaly_consensus.py` | `data/processed/country_anomaly_consensus.csv` |
| 7. Article index | `src/preprocessing/country_article_index.py` | `data/processed/country_articles/` |
| 8. TF-IDF | `src/preprocessing/country_event_tfidf.py` | `data/processed/country_tfidf/` |
| 9. **Human step** | Review TF-IDF top terms → write `event.txt` | 8 events identified |
| 10. Event news counts | `src/preprocessing/event_news_count.py` | `data/processed/event_news_daily.csv` |
| 11. ETF regression | `src/preprocessing/etf_regression.py` | `data/results/etf_regression_results.csv` |
| 12. Figures & tables | `src/analyses/fig_*.py`, `src/analyses/table_*.py` | `data/results/` |

---

## Setup

```bash
# 1. Clone
git clone <repo-url> ESG07d && cd ESG07d

# 2. Create .env
echo "myproj=ESG07d"        > .env
echo "PROJECT_ROOT=$(pwd)"  >> .env

# 3. Install dependencies
uv sync
pre-commit install

# 4. Pull data
dvc pull
```

---

## Daily Workflow

```bash
source .venv/bin/activate
set -a && source .env && set +a
git pull && dvc pull
```

---

## Export for Jiazi

Run `build_export.py` to assemble the `export/` directory:

```bash
python build_export.py
```

This copies figures, converts tables to `.xlsx`, packages `src/` scripts, writes `export/code/run_analysis.py`, and copies `docs/AnaSOP.md`.

---

See `docs/TEAM_RULES.md` for conventions and `docs/COMMANDS.md` for the full command reference.
