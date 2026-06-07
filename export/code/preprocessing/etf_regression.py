"""
ETF Regression Analysis

Variables (all windows are 14 calendar days):
  Y_i(t)     = mean(Close_i, [t-12, t+1])            forward-shifted 14-day average
  X_AR_i(t)  = mean(Close_i, [t-13, t])              autoregressive smoothed price
  X_ED(t)    = sum(N_energy, [t-13,t]) / sum(N_total, [t-13,t])
  X_E_j(t)   = sum(N_event_j, [t-13,t]) / sum(N_energy, [t-13,t])

Model (per ETF):
  Y_i(t) = a + sum_m b_im * X_AR_m(t) + c_i * X_ED(t) + sum_j d_ij * X_E_j(t) + e

Input:  data/raw/ETF_daily_US_2021-09-01_2026-05-22.xlsx
        data/processed/event_news_daily.csv
        data/processed/dedup_aug_stats.csv

Output: data/results/etf_regression_results.csv
"""

import os
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

ETF_FILE    = PROJECT_ROOT / "data/raw/ETF_daily_US_2021-09-01_2026-05-22.xlsx"
EVENT_FILE  = PROJECT_ROOT / "data/processed/event_news_daily.csv"
DEDUP_FILE  = PROJECT_ROOT / "data/processed/dedup_aug_stats.csv"
OUT_PATH    = PROJECT_ROOT / "data/results/etf_regression_results.csv"

TICKERS       = ['ICLN', 'IXC', 'VDE', 'XLE', 'XOP']
MERGED_EVENTS = ['ARE_SAU', 'ESP', 'BRA', 'GEO', 'DEU', 'IRN_ISR', 'UKR']  # IRN + ISR combined
WINDOW        = 14


def load_data():
    etf_df   = pd.read_excel(ETF_FILE, parse_dates=['date'])
    event_df = pd.read_csv(EVENT_FILE, parse_dates=['date'])
    dedup_df = pd.read_csv(DEDUP_FILE, parse_dates=['date'])
    return etf_df, event_df, dedup_df


def build_features(etf_df, event_df, dedup_df):
    etf_pivot = etf_df.pivot(index='date', columns='ticker', values='close')[TICKERS]
    trading_dates = etf_pivot.index

    all_dates = pd.date_range(trading_dates.min(), trading_dates.max() + pd.Timedelta(days=1))
    etf_daily = etf_pivot.reindex(all_dates).ffill()

    X_AR = etf_daily.rolling(WINDOW).mean()
    X_AR.columns = [f'X_AR_{c}' for c in TICKERS]

    Y = etf_daily.rolling(WINDOW).mean().shift(-1)
    Y.columns = [f'Y_{c}' for c in TICKERS]

    event_idx = event_df.set_index('date').reindex(all_dates).fillna(0)
    dedup_idx = (
        dedup_df.set_index('date')[['saved']]
        .rename(columns={'saved': 'total_news'})
        .reindex(all_dates)
        .fillna(0)
    )
    news = event_idx.join(dedup_idx)
    news_roll = news.rolling(WINDOW).sum()

    X_ED = (news_roll['total_relevant_articles'] / news_roll['total_news']).rename('X_ED')

    news_roll['IRN_ISR'] = news_roll['IRN'] + news_roll['ISR']
    X_events = pd.DataFrame(
        {f'X_E_{ev}': news_roll[ev] / news_roll['total_relevant_articles'] for ev in MERGED_EVENTS}
    )

    features = pd.concat([Y, X_AR, X_ED, X_events], axis=1)
    features = features[features.index.isin(trading_dates)].dropna()
    return features


def run_regressions(features):
    x_cols = [f'X_AR_{t}' for t in TICKERS] + ['X_ED'] + [f'X_E_{ev}' for ev in MERGED_EVENTS]
    X = sm.add_constant(features[x_cols])
    return {ticker: sm.OLS(features[f'Y_{ticker}'], X).fit() for ticker in TICKERS}


def _sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.1:   return '.'
    return ''


def print_results(models):
    for ticker, m in models.items():
        print(f"\n{'='*72}")
        print(f"ETF: {ticker}  |  R² = {m.rsquared:.4f}  |  "
              f"Adj. R² = {m.rsquared_adj:.4f}  |  N = {int(m.nobs)}")
        print(f"{'='*72}")
        print(f"{'Variable':<25} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10} {'':>5}")
        print('-' * 72)
        for var in m.params.index:
            print(
                f"{var:<25} {m.params[var]:>12.6f} {m.bse[var]:>12.6f} "
                f"{m.tvalues[var]:>10.4f} {m.pvalues[var]:>10.4f} {_sig(m.pvalues[var]):>5}"
            )
    print("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")


def save_results(models):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker, m in models.items():
        for var in m.params.index:
            rows.append({
                'ETF':         ticker,
                'Variable':    var,
                'Coefficient': round(m.params[var],  6),
                'Std_Error':   round(m.bse[var],     6),
                't_stat':      round(m.tvalues[var], 4),
                'p_value':     round(m.pvalues[var], 4),
                'Sig':         _sig(m.pvalues[var]),
                'R2':          round(m.rsquared,     4),
                'Adj_R2':      round(m.rsquared_adj, 4),
                'N':           int(m.nobs),
            })
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


def main():
    print("Loading data...")
    etf_df, event_df, dedup_df = load_data()

    print("Building features...")
    features = build_features(etf_df, event_df, dedup_df)
    print(f"Feature matrix: {features.shape[0]} observations × {features.shape[1]} variables")
    print(f"Date range: {features.index.min().date()} to {features.index.max().date()}")

    print("\nRunning OLS regressions...")
    models = run_regressions(features)

    print_results(models)
    save_results(models)


if __name__ == "__main__":
    main()
