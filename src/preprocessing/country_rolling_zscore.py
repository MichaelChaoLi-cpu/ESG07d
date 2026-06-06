"""
Forward rolling country mention proportion and z-score for multiple window sizes.

For each date t and window W:
  proportion[t, c] = sum(count_c, t..t+W-1) / sum(rel_count, t..t+W-1)
  zscore[t, c]     = (proportion - mean) / std  across all dates

Outputs (one pair per window):
  data/processed/country_rolling_proportion_W{W}d.csv  (dates × countries)
  data/processed/country_rolling_zscore_W{W}d.csv      (dates × countries)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MENTIONS_PATH = ROOT / "data/processed/country_daily_mentions.csv"
TOP30_PATH = ROOT / "data/processed/top30_countries.csv"
OUT_DIR = ROOT / "data/processed"

WINDOWS = [14, 21, 28]


def compute_for_window(count_wide, roll_rel_cache, id_to_name, window):
    # Forward rolling: sum(t .. t+W-1) via reversing the series
    roll_count = count_wide.iloc[::-1].rolling(window=window, min_periods=window).sum().iloc[::-1]
    roll_rel = roll_rel_cache.iloc[::-1].rolling(window=window, min_periods=window).sum().iloc[::-1]

    proportion = roll_count.div(roll_rel, axis=0).dropna(how="all")
    zscore = (proportion - proportion.mean()) / proportion.std()

    proportion.columns = [id_to_name.get(c, c) for c in proportion.columns]
    zscore.columns = [id_to_name.get(c, c) for c in zscore.columns]
    proportion.index.name = "date"
    zscore.index.name = "date"

    return proportion, zscore


def main():
    top30 = pd.read_csv(TOP30_PATH)
    top30_ids = set(top30["country_id"])
    id_to_name = top30.set_index("country_id")["country_name"].to_dict()

    df = pd.read_csv(MENTIONS_PATH, parse_dates=["date"])
    df = df[df["country_id"].isin(top30_ids)].copy()

    count_wide = df.pivot_table(
        index="date", columns="country_id", values="count", aggfunc="sum", fill_value=0
    )
    rel_wide = df.pivot_table(
        index="date", columns="country_id", values="rel_count", aggfunc="max", fill_value=0
    )

    all_dates = count_wide.index.union(rel_wide.index).sort_values()
    count_wide = count_wide.reindex(all_dates, fill_value=0)
    rel_wide = rel_wide.reindex(all_dates, fill_value=0)
    rel_series = rel_wide.iloc[:, 0]

    for w in WINDOWS:
        proportion, zscore = compute_for_window(count_wide, rel_series, id_to_name, w)

        prop_path = OUT_DIR / f"country_rolling_proportion_W{w}d.csv"
        zscore_path = OUT_DIR / f"country_rolling_zscore_W{w}d.csv"
        proportion = proportion[proportion.index <= "2026-04-15"]
        zscore = zscore[zscore.index <= "2026-04-15"]
        proportion.to_csv(prop_path)
        zscore.to_csv(zscore_path)

        print(f"W={w:2d}d  shape={proportion.shape}  "
              f"date={proportion.index[0].date()} ~ {proportion.index[-1].date()}")
        print(f"       proportion → {prop_path.name}")
        print(f"       zscore     → {zscore_path.name}")


if __name__ == "__main__":
    main()
