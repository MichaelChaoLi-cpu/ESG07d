"""
Find date-country pairs where z-score > 3 across all three windows (14, 21, 28).

Output:
  data/processed/country_anomaly_consensus.csv
  columns: date, country, zscore_W14, zscore_W21, zscore_W28
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/country_anomaly_consensus.csv"
THRESHOLD = 3.0
WINDOWS = [14, 21, 28]


def main():
    zscores = {
        w: pd.read_csv(
            ROOT / f"data/processed/country_rolling_zscore_W{w}d.csv",
            index_col="date",
        )
        for w in WINDOWS
    }

    # Align to common dates and countries
    common_dates = zscores[14].index
    for w in WINDOWS[1:]:
        common_dates = common_dates.intersection(zscores[w].index)

    common_cols = zscores[14].columns
    for w in WINDOWS[1:]:
        common_cols = common_cols.intersection(zscores[w].columns)

    z = {w: zscores[w].loc[common_dates, common_cols] for w in WINDOWS}

    # Mask: all three windows exceed threshold
    consensus = (z[14] > THRESHOLD) & (z[21] > THRESHOLD) & (z[28] > THRESHOLD)

    # Convert to long format
    rows = []
    for date, country in zip(*consensus.values.nonzero()):
        d = common_dates[date]
        c = common_cols[country]
        rows.append({
            "date": d,
            "country": c,
            "zscore_W14": round(z[14].loc[d, c], 4),
            "zscore_W21": round(z[21].loc[d, c], 4),
            "zscore_W28": round(z[28].loc[d, c], 4),
        })

    result = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    result.to_csv(OUT, index=False)

    print(f"Consensus anomalies: {len(result)} records, {result['country'].nunique()} countries")
    print(f"Saved → {OUT.name}")
    print()
    print(result.groupby("country").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
