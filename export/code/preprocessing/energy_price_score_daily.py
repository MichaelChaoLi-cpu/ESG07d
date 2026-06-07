"""
Aggregate per-article TMPT Energy Price scores to daily means.

For each day, load energy_price_scores parquet, filter to rel articles
(min of all 10 score_* columns > 0.5), then compute the daily mean of
each score column and the grand mean across all 10.

Input:  data/processed/energy_price_scores/YYYY-MM-DD.parquet
Output: data/processed/energy_price_score_daily.csv
  Columns: date, mean_score_energy_price, ..., mean_score_energy_pricing,
           grand_mean, N_rel
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

SCORES_DIR = PROJECT_ROOT / "data/processed/energy_price_scores"
OUT_PATH   = PROJECT_ROOT / "data/processed/energy_price_score_daily.csv"

from config import ENERGY_PRICE_THRESHOLD

SCORE_COLS = [
    "score_energy_price", "score_electricity_price", "score_energy_cost",
    "score_power_price", "score_fuel_price", "score_energy_tariff",
    "score_electricity_tariff", "score_energy_market_price",
    "score_power_cost", "score_energy_pricing",
]


def main():
    files = sorted(SCORES_DIR.glob("*.parquet"))
    print(f"Processing {len(files)} days …")

    rows = []
    for f in tqdm(files, unit="day", dynamic_ncols=True):
        date = f.stem
        df = pd.read_parquet(f, columns=SCORE_COLS)
        rel = df[(df[SCORE_COLS] > ENERGY_PRICE_THRESHOLD).all(axis=1)]
        n_rel = len(rel)
        if n_rel == 0:
            row = {"date": date, "grand_mean": float("nan"), "N_rel": 0}
            for c in SCORE_COLS:
                row[f"mean_{c}"] = float("nan")
        else:
            row = {"date": date, "N_rel": n_rel}
            for c in SCORE_COLS:
                row[f"mean_{c}"] = rel[c].mean()
            row["grand_mean"] = rel[SCORE_COLS].values.mean()
        rows.append(row)

    out_cols = ["date"] + [f"mean_{c}" for c in SCORE_COLS] + ["grand_mean", "N_rel"]
    out = pd.DataFrame(rows, columns=out_cols).sort_values("date").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
