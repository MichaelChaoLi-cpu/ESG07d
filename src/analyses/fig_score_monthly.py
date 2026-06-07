"""
Q2 — Semantic concreteness of Energy Price coverage over time.

Monthly Z-score of the grand mean TMPT relevance score for Energy Price
rel articles. Positive Z = more semantically focused than average month.

Input:  data/processed/energy_price_score_daily.csv
Output: data/results/fig_score_monthly.png
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from labels import SERIES_LABELS

matplotlib.use("Agg")

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

SCORE_CSV = PROJECT_ROOT / "data/processed/energy_price_score_daily.csv"
OUT_PATH  = PROJECT_ROOT / "data/results/fig_score_monthly.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(SCORE_CSV, parse_dates=["date"]).set_index("date")[["grand_mean"]]
    monthly = df["grand_mean"].resample("ME").mean()
    zscore  = (monthly - monthly.mean()) / monthly.std()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(zscore.index, zscore.values, color="#7B1FA2", linewidth=1.8, zorder=3)
    ax.fill_between(zscore.index, zscore.values, 0,
                    where=zscore.values >= 0, alpha=0.18, color="#7B1FA2")
    ax.fill_between(zscore.index, zscore.values, 0,
                    where=zscore.values < 0,  alpha=0.18, color="grey")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel(f"Z-score ({SERIES_LABELS['grand_mean']})", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(zscore.index.min(), zscore.index.max())
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
