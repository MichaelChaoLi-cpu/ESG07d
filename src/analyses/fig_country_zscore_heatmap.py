"""
Q4 — Top 30 country anomaly heatmap.

Heatmap: date (y-axis, daily) × country (x-axis), color = mean z-score
across W14, W21, W28 windows. White = 0, red = high positive, purple = anomaly (z > 3).

Input:  data/processed/country_rolling_zscore_W{14,21,28}d.csv  (wide: date × country name)
        data/processed/top30_countries.csv                       (rank, country_name)
Output: data/results/fig_country_zscore_heatmap.png
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from labels import SERIES_LABELS

matplotlib.use("Agg")

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

ZSCORE_CSVS = [PROJECT_ROOT / f"data/processed/country_rolling_zscore_W{w}d.csv"
               for w in [14, 21, 28]]
TOP30_CSV   = PROJECT_ROOT / "data/processed/top30_countries.csv"
OUT_PATH   = PROJECT_ROOT / "data/results/fig_country_zscore_heatmap.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    top30 = pd.read_csv(TOP30_CSV).sort_values("rank")
    country_names = top30["country_name"].tolist()

    # Load all three windows and take the mean
    frames = [pd.read_csv(f, parse_dates=["date"]).set_index("date")
              for f in ZSCORE_CSVS]
    common_idx = frames[0].index
    for f in frames[1:]:
        common_idx = common_idx.intersection(f.index)
    zdf = sum(f.loc[common_idx] for f in frames) / len(frames)

    # Keep only top 30 countries that exist as columns
    available = [c for c in country_names if c in zdf.columns]
    zdf = zdf[available]

    # Shape: dates (rows) × countries (cols)  — portrait orientation
    data = zdf[available].clip(lower=-4, upper=6)

    # Custom diverging colormap: blue→green→white(z=0)→yellow→red; purple for z>3
    # Symmetric range -3 to 3, white at center (0.5)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bwr_custom",
        [(0.0, "blue"), (0.25, "green"), (0.5, "white"), (0.75, "yellow"), (1.0, "red")],
    )
    cmap.set_over("purple")

    fig, ax = plt.subplots(figsize=(12, 22))
    im = ax.imshow(
        data.values,
        aspect="auto",
        cmap=cmap,
        vmin=-3, vmax=3,
        interpolation="nearest",
    )

    # X-axis: country names
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, fontsize=7.5, rotation=45, ha="right")

    # Y-axis: one tick per month at the first available date of that month
    dates = zdf.index
    month_positions, month_labels = [], []
    seen = set()
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            month_positions.append(i)
            month_labels.append(d.strftime("%Y-%m"))
    ax.set_yticks(month_positions)
    ax.set_yticklabels(month_labels, fontsize=7)
    ax.set_ylabel("Date (daily)", fontsize=11)

    # Colorbar: extend='max' shows the orange anomaly band above z=3
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, extend="max")
    cbar.set_label(f"Mean {SERIES_LABELS['zscore']} W14/W21/W28 (daily)", fontsize=10)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.text(1.05, 1.0, "z > 3\n(anomaly)", va="bottom", fontsize=7.5,
                 color="purple", transform=cbar.ax.transAxes)


    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
