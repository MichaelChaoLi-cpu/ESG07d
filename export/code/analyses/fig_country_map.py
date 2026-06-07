"""
Q3 — Geographic distribution of country co-mentions in Energy Price news.

World map: USA in gray, other countries colored white-to-red by grand mean
mention rate in Energy Price rel articles.

Input:  data/processed/country_daily_mentions.csv
Output: data/results/fig_country_map.png
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
from dotenv import load_dotenv
from matplotlib.patches import Patch
from labels import SERIES_LABELS

matplotlib.use("Agg")

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

COUNTRY_CSV = PROJECT_ROOT / "data/processed/country_daily_mentions.csv"
OUT_PATH    = PROJECT_ROOT / "data/results/fig_country_map.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

NE_PATH = Path(pyogrio.__file__).parent / "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"

USA_GRAY = "#BDBDBD"
GRID_COLOR, GRID_LW, GRID_LS = "#AAAAAA", 0.4, "--"
LONS = list(range(-150, 181, 30))
LATS = list(range(-60, 91, 30))


def main():
    cd  = pd.read_csv(COUNTRY_CSV, parse_dates=["date"])
    agg = cd.groupby("country_id")[["count", "rel_count"]].sum()
    agg["mean_rate"] = agg.apply(
        lambda r: r["count"] / r["rel_count"] if r["rel_count"] > 0 else 0.0, axis=1
    )
    grand_mean = agg[["mean_rate"]].rename_axis("iso_a3").reset_index()

    world = gpd.read_file(NE_PATH)
    world = world.merge(grand_mean, on="iso_a3", how="left")
    world["color_val"] = world["mean_rate"].fillna(0)

    vmax = world.loc[world["iso_a3"] != "USA", "color_val"].quantile(0.98)
    cmap = plt.cm.Reds

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_facecolor("#E3F2FD")

    world[world["iso_a3"] == "USA"].plot(ax=ax, color=USA_GRAY, edgecolor="white", linewidth=0.3)

    non_usa      = world[world["iso_a3"] != "USA"].copy()
    non_usa_zero = non_usa[non_usa["color_val"] == 0]
    non_usa_pos  = non_usa[non_usa["color_val"] >  0]

    non_usa_zero.plot(ax=ax, color="#F5F5F5", edgecolor="white", linewidth=0.3)
    if not non_usa_pos.empty:
        norm = mcolors.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1)
        non_usa_pos.plot(ax=ax, column="color_val", cmap=cmap, norm=norm,
                         edgecolor="white", linewidth=0.3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, shrink=0.6)
        cbar.set_label(SERIES_LABELS["mean_rate"], fontsize=10)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)

    for lon in LONS:
        ax.axvline(lon, color=GRID_COLOR, linewidth=GRID_LW, linestyle=GRID_LS, zorder=1)
    for lat in LATS:
        ax.axhline(lat, color=GRID_COLOR, linewidth=GRID_LW, linestyle=GRID_LS, zorder=1)

    ax.set_xticks(LONS)
    ax.set_xticklabels(
        [f"{abs(l)}°{'W' if l < 0 else ('E' if l > 0 else '')}" for l in LONS],
        fontsize=6.5, color="#444444",
    )
    ax.set_yticks(LATS)
    ax.set_yticklabels(
        [f"{abs(l)}°{'S' if l < 0 else ('N' if l > 0 else '0°')}" for l in LATS],
        fontsize=6.5, color="#444444",
    )
    ax.tick_params(axis="both", which="both", length=3, width=0.6,
                   direction="out", color="#444444")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#444444")

    legend_elements = [
        Patch(facecolor=USA_GRAY, edgecolor="white", label="United States (reference)"),
        Patch(facecolor="#F5F5F5", edgecolor="#CCCCCC", label="No mentions"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
