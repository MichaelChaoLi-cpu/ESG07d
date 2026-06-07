"""
Q1 — Energy Price news coverage growth.

Produces:
  data/results/fig_prop_monthly.png   Monthly line chart of Energy Price article proportion
  data/results/table_yearly_stats.csv Annual summary table (last row = Total)

Input:  data/processed/dedup_aug_stats.csv       (saved = N_total per day)
        data/processed/event_news_daily.csv       (total_relevant_articles = N_rel per day)
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

STATS_CSV = PROJECT_ROOT / "data/processed/dedup_aug_stats.csv"
EVENT_CSV = PROJECT_ROOT / "data/processed/event_news_daily.csv"
OUT_DIR   = PROJECT_ROOT / "data/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    stats = pd.read_csv(STATS_CSV, parse_dates=["date"]).set_index("date")[["saved"]]
    stats.columns = ["N_total"]

    event = pd.read_csv(EVENT_CSV, parse_dates=["date"]).set_index("date")[["total_relevant_articles"]]
    event.columns = ["N_rel"]

    df = stats.join(event, how="left").fillna(0)
    df["N_irrel"] = df["N_total"] - df["N_rel"]

    # ── Monthly proportion ──────────────────────────────────────────────────
    monthly_rel   = df["N_rel"].resample("ME").sum()
    monthly_total = df["N_total"].resample("ME").sum()
    monthly_prop  = (monthly_rel / monthly_total * 100).where(monthly_total > 0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly_prop.index, monthly_prop.values, color="#E53935", linewidth=1.8, zorder=3)
    ax.fill_between(monthly_prop.index, monthly_prop.values, alpha=0.18, color="#E53935")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel(SERIES_LABELS["energy_price_proportion"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(monthly_prop.index.min(), monthly_prop.index.max())
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out_fig = OUT_DIR / "fig_prop_monthly.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_fig}")
    plt.close()

    # ── Yearly stats table ──────────────────────────────────────────────────
    yearly = df.resample("YE").sum()
    yearly.index = yearly.index.year
    yearly.index.name = "Year"
    yearly["EP Share (%)"] = (yearly["N_rel"] / yearly["N_total"] * 100).round(4)
    yearly = yearly.rename(columns={
        "N_rel":   SERIES_LABELS["N_rel"],
        "N_irrel": SERIES_LABELS["N_irrel"],
        "N_total": SERIES_LABELS["N_total"],
    })[[SERIES_LABELS["N_rel"], SERIES_LABELS["N_irrel"], SERIES_LABELS["N_total"], "EP Share (%)"]]

    total = yearly[[SERIES_LABELS["N_rel"], SERIES_LABELS["N_irrel"], SERIES_LABELS["N_total"]]].sum()
    total["EP Share (%)"] = round(total[SERIES_LABELS["N_rel"]] / total[SERIES_LABELS["N_total"]] * 100, 4)
    total.name = "Total"

    result = pd.concat([yearly, total.to_frame().T])
    out_tbl = OUT_DIR / "table_yearly_stats.csv"
    result.to_csv(out_tbl)
    print(f"Saved → {out_tbl}")
    print(result.to_string())


if __name__ == "__main__":
    main()
