"""
Q6 — ETF regression results table (stargazer-style).

One column per ETF; rows alternate coefficient (with stars) and
standard error (in parentheses); model stats at the bottom.

Input:  data/results/etf_regression_results.csv
Output: data/results/table_regression_wide.csv   (stargazer-style)
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from labels import label

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

IN_PATH  = PROJECT_ROOT / "data/results/etf_regression_results.csv"
OUT_WIDE = PROJECT_ROOT / "data/results/table_regression_wide.csv"
OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)

TICKERS = ["ICLN", "IXC", "VDE", "XLE", "XOP"]


def main():
    df = pd.read_csv(IN_PATH)
    df["Sig"] = df["Sig"].fillna("")

    # ── Stargazer-style wide table ──────────────────────────────────────────
    variables = df[df["ETF"] == TICKERS[0]]["Variable"].tolist()

    rows = []
    for var in variables:
        coef_row = {"Variable": label(var)}
        se_row   = {"Variable": ""}
        for etf in TICKERS:
            r = df[(df["ETF"] == etf) & (df["Variable"] == var)].iloc[0]
            coef_row[etf] = f"{r['Coefficient']:.4f}{r['Sig']}"
            se_row[etf]   = f"({r['Std_Error']:.4f})"
        rows.append(coef_row)
        rows.append(se_row)

    # Model stats at the bottom
    for stat_label, col in [("R²", "R2"), ("Adj. R²", "Adj_R2"), ("N", "N")]:
        stat_row = {"Variable": stat_label}
        for etf in TICKERS:
            val = df[df["ETF"] == etf][col].iloc[0]
            stat_row[etf] = int(val) if col == "N" else val
        rows.append(stat_row)

    wide = pd.DataFrame(rows, columns=["Variable"] + TICKERS)
    wide.to_csv(OUT_WIDE, index=False)
    print(f"Saved → {OUT_WIDE}")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
