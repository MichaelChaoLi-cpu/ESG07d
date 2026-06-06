import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

MENTIONS_CSV = PROJECT_ROOT / "data/processed/country_daily_mentions.csv"
COUNTRY_JSON = PROJECT_ROOT / "data/raw/country_dict_0.json"
OUT_CSV      = PROJECT_ROOT / "data/processed/top30_countries.csv"


def load_country_names(country_json: Path) -> dict:
    data = json.loads(country_json.read_text(encoding="utf-8"))
    return {
        entry["country_id"]: entry["country_eng"][0]
        for entry in data
        if entry.get("country_eng")
    }


def main():
    df = pd.read_csv(MENTIONS_CSV, dtype={"country_id": str})

    total = (
        df[df["country_id"] != "USA"]
        .groupby("country_id")["count"]
        .sum()
        .rename("total_count")
        .reset_index()
        .sort_values("total_count", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    total.index += 1
    total.index.name = "rank"

    country_names = load_country_names(COUNTRY_JSON)
    total.insert(1, "country_name", total["country_id"].map(country_names))

    print(total.to_string())
    total.to_csv(OUT_CSV)
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
