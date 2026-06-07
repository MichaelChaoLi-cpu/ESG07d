import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import ENERGY_PRICE_THRESHOLD

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

SCORES_DIR  = PROJECT_ROOT / "data/processed/energy_price_scores"
DEDUPED_DIR = PROJECT_ROOT / "data/processed/deduped_aug"
COUNTRY_JSON = PROJECT_ROOT / "data/raw/country_dict_0.json"
OUT_CSV      = PROJECT_ROOT / "data/processed/country_daily_mentions.csv"

MIN_ALIAS_LEN = 4


def build_country_patterns(country_json):
    """Return list of (country_id, compiled_regex), skipping countries with no valid alias."""
    data = json.loads(country_json.read_text(encoding="utf-8"))
    patterns = []
    for entry in data:
        aliases = [
            name for name in entry.get("country_eng", [])
            if len(name) >= MIN_ALIAS_LEN
        ]
        if not aliases:
            continue
        regex = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )
        patterns.append((entry["country_id"], regex))
    return patterns


def process_day(score_path, patterns):
    """Return list of dicts {country_id, count, rel_count} for one day."""
    scores_df = pd.read_parquet(score_path)
    score_cols = [c for c in scores_df.columns if c.startswith("score_")]
    if not score_cols:
        return None

    rel_ids = scores_df.loc[
        (scores_df[score_cols] > ENERGY_PRICE_THRESHOLD).all(axis=1), "article_id"
    ]
    rel_count = len(rel_ids)
    if rel_count == 0:
        return None

    deduped_path = DEDUPED_DIR / score_path.name
    if not deduped_path.exists():
        return None

    text_df = pd.read_parquet(
        deduped_path, columns=["article_id", "title", "description", "content"]
    )
    text_df = text_df[text_df["article_id"].isin(rel_ids)].copy()
    text_df["text"] = (
        text_df["title"].fillna("") + " "
        + text_df["description"].fillna("") + " "
        + text_df["content"].fillna("")
    )
    texts = text_df["text"].tolist()

    rows = []
    for country_id, regex in patterns:
        count = sum(1 for t in texts if regex.search(t))
        rows.append({
            "country_id": country_id,
            "count":      count,
            "rel_count":  rel_count,
        })
    return rows


def main():
    score_files = sorted(SCORES_DIR.glob("*.parquet"))
    if not score_files:
        print("No score files found.")
        return
    print(f"Found {len(score_files)} score files")

    print("Building country regex patterns …")
    patterns = build_country_patterns(COUNTRY_JSON)
    print(f"{len(patterns)} countries with valid aliases (>= {MIN_ALIAS_LEN} chars)")

    all_rows = []
    for score_path in tqdm(score_files, desc="processing", unit="day", dynamic_ncols=True):
        date = score_path.stem
        rows = process_day(score_path, patterns)
        if rows is None:
            continue
        for r in rows:
            r["date"] = date
        all_rows.extend(rows)

    if not all_rows:
        print("No data produced.")
        return

    result_df = pd.DataFrame(all_rows, columns=["date", "country_id", "count", "rel_count"])
    result_df["proportion"] = (result_df["count"] / result_df["rel_count"]).round(6)
    result_df = result_df[["date", "country_id", "count", "proportion", "rel_count"]]
    result_df = result_df.sort_values(["date", "country_id"]).reset_index(drop=True)

    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nWritten to {OUT_CSV}  ({len(result_df):,} rows)")


if __name__ == "__main__":
    main()
