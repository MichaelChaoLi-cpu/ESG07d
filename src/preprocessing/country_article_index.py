"""
Step 1: Build per-country article index.

For each day, find energy-price-relevant articles that mention each anomaly country.
Save article_ids to data/processed/country_articles/{country_id}/{date}.parquet

Run this before country_event_tfidf.py.
"""

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

SCORES_DIR   = PROJECT_ROOT / "data/processed/energy_price_scores"
DEDUPED_DIR  = PROJECT_ROOT / "data/processed/deduped_aug"
COUNTRY_JSON = PROJECT_ROOT / "data/raw/country_dict_0.json"
CONSENSUS    = PROJECT_ROOT / "data/processed/country_anomaly_consensus.csv"
TOP30_CSV    = PROJECT_ROOT / "data/processed/top30_countries.csv"
INDEX_DIR    = PROJECT_ROOT / "data/processed/country_articles"

MIN_ALIAS_LEN = 4


def build_patterns(country_ids):
    data = json.loads(COUNTRY_JSON.read_text(encoding="utf-8"))
    patterns = []
    for entry in data:
        cid = entry["country_id"]
        if cid not in country_ids:
            continue
        aliases = [a for a in entry.get("country_eng", []) if len(a) >= MIN_ALIAS_LEN]
        if not aliases:
            continue
        regex = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )
        patterns.append((cid, regex))
    return patterns


def main():
    top30 = pd.read_csv(TOP30_CSV)
    consensus = pd.read_csv(CONSENSUS)
    name_to_id = top30.set_index("country_name")["country_id"].to_dict()
    anomaly_ids = {name_to_id[n] for n in consensus["country"].unique() if n in name_to_id}
    print(f"Anomaly countries: {len(anomaly_ids)}")

    # Create per-country output dirs
    for cid in anomaly_ids:
        (INDEX_DIR / cid).mkdir(parents=True, exist_ok=True)

    patterns = build_patterns(anomaly_ids)
    score_files = sorted(SCORES_DIR.glob("*.parquet"))
    print(f"Processing {len(score_files)} days …")

    for score_path in tqdm(score_files, unit="day", dynamic_ncols=True):
        date = score_path.stem

        # Skip if all country files for this date already exist
        if all((INDEX_DIR / cid / f"{date}.parquet").exists() for cid in anomaly_ids):
            continue

        scores_df = pd.read_parquet(score_path)
        score_cols = [c for c in scores_df.columns if c.startswith("score_")]
        if not score_cols:
            continue

        rel_ids = set(
            scores_df.loc[
                (scores_df[score_cols] > ENERGY_PRICE_THRESHOLD).all(axis=1),
                "article_id",
            ]
        )

        deduped_path = DEDUPED_DIR / score_path.name
        if not deduped_path.exists():
            continue

        text_df = pd.read_parquet(
            deduped_path, columns=["article_id", "title", "description", "content"]
        )
        text_df = text_df[text_df["article_id"].isin(rel_ids)].copy()
        text_df["text"] = (
            text_df["title"].fillna("") + " "
            + text_df["description"].fillna("") + " "
            + text_df["content"].fillna("")
        )

        # Match each country and write its article_ids
        for cid, regex in patterns:
            out_path = INDEX_DIR / cid / f"{date}.parquet"
            if out_path.exists():
                continue
            matched_ids = text_df.loc[
                text_df["text"].apply(lambda t: bool(regex.search(t))), "article_id"
            ]
            pd.DataFrame({"article_id": matched_ids}).to_parquet(out_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
