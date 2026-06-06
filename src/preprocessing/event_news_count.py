"""
Count daily occurrences of 7 international events in energy-price-relevant news.

Input:  data/processed/country_tfidf/event.txt
        data/processed/energy_price_scores/{date}.parquet
        data/processed/deduped_aug/{date}.parquet

Matching rules (case-insensitive):
  - "term1; term2"  → both terms must appear anywhere in the text (non-consecutive)
  - "term1 term2"   → the exact phrase must appear consecutively

Output: data/processed/event_news_daily.csv
  Columns: date, {event_label...}, total_relevant_articles
"""

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

EVENT_FILE   = PROJECT_ROOT / "data/processed/country_tfidf/event.txt"
SCORES_DIR   = PROJECT_ROOT / "data/processed/energy_price_scores"
DEDUPED_DIR  = PROJECT_ROOT / "data/processed/deduped_aug"
OUT_PATH     = PROJECT_ROOT / "data/processed/event_news_daily.csv"

from config import ENERGY_PRICE_THRESHOLD


# ── Parse event.txt ───────────────────────────────────────────────────────────

def parse_events(path: Path) -> dict:
    """
    Returns {label: compiled_matcher} where matcher is a function(text) -> bool.
    Line format:  LABEL: term1[; term2]
    """
    events = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        label_part, _, terms_part = raw.partition(":")
        label = label_part.strip().replace("; ", "_").replace(";", "_").replace(" ", "_")
        terms = [t.strip() for t in terms_part.split(";")]

        if len(terms) == 1:
            # Consecutive phrase match
            pattern = re.compile(re.escape(terms[0]), re.IGNORECASE)
            events[label] = lambda text, p=pattern: bool(p.search(text))
        else:
            # Both terms must appear independently (non-consecutive)
            patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in terms]
            events[label] = lambda text, ps=patterns: all(p.search(text) for p in ps)

    return events


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    events = parse_events(EVENT_FILE)
    print(f"Events loaded ({len(events)}):")
    for label in events:
        print(f"  {label}")

    score_files = sorted(SCORES_DIR.glob("*.parquet"))
    print(f"\nProcessing {len(score_files)} days …")

    rows = []
    for score_path in tqdm(score_files, unit="day", dynamic_ncols=True):
        date = score_path.stem

        # Filter to energy-price-relevant articles
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
        total = len(rel_ids)
        if total == 0:
            rows.append({"date": date, **{k: 0 for k in events}, "total_relevant_articles": 0})
            continue

        # Load texts for relevant articles
        deduped_path = DEDUPED_DIR / score_path.name
        if not deduped_path.exists():
            rows.append({"date": date, **{k: 0 for k in events}, "total_relevant_articles": total})
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

        row = {"date": date, "total_relevant_articles": total}
        for label, matcher in events.items():
            row[label] = int(text_df["text"].apply(matcher).sum())
        rows.append(row)

    # Build output table
    cols = ["date"] + list(events.keys()) + ["total_relevant_articles"]
    out_df = pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}  ({len(out_df)} rows × {len(out_df.columns)} cols)")
    print(out_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
