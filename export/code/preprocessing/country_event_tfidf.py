"""
Step 2: Country-level event TF-IDF analysis.

Requires: country_article_index.py to have been run first.

For each anomaly country:
  - Merge anomaly dates into windows (gap > 14 days → separate window)
  - Window group : articles in [window_start, window_end + 14)
  - Control group: all other articles for that country
  - TF-IDF (ngram 1-4, lowercase, min_df=200)
  - Distinctiveness = mean_tfidf_window - mean_tfidf_control
  - Top 100 terms saved to data/processed/country_tfidf/{country_id}.csv
    Columns: window_id, window_start, window_end, ngram,
             distinctiveness, mean_tfidf_window, mean_tfidf_control, doc_freq_window
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

DEDUPED_DIR = PROJECT_ROOT / "data/processed/deduped_aug"
CONSENSUS   = PROJECT_ROOT / "data/processed/country_anomaly_consensus.csv"
TOP30_CSV   = PROJECT_ROOT / "data/processed/top30_countries.csv"
INDEX_DIR   = PROJECT_ROOT / "data/processed/country_articles"
OUT_DIR     = PROJECT_ROOT / "data/processed/country_tfidf"
LOG_FILE    = OUT_DIR / "tfidf_log.csv"

LOG_FIELDS = [
    "timestamp", "country_name", "country_id",
    "window_id", "window_start", "window_end_ext",
    "n_window_articles", "n_control_articles",
    "n_output_rows", "status", "note",
]

WINDOW_DAYS   = 14
GAP_THRESHOLD = 14
MIN_DF        = 200
MIN_WINDOW_ARTICLES      = 200
MIN_WINDOW_ARTICLES_EARLY = 100   # relaxed threshold for 2022-2023 windows
TOP_N         = 100


def append_log(row: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def compute_windows(dates):
    """
    Group anomaly dates into windows.
    Merge criterion: if the current window's extended end (w_end + WINDOW_DAYS)
    covers the next date's start, absorb it into the same window.
    If the extended end of window W1 reaches into a separately-formed window W2,
    the two are combined and the final cutoff becomes W2_end + WINDOW_DAYS.
    """
    dates = sorted(dates)
    windows, wid = [], 1
    start = end = dates[0]
    for d in dates[1:]:
        # Merge if w_end + WINDOW_DAYS covers d (i.e. extended end >= d)
        if end + pd.Timedelta(days=WINDOW_DAYS) >= d:
            end = d
        else:
            windows.append((wid, start, end))
            wid += 1
            start = end = d
    windows.append((wid, start, end))
    return windows


def load_texts_for_country(country_id, date_strs):
    """Load texts for the given country on the given dates. Returns {date_str: [texts]}."""
    country_dir = INDEX_DIR / country_id
    result = {}
    for date in date_strs:
        idx_path = country_dir / f"{date}.parquet"
        if not idx_path.exists():
            continue
        aids = set(pd.read_parquet(idx_path)["article_id"])
        if not aids:
            continue
        deduped_path = DEDUPED_DIR / f"{date}.parquet"
        if not deduped_path.exists():
            continue
        text_df = pd.read_parquet(
            deduped_path, columns=["article_id", "title", "description", "content"]
        )
        text_df = text_df[text_df["article_id"].isin(aids)].copy()
        text_df["text"] = (
            text_df["title"].fillna("") + " "
            + text_df["description"].fillna("") + " "
            + text_df["content"].fillna("")
        )
        if not text_df.empty:
            result[date] = text_df["text"].tolist()
    return result


def run_tfidf(window_texts, control_texts):
    if not window_texts or not control_texts:
        return None
    n_window = len(window_texts)
    all_texts = window_texts + control_texts
    vec = TfidfVectorizer(ngram_range=(1, 4), lowercase=True, min_df=MIN_DF)
    try:
        matrix = vec.fit_transform(all_texts)
    except ValueError:
        return None

    feature_names = np.array(vec.get_feature_names_out())
    mean_win  = np.asarray(matrix[:n_window].mean(axis=0)).flatten()
    mean_ctrl = np.asarray(matrix[n_window:].mean(axis=0)).flatten()
    distinctiveness = mean_win - mean_ctrl

    doc_freq = np.diff(matrix[:n_window].tocsc().indptr)
    top_idx  = np.argsort(distinctiveness)[::-1][:TOP_N]

    return pd.DataFrame({
        "ngram":              feature_names[top_idx],
        "distinctiveness":    distinctiveness[top_idx].round(6),
        "mean_tfidf_window":  mean_win[top_idx].round(6),
        "mean_tfidf_control": mean_ctrl[top_idx].round(6),
        "doc_freq_window":    doc_freq[top_idx],
    })


def main(rerun: bool = False):
    consensus = pd.read_csv(CONSENSUS, parse_dates=["date"])
    top30     = pd.read_csv(TOP30_CSV)
    name_to_id = top30.set_index("country_name")["country_id"].to_dict()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for country_name, grp in tqdm(consensus.groupby("country"), unit="country", dynamic_ncols=True):
        country_id = name_to_id.get(country_name)
        if not country_id:
            continue

        out_path = OUT_DIR / f"{country_id}.csv"
        if out_path.exists() and not rerun:
            tqdm.write(f"[SKIP] {country_name}: already done")
            continue

        country_dir = INDEX_DIR / country_id
        if not country_dir.exists():
            tqdm.write(f"[SKIP] {country_name}: index not found")
            continue

        # All dates available for this country
        all_dates = sorted(p.stem for p in country_dir.glob("*.parquet"))
        if not all_dates:
            tqdm.write(f"[SKIP] {country_name}: no index files")
            continue

        windows = compute_windows(grp["date"].tolist())

        # Build article date sets per window: [w_start, w_end + WINDOW_DAYS)
        # compute_windows already merges W1 into W2 when W1_end + WINDOW_DAYS >= W2_start,
        # so each window here is already the combined span.
        window_date_sets = []
        for wid, w_start, w_end in windows:
            cutoff = w_end + pd.Timedelta(days=WINDOW_DAYS)
            wdates = [d for d in all_dates
                      if w_start <= pd.Timestamp(d) < cutoff]
            window_date_sets.append(wdates)

        all_window_dates = set(d for wdates in window_date_sets for d in wdates)
        control_dates    = [d for d in all_dates if d not in all_window_dates]

        # Load control texts once
        ctrl_texts_by_date = load_texts_for_country(country_id, control_dates)
        control_texts = [t for texts in ctrl_texts_by_date.values() for t in texts]

        all_results = []
        for (wid, w_start, w_end), wdates in zip(windows, window_date_sets):
            win_texts_by_date = load_texts_for_country(country_id, wdates)
            window_texts = [t for texts in win_texts_by_date.values() for t in texts]

            tqdm.write(f"  {country_name} W{wid}: window={len(window_texts)} ctrl={len(control_texts)} articles")

            threshold = MIN_WINDOW_ARTICLES_EARLY if w_start.year in (2022, 2023) else MIN_WINDOW_ARTICLES
            w_end_ext = (w_end + pd.Timedelta(days=WINDOW_DAYS) - pd.Timedelta(days=1)).date()
            base_log  = dict(
                timestamp          = datetime.now().isoformat(timespec="seconds"),
                country_name       = country_name,
                country_id         = country_id,
                window_id          = wid,
                window_start       = w_start.date(),
                window_end_ext     = w_end_ext,
                n_window_articles  = len(window_texts),
                n_control_articles = len(control_texts),
            )
            force = country_id in {"RUS"}
            if not force and len(window_texts) < threshold:
                tqdm.write(f"    [SKIP] window too small ({len(window_texts)} < {threshold})")
                append_log({**base_log, "status": "skip_too_small",
                            "note": f"{len(window_texts)} < {threshold}"})
                continue

            result = run_tfidf(window_texts, control_texts)
            if result is None:
                tqdm.write(f"    [SKIP] not enough terms (min_df={MIN_DF})")
                append_log({**base_log, "status": "skip_min_df",
                            "note": f"min_df={MIN_DF}"})
                continue

            result.insert(0, "window_end",   w_end_ext)
            result.insert(0, "window_start", w_start.date())
            result.insert(0, "window_id",    wid)
            all_results.append(result)
            append_log({**base_log, "status": "ok", "n_output_rows": len(result)})

        if not all_results:
            continue

        out_df = pd.concat(all_results, ignore_index=True)
        out_df.to_csv(out_path, index=False)
        tqdm.write(f"  Saved → {out_path.name}  ({len(out_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="Reprocess already-done countries")
    args = parser.parse_args()
    main(rerun=args.rerun)
