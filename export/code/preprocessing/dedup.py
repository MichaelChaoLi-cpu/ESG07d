import argparse
import csv
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
RAW_DATA_PATH = Path(os.environ["RAW_DATA_PATH"])

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "deduped_aug"
DONE_DIR   = OUTPUT_DIR / ".done"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "dedup_aug_stats.csv"
WINDOW = 7

META_COLS = ["article_id", "publication_date", "title", "description", "content"]
STATS_HEADER = ["date", "loaded", "removed_by_id", "after_id_dedup", "removed_by_title", "after_title_dedup", "saved"]


def load_parquet(d: date, columns: Optional[list] = None) -> Optional[pd.DataFrame]:
    raw_path = RAW_DATA_PATH / f"{d}.parquet"
    if not raw_path.exists():
        return None
    return pd.read_parquet(raw_path, columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = sorted(
        date.fromisoformat(p.stem)
        for p in RAW_DATA_PATH.glob("*.parquet")
        if p.stem.count("-") == 2
    )
    if not all_dates:
        print("No parquet files found.")
        return

    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")

    existing_stats: dict[str, dict] = {}
    if STATS_PATH.exists():
        with open(STATS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                existing_stats[row["date"]] = row

    for target_date in all_dates:
        done_marker = DONE_DIR / str(target_date)
        out_path    = OUTPUT_DIR / f"{target_date}.parquet"
        if done_marker.exists() and out_path.exists() and not args.rerun:
            saved = existing_stats.get(str(target_date), {})
            if saved:
                print(
                    f"skip {target_date}: loaded={saved['loaded']}  "
                    f"-id={saved['removed_by_id']}  after_id={saved['after_id_dedup']}  "
                    f"-title={saved['removed_by_title']}  after_title={saved['after_title_dedup']}  "
                    f"saved={saved['saved']}"
                )
            else:
                print(f"skip {target_date}")
            continue

        window_dates = sorted([
            target_date + timedelta(days=d) for d in range(-WINDOW, WINDOW + 1)
        ])

        # Phase 1: load metadata only (no embedding) for dedup
        frames = [
            df for wd in window_dates
            if (df := load_parquet(wd, columns=META_COLS)) is not None
        ]
        combined = pd.concat(frames, ignore_index=True)
        combined["publication_date"] = pd.to_datetime(combined["publication_date"], utc=True)
        n_raw = len(combined)

        # Step 1: deduplicate article_id, keep first occurrence (earliest file date)
        combined = combined.drop_duplicates(subset=["article_id"], keep="first")
        n_after_id = len(combined)
        n_removed_id = n_raw - n_after_id

        # Step 2: deduplicate by title+content, keep earliest publication_date
        combined["_content_key"] = combined["title"].str.strip() + "\n" + combined["content"].str.strip()
        combined = (
            combined
            .sort_values("publication_date", ascending=True)
            .drop_duplicates(subset=["_content_key"], keep="first")
            .drop(columns=["_content_key"])
        )
        n_after_title = len(combined)
        n_removed_title = n_after_id - n_after_title

        # Keep only articles whose publication date matches target_date (New York time)
        pub_dates = combined["publication_date"].dt.tz_convert("America/New_York").dt.date
        surviving = combined[pub_dates == target_date][["article_id"]].copy()

        if surviving.empty:
            out_path.touch()
            done_marker.touch()
            existing_stats[str(target_date)] = {
                "date": target_date, "loaded": n_raw,
                "removed_by_id": n_removed_id, "after_id_dedup": n_after_id,
                "removed_by_title": n_removed_title, "after_title_dedup": n_after_title,
                "saved": 0,
            }
            print(
                f"done {target_date}: loaded={n_raw}  "
                f"-id={n_removed_id}  after_id={n_after_id}  "
                f"-title={n_removed_title}  after_title={n_after_title}  saved=0"
            )
            continue

        # Phase 2: fetch embeddings for surviving articles only
        surviving_ids = set(surviving["article_id"])
        emb_frames = []
        for wd in window_dates:
            df_emb = load_parquet(wd, columns=["article_id", "embedding"])
            if df_emb is None:
                continue
            matches = df_emb[df_emb["article_id"].isin(surviving_ids)]
            if not matches.empty:
                emb_frames.append(matches)
                surviving_ids -= set(matches["article_id"])
            if not surviving_ids:
                break

        # Phase 3: merge metadata + embeddings and save
        result = combined[pub_dates == target_date].reset_index(drop=True)
        if emb_frames:
            emb_df = pd.concat(emb_frames, ignore_index=True).drop_duplicates(subset=["article_id"])
            result = result.merge(emb_df, on="article_id", how="left")

        try:
            result.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"[ERROR] {target_date}: failed to write parquet — {e}")
            continue

        done_marker.touch()
        existing_stats[str(target_date)] = {
            "date": target_date, "loaded": n_raw,
            "removed_by_id": n_removed_id, "after_id_dedup": n_after_id,
            "removed_by_title": n_removed_title, "after_title_dedup": n_after_title,
            "saved": len(result),
        }
        print(
            f"done {target_date}: loaded={n_raw}  "
            f"-id={n_removed_id}  after_id={n_after_id}  "
            f"-title={n_removed_title}  after_title={n_after_title}  saved={len(result)}"
        )


    with open(STATS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_HEADER)
        writer.writeheader()
        for d in sorted(existing_stats):
            writer.writerow({k: existing_stats[d][k] for k in STATS_HEADER})
    print(f"Stats written to {STATS_PATH}")


if __name__ == "__main__":
    main()
