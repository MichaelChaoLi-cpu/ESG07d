import argparse
import os
import re

# Must be set BEFORE tensorflow is imported
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer

from config import ENERGY_PRICE_KEYWORDS

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

DEDUPED_DIR = PROJECT_ROOT / "data/processed/deduped_aug"
OUTPUT_DIR  = PROJECT_ROOT / "data/processed/energy_price_scores"
MODEL_DIR   = PROJECT_ROOT / "model"

KEYWORDS = ENERGY_PRICE_KEYWORDS

KW_SEQ_LEN      = 16
BERT_CHECKPOINT = "bert-base-multilingual-cased"


def _col_name(keyword: str) -> str:
    return "score_" + re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")


SCORE_COLS = [_col_name(kw) for kw in KEYWORDS]

# ---------------------------------------------------------------------------
# Phase 0 — Load models
# ---------------------------------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Loading TMPTv2 (KwExtractor + Matcher)...")
kw_extractor = tf.saved_model.load(str(MODEL_DIR / "TMPTv2_KwExtractor"))
matcher      = tf.saved_model.load(str(MODEL_DIR / "TMPTv2_Matcher"))
kw_fn        = kw_extractor.signatures["serving_default"]
match_fn     = matcher.signatures["serving_default"]
print("TMPTv2 loaded.")

tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)
print("Tokenizer ready.")

# ---------------------------------------------------------------------------
# Phase 1 — Pre-compute all keyword representations (once)
# ---------------------------------------------------------------------------

kw_reprs = []
for kw in KEYWORDS:
    print(f"Computing keyword representation for '{kw}'...")
    with tf.device("/CPU:0"):
        kw_ids = tokenizer(
            [kw],
            return_tensors="tf",
            max_length=KW_SEQ_LEN,
            truncation=True,
            padding="max_length",
        )["input_ids"]
        kw_repr = kw_fn(input_kw=kw_ids)["kw_difference_dropout_6_res"]
    kw_reprs.append(kw_repr)
    print(f"  kw_repr shape: {kw_repr.shape}")

# ---------------------------------------------------------------------------
# Phase 2 — Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    day_files = sorted(DEDUPED_DIR.glob("*.parquet"))
    if not day_files:
        print("No deduped parquet files found.")
        return

    print(f"Found {len(day_files)} day files: {day_files[0].stem} to {day_files[-1].stem}")

    skipped = sum(1 for f in day_files if (OUTPUT_DIR / f.name).exists() and not args.rerun)
    pending = [f for f in day_files if not (OUTPUT_DIR / f.name).exists() or args.rerun]
    print(f"Already scored: {skipped} | To process: {len(pending)}")

    total_articles = 0
    total_errors   = 0

    for day_path in tqdm(pending, unit="day", dynamic_ncols=True):
        out_path = OUTPUT_DIR / day_path.name

        if day_path.stat().st_size == 0:
            tqdm.write(f"[SKIP] {day_path.stem}: empty file")
            continue

        df = pd.read_parquet(day_path, columns=["article_id", "publication_date", "embedding"])
        if df.empty:
            tqdm.write(f"[SKIP] {day_path.stem}: empty file")
            continue

        # Validate embeddings
        valid_mask = df["embedding"].apply(
            lambda e: np.asarray(e, dtype=np.float32).shape == (768,)
        )
        n_bad = (~valid_mask).sum()
        if n_bad:
            tqdm.write(f"[WARN] {day_path.stem}: {n_bad} rows with invalid embeddings dropped")
            total_errors += n_bad
            df = df[valid_mask].reset_index(drop=True)

        if df.empty:
            continue

        emb_matrix = np.stack(df["embedding"].to_numpy()).astype(np.float32)
        N = len(emb_matrix)

        score_arrays = {}
        with tf.device("/CPU:0"):
            emb_tf = tf.constant(emb_matrix)
            for kw, col, kw_repr in zip(KEYWORDS, SCORE_COLS, kw_reprs):
                kw_tiled = tf.repeat(kw_repr, N, axis=0)
                scores = match_fn(
                    input_para_trans=emb_tf,
                    input_kw_trans=kw_tiled,
                )["output"].numpy().flatten().astype(np.float32)
                score_arrays[col] = scores

        out_df = pd.DataFrame({
            "article_id":         df["article_id"].values,
            "publication_date":   df["publication_date"].values,
            **score_arrays,
        })
        out_df.to_parquet(out_path, index=False)
        total_articles += N

    print(f"\nDone. {total_articles:,} articles scored | {total_errors:,} embedding errors")


if __name__ == "__main__":
    main()
