"""
run_analysis.py — Reproduces all exported figures and tables for ESG07d.

Assumes preprocessing outputs already exist in data/processed/ and
data/results/etf_regression_results.csv is present.

Usage:
    python run_analysis.py [--output-dir PATH]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

ANALYSES_DIR = Path(__file__).parent / "analyses"

SCRIPTS = [
    "fig_prop_monthly.py",
    "fig_score_monthly.py",
    "fig_country_map.py",
    "fig_country_zscore_heatmap.py",
    "table_events.py",
    "table_regression.py",
]

parser = argparse.ArgumentParser(description="Run all ESG07d analyses.")
parser.add_argument(
    "--output-dir",
    default=str(PROJECT_ROOT / "data/results"),
    help="Directory where figures and tables are written.",
)
args = parser.parse_args()

for script in SCRIPTS:
    script_path = ANALYSES_DIR / script
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path), "--output-dir", args.output_dir],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

print("\nAll analyses completed successfully.")
