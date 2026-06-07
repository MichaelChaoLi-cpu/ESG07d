"""
build_export.py — Assembles the export/ directory for Jiazi handoff.

Steps:
  1. Copy figures (data/results/fig_*.png → export/figures/)
  2. Convert tables (data/results/table_*.csv → export/tables/*.xlsx)
  3. Copy src/ .py files → export/code/ (preserving subdirectory structure)
  4. Write export/code/run_analysis.py
  5. Copy docs/AnaSOP.md → export/AnaSOP.md
  6. Metadata files already live at export/metadata/ (committed to repo)

Usage:
    python build_export.py
"""

import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

SRC_DIR     = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "data/results"
EXPORT_DIR  = PROJECT_ROOT / "export"

FIGURES_OUT = EXPORT_DIR / "figures"
TABLES_OUT  = EXPORT_DIR / "tables"
CODE_OUT    = EXPORT_DIR / "code"


def copy_figures():
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(RESULTS_DIR.glob("fig_*.png")):
        dst = FIGURES_OUT / src.name
        shutil.copy2(src, dst)
        copied.append(dst.name)
    print(f"Figures  ({len(copied)}): {', '.join(copied)}")


def convert_tables():
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(RESULTS_DIR.glob("table_*.csv")):
        dst = TABLES_OUT / src.with_suffix(".xlsx").name
        df = pd.read_csv(src)
        df.to_excel(dst, index=False)
        copied.append(dst.name)
    print(f"Tables   ({len(copied)}): {', '.join(copied)}")


def copy_src():
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in SRC_DIR.rglob("*.py"):
        if ".ipynb_checkpoints" in src.parts:
            continue
        rel = src.relative_to(SRC_DIR)
        dst = CODE_OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    print(f"Code     ({count} .py files copied from src/)")


def write_run_analysis():
    content = '''\
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
    print(f"\\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path), "--output-dir", args.output_dir],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

print("\\nAll analyses completed successfully.")
'''
    dst = CODE_OUT / "run_analysis.py"
    dst.write_text(content, encoding="utf-8")
    print(f"Code     run_analysis.py written")


def copy_anasop():
    src = PROJECT_ROOT / "docs/AnaSOP.md"
    dst = EXPORT_DIR / "AnaSOP.md"
    shutil.copy2(src, dst)
    print(f"AnaSOP   {dst.relative_to(PROJECT_ROOT)}")


def main():
    print(f"Building export/ at {EXPORT_DIR}\n")
    copy_figures()
    convert_tables()
    copy_src()
    write_run_analysis()
    copy_anasop()
    print(f"\nDone. Export directory: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
