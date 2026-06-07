"""
Q5 — High-impact international events summary table.

Parses event.txt and generates a human-readable event table with
countries, search keywords, and a short description of each event.

Input:  data/processed/country_tfidf/event.txt
Output: data/results/table_events.csv
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

EVENT_FILE = PROJECT_ROOT / "data/processed/country_tfidf/event.txt"
OUT_PATH   = PROJECT_ROOT / "data/results/table_events.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Short human-readable descriptions for each event label
DESCRIPTIONS = {
    "ARE_SAU": "COP28 climate and energy summit hosted in UAE (Dubai), with Saudi Arabia as key OPEC actor",
    "ESP":     "Spain power grid outage and energy supply disruption",
    "BRA":     "Brazil as host of COP30; domestic energy and climate policy spotlight",
    "GEO":     "Georgia natural gas pipeline dispute and transit risk",
    "DEU":     "Nord Stream pipeline sabotage and Germany's energy supply crisis",
    "IRN":     "Iran military conflict and oil export sanction escalation",
    "ISR":     "Iran–Israel conflict in the Gulf affecting regional oil supply routes",
    "UKR":     "Russia–Ukraine war and its direct impact on European energy prices",
}


def parse_events(path: Path) -> list:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        label_part, _, terms_part = raw.partition(":")
        label    = label_part.strip().replace("; ", "_").replace(";", "_")
        keywords = "; ".join(t.strip() for t in terms_part.split(";"))
        rows.append({
            "Event Label": label,
            "Keywords":    keywords,
            "Description": DESCRIPTIONS.get(label, ""),
        })
    return rows


def main():
    rows = parse_events(EVENT_FILE)
    df = pd.DataFrame(rows, columns=["Event Label", "Keywords", "Description"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
