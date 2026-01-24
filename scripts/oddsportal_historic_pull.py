"""One-time OddsPortal historical pull (local run).

Goal
----
Build a local historical "market priors" database to support Enhancements.txt.

We want, per game:
- pregame total line
- pregame spread line
- implied team totals fallback (OddsHarvester does not support team totals directly)

Design principles (because we’re not animals)
-------------------------------------------
- Deterministic: overwrites output files each run unless `--resume`
- Resumable: writes JSONL incrementally and can skip already-seen match_links
- Fast-ish: uses OddsHarvester `--preview_submarkets_only` so we scrape the visible
  main market tables instead of clicking 50 submarkets.

How it works
------------
1) Use OddsHarvester to scrape a season directly (it handles JS + pagination).
2) Normalize OddsHarvester output into compact JSONL rows.

Usage
-----
  python3 scripts/oddsportal_historic_pull.py \
    --odds-harvester /path/to/OddsHarvester \
    --season 2023-2024 \
    --pages 60 \
    --out data/oddsportal/historic_bet365_2023-2024.jsonl \
    --headless

Tip:
- Start with `--pages 1 --dry-run` to validate quickly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.oddsportal.market_priors import extract_priors_from_odds_harvester_match


def _run(cmd: list[str], cwd: Path) -> None:
    res = subprocess.run(cmd, cwd=str(cwd), check=False, text=True, capture_output=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def _load_json(path: Path):
    return json.loads(path.read_text())


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_seen_match_links(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ml = obj.get("match_link")
            if isinstance(ml, str) and ml:
                seen.add(ml)
    return seen



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds-harvester", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument(
        "--pages",
        type=int,
        required=True,
        help="Max pages for OddsHarvester season pagination (upper bound).",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--tmp-dir", default="data/oddsportal/_tmp_historic")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    oh_dir = Path(args.odds_harvester).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output file exists so downstream steps don't crash on FileNotFound.
    if not args.resume and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)

    seen = _read_seen_match_links(out_path) if args.resume else set()

    if args.dry_run:
        print("Dry-run enabled; not scraping OddsHarvester.")
        print(f"Would scrape season {args.season} with max_pages={args.pages}")
        return

    # Preview mode trick: request only representative markets.
    markets = [
        "home_away",
        "over_under_games_220_5",  # representative for Over/Under tab
        "asian_handicap_games_-3_5_games",  # representative for Asian Handicap tab
    ]

    tmp_raw = tmp_dir / f"historic_{args.season}.raw.json"
    try:
        tmp_raw.unlink()
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "scrape_historic",
        "--sport",
        "basketball",
        "--leagues",
        "nba",
        "--season",
        args.season,
        "--max_pages",
        str(args.pages),
        "--markets",
        ",".join(markets),
        "--target_bookmaker",
        "bet365.us",
        "--preview_submarkets_only",
        "--format",
        "json",
        "--file_path",
        str(tmp_raw),
    ]
    if args.headless:
        cmd.append("--headless")

    print(f"\nScraping season {args.season} via OddsHarvester (max_pages={args.pages})")
    _run(cmd, cwd=oh_dir)

    scraped = _load_json(tmp_raw)
    if not isinstance(scraped, list):
        raise TypeError(f"Expected list output from OddsHarvester, got {type(scraped)}")

    normalized: list[dict] = []
    skipped_seen = 0

    for m in scraped:
        if not isinstance(m, dict):
            continue
        ml = m.get("match_link")
        if args.resume and isinstance(ml, str) and ml in seen:
            skipped_seen += 1
            continue

        priors = extract_priors_from_odds_harvester_match(m)
        normalized.append(
            {
                "match_link": ml,
                "match_date": m.get("match_date"),
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "bookmaker": "bet365.us",
                **asdict(priors),
            }
        )

    _append_jsonl(out_path, normalized)

    print(f"\nDone. Historical priors DB at: {out_path}")
    print(f"Raw matches scraped: {len(scraped)}")
    if args.resume:
        print(f"Skipped already-seen match_links: {skipped_seen}")
    print(f"Rows appended this run: {len(normalized)}\n")


if __name__ == "__main__":
    main()