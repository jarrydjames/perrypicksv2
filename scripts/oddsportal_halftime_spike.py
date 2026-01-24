"""OddsPortal halftime (in-play) odds feasibility spike.

What this does:
- Runs OddsHarvester as an offline subprocess (Playwright scraping).
- Scrapes a small sample of NBA games from OddsPortal.
- Requests odds movement history (hover modal) and checks if it contains timestamps
  AFTER tipoff (a prerequisite for reconstructing halftime prices).

This is intentionally small + dumb. If this fails, we stop early and avoid building
half-time ROI backtesting on fantasy data.

Usage (from repo root):

  .venv/bin/python scripts/oddsportal_halftime_spike.py \
    --odds-harvester /Users/jarrydhawley/Desktop/Predictor/OddsHarvester \
    --season 2023-2024 \
    --max-pages 1 \
    --out data/oddsportal/spike_23_24.json

Notes:
- This does NOT join to NBA game_ids yet.
- This does NOT guarantee we can reconstruct halftime *lines* (only odds timestamps).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


@dataclass(frozen=True)
class SpikeConfig:
    odds_harvester_dir: Path
    season: str
    max_pages: int
    limit_matches: int
    target_bookmaker: str
    out_path: Path
    headless: bool


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _run(cmd: list[str], cwd: Path) -> None:
    # Keep output so user can debug scraping blocks.
    res = subprocess.run(cmd, cwd=str(cwd), check=False, text=True, capture_output=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dec_odds_to_prob(odds: float) -> float:
    # Decimal odds implied probability (ignores vig normalization)
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def _parse_market_line_from_key(key: str) -> float | None:
    # Examples:
    # - over_under_games_220_5_market -> 220.5
    # - asian_handicap_games_-3_5_games_market -> -3.5
    try:
        if key.startswith("over_under_games_"):
            s = key.replace("over_under_games_", "").replace("_market", "")
            return float(s.replace("_", "."))
        if key.startswith("asian_handicap_games_"):
            s = key.replace("asian_handicap_games_", "")
            s = s.replace("_games_market", "")
            s = s.replace("_games", "")
            return float(s.replace("_", "."))
    except Exception:
        return None
    return None


def _extract_pregame_priors_from_match(match: dict[str, Any]) -> dict[str, float | None]:
    """Extract spread/total from the scraped market rows.

    This is a pragmatic approximation to meet Enhancements.txt priors.
    """

    best_total_line: float | None = None
    best_total_score = 1e9

    best_spread_line: float | None = None
    best_spread_score = 1e9

    for k, v in (match or {}).items():
        if not (isinstance(k, str) and k.endswith("_market") and isinstance(v, list) and v):
            continue

        line = _parse_market_line_from_key(k)
        row = v[0] if isinstance(v[0], dict) else None
        if line is None or not row:
            continue

        # Totals
        if k.startswith("over_under_games_"):
            try:
                o = float(row.get("odds_over"))
                u = float(row.get("odds_under"))
            except Exception:
                continue
            score = abs(_dec_odds_to_prob(o) - 0.5) + abs(_dec_odds_to_prob(u) - 0.5)
            if score < best_total_score:
                best_total_score = score
                best_total_line = line

        # Spreads
        if k.startswith("asian_handicap_games_"):
            try:
                h1 = float(row.get("handicap_team_1"))
                h2 = float(row.get("handicap_team_2"))
            except Exception:
                continue
            score = abs(_dec_odds_to_prob(h1) - 0.5) + abs(_dec_odds_to_prob(h2) - 0.5)
            if score < best_spread_score:
                best_spread_score = score
                best_spread_line = line

    # Convention: spread line is from the market key. In OddsPortal, “team1” is home.
    total_line = best_total_line
    home_spread_line = best_spread_line

    home_tt = None
    away_tt = None
    if total_line is not None and home_spread_line is not None:
        # total = home + away
        # margin = home - away = -spread (if spread is home handicap)
        # If line is -3.5, home is favored by 3.5 -> implied home margin +3.5
        implied_margin = -home_spread_line
        home_tt = (total_line + implied_margin) / 2.0
        away_tt = (total_line - implied_margin) / 2.0

    return {
        "total_line": total_line,
        "home_spread_line": home_spread_line,
        "home_team_total_line_implied": home_tt,
        "away_team_total_line_implied": away_tt,
    }


def _extract_all_history_timestamps(scraped: list[dict[str, Any]]) -> list[datetime]:
    """Pull all parsed odds history timestamps into a flat list."""

    out: list[datetime] = []

    def parse_ts(ts: str) -> datetime | None:
        try:
            # OddsHarvester stores isoformat strings.
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    for match in scraped:
        for k, v in (match or {}).items():
            if not isinstance(k, str) or not k.endswith("_market"):
                continue
            if not isinstance(v, list):
                continue

            for odds_entry in v:
                if not isinstance(odds_entry, dict):
                    continue
                histories = odds_entry.get("odds_history_data")
                if not histories:
                    continue
                if not isinstance(histories, list):
                    continue

                for h in histories:
                    if not isinstance(h, dict):
                        continue
                    for item in h.get("odds_history") or []:
                        if not isinstance(item, dict):
                            continue
                        dt = parse_ts(str(item.get("timestamp") or ""))
                        if dt:
                            out.append(dt)

                    opening = h.get("opening_odds")
                    if isinstance(opening, dict):
                        dt = parse_ts(str(opening.get("timestamp") or ""))
                        if dt:
                            out.append(dt)

    return out


def _dedupe_matches(scraped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for match in scraped:
        if not isinstance(match, dict):
            continue
        link = match.get("match_link")
        if not isinstance(link, str) or not link:
            continue
        if link in seen:
            continue
        seen.add(link)
        out.append(match)

    return out


def _filter_bookmaker(scraped: list[dict[str, Any]], bookmaker_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for match in scraped:
        if not isinstance(match, dict):
            continue
        m2 = dict(match)
        for k, v in (match or {}).items():
            if not (isinstance(k, str) and k.endswith("_market") and isinstance(v, list)):
                continue
            m2[k] = [
                row
                for row in v
                if isinstance(row, dict) and row.get("bookmaker_name") == bookmaker_name
            ]
        out.append(m2)

    return out


def _summarize(scraped: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = _extract_all_history_timestamps(scraped)
    timestamps = sorted(set(timestamps))

    # We don't have tipoff times in this scrape-only phase, so we use a weaker check:
    # If OddsPortal only stores opening odds and pregame drift, timestamps will cluster
    # well before match start. If it stores in-play movement, we'd *expect* to see
    # timestamps all over the place (including hours after start).
    #
    # For now: we report range + count and let you inspect the distribution.

    if timestamps:
        t_min = timestamps[0]
        t_max = timestamps[-1]
    else:
        t_min = None
        t_max = None

    matches_with_any_history = 0
    for match in scraped:
        found = False
        for k, v in (match or {}).items():
            if isinstance(k, str) and k.endswith("_market") and isinstance(v, list):
                for odds_entry in v:
                    if isinstance(odds_entry, dict) and odds_entry.get("odds_history_data"):
                        found = True
                        break
            if found:
                break
        if found:
            matches_with_any_history += 1

    return {
        "scraped_matches": len(scraped),
        "matches_with_any_history": matches_with_any_history,
        "unique_history_timestamps": len(timestamps),
        "min_timestamp_utc": t_min.isoformat() if t_min else None,
        "max_timestamp_utc": t_max.isoformat() if t_max else None,
        "note": (
            "This spike only checks whether *any* odds-history timestamps exist. "
            "Next step (if promising) is to join to NBA game_id/tipoff time and check for "
            "timestamps within a halftime window."
        ),
    }


def _fetch_match_links_playwright(season: str, limit_matches: int, headless: bool) -> list[str]:
    """Extract a few match links from OddsPortal using Playwright.

    OddsPortal is JS-heavy, so this is more reliable than plain requests.
    """

    url = f"https://www.oddsportal.com/basketball/usa/nba-{season}/results/"

    links: list[str] = []

    # Match URLs look like:
    #   https://www.oddsportal.com/basketball/usa/nba-2023-2024/team-a-team-b-4dTtOD64/
    rx_match = re.compile(rf"^https://www\.oddsportal\.com/basketball/usa/nba-{re.escape(season)}/.+-[A-Za-z0-9]+/$")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        # A couple of scrolls to encourage lazy-loading.
        for _ in range(4):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(1000)

        anchors = page.query_selector_all("a[href]")
        for a in anchors:
            href = a.get_attribute("href")
            if not href:
                continue
            if not href.startswith("/basketball/usa/nba-"):
                continue
            if "/results" in href:
                continue
            if not href.endswith("/"):
                continue

            full = "https://www.oddsportal.com" + href
            if not rx_match.match(full):
                continue

            if full not in links:
                links.append(full)
            if len(links) >= limit_matches:
                break

        browser.close()

    return links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--odds-harvester",
        required=True,
        help="Path to local OddsHarvester repo (contains src/main.py)",
    )
    ap.add_argument("--season", required=True, help="NBA season string for OddsPortal (e.g. 2023-2024)")
    ap.add_argument(
        "--limit-matches",
        type=int,
        default=5,
        help="How many match links to scrape for the spike (default: 5).",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Unused in match_links mode (kept for backwards compatibility).",
    )
    ap.add_argument(
        "--target-bookmaker",
        default="bet365.us",
        help="Bookmaker name to isolate (must match OddsPortal naming exactly).",
    )
    ap.add_argument("--out", default="data/oddsportal/spike.json")
    ap.add_argument("--headless", action="store_true", help="Run browser headless (OddsHarvester flag).")
    args = ap.parse_args()

    cfg = SpikeConfig(
        odds_harvester_dir=Path(args.odds_harvester),
        season=str(args.season),
        max_pages=int(args.max_pages),
        limit_matches=int(args.limit_matches),
        target_bookmaker=str(args.target_bookmaker),
        out_path=Path(args.out),
        headless=bool(args.headless),
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_out_abs = cfg.out_path.expanduser().resolve()
    tmp_json = cfg_out_abs.with_suffix(".raw.json")

    # Ensure deterministic output (OddsHarvester may append/merge on existing files).
    for p in [tmp_json, cfg_out_abs.with_suffix(f".{cfg.target_bookmaker}.raw.json")]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    match_links = _fetch_match_links_playwright(
        season=cfg.season,
        limit_matches=cfg.limit_matches,
        headless=cfg.headless,
    )
    if not match_links:
        raise RuntimeError("Could not extract any match links via Playwright.")

    # Keep requested markets small (YAGNI): just enough for Enhancements priors.
    # - moneyline: home_away
    # - totals: a reasonable NBA band
    # - spreads: a reasonable NBA band
    totals_band = [
        f"over_under_games_{n}_5" for n in range(205, 241)
    ]  # 205.5 .. 240.5 (reasonable NBA band)
    spreads_band = [
        f"asian_handicap_games_{sign}{n}_5_games"
        for sign in ("-", "+")
        for n in range(1, 11)
    ] + ["asian_handicap_games_+0_5_games"]

    markets = ["home_away", *totals_band, *spreads_band]
    markets_arg = ",".join(markets)

    cmd = [
        "python3",
        "-m",
        "src.main",
        "scrape_historic",
        "--sport",
        "basketball",
        "--season",
        cfg.season,
        "--match_links",
        *match_links,
        "--markets",
        markets_arg,
        "--target_bookmaker",
        cfg.target_bookmaker,
        "--scrape_odds_history",
        "--format",
        "json",
        "--file_path",
        str(tmp_json),
    ]

    if cfg.headless:
        cmd.append("--headless")

    started = _utcnow().isoformat()
    _run(cmd, cwd=cfg.odds_harvester_dir)

    if not tmp_json.exists():
        raise FileNotFoundError(
            "OddsHarvester did not write output file. "
            f"Expected at: {tmp_json} (absolute path)."
        )

    scraped = _load_json(tmp_json)
    if not isinstance(scraped, list):
        raise TypeError(f"Expected OddsHarvester JSON output to be a list, got: {type(scraped)}")

    deduped = _dedupe_matches(scraped)
    filtered = _filter_bookmaker(deduped, cfg.target_bookmaker)

    # Validate presence
    all_bookies = sorted(
        {
            row.get("bookmaker_name")
            for match in scraped
            for k, v in (match or {}).items()
            if isinstance(k, str) and k.endswith("_market") and isinstance(v, list)
            for row in v
            if isinstance(row, dict) and isinstance(row.get("bookmaker_name"), str)
        }
    )
    any_rows = any(
        isinstance(v, list) and len(v) > 0
        for match in filtered
        for k, v in (match or {}).items()
        if isinstance(k, str) and k.endswith("_market")
    )
    if not any_rows:
        raise RuntimeError(
            f"No rows found for target bookmaker '{cfg.target_bookmaker}'. "
            f"Available bookmakers in this scrape: {all_bookies}"
        )

    filtered_raw_path = cfg_out_abs.with_suffix(f".{cfg.target_bookmaker}.raw.json")
    filtered_raw_path.write_text(json.dumps(filtered, indent=2))

    priors = [_extract_pregame_priors_from_match(m) for m in filtered]
    priors_summary = {
        "matches_with_total_line": sum(1 for p in priors if p.get("total_line") is not None),
        "matches_with_spread_line": sum(1 for p in priors if p.get("home_spread_line") is not None),
    }

    summary = _summarize(filtered)
    payload = {
        "started_utc": started,
        "finished_utc": _utcnow().isoformat(),
        "odds_harvester_dir": str(cfg.odds_harvester_dir),
        "season": cfg.season,
        "max_pages": cfg.max_pages,
        "summary": summary,
        "priors_summary": priors_summary,
        "filtered_raw_path": str(filtered_raw_path),
        "example_match": (
            {**filtered[0], "pregame_priors": priors[0]} if filtered else None
        ),
    }

    cfg_out_abs.parent.mkdir(parents=True, exist_ok=True)
    cfg_out_abs.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote spike report -> {cfg_out_abs}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
