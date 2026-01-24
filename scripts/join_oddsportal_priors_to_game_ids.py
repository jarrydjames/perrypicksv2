"""Join OddsPortal priors DB to NBA game_ids.

Why
---
Your training data is keyed by NBA `game_id`.
Our OddsPortal priors DB is keyed by (match_date, home_team, away_team).

So we create a join mapping:
- input: oddsportal priors JSONL (from scripts/oddsportal_historic_pull.py)
- input: NBA box cache dir (data/raw/box/*.json) from your existing dataset
- output: JSONL keyed by game_id containing the market priors

This is intentionally simple and auditable (no fuzzy ML matching).

Usage:
  python3 scripts/join_oddsportal_priors_to_game_ids.py \
    --priors data/oddsportal/historic_bet365_2023-2024.jsonl \
    --box-dir /Users/jarrydhawley/Desktop/Predictor/perrypicks/data/raw/box \
    --out data/oddsportal/priors_by_game_id_2023-2024.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_iso_utc(s: str) -> datetime | None:
    # supports: "2024-06-18 00:30:00 UTC" and nba "2023-11-03T23:00:00Z"
    try:
        if s.endswith(" UTC"):
            s2 = s.replace(" UTC", "")
            return datetime.strptime(s2, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


_RX_NONWORD = re.compile(r"[^a-z0-9]+")


def _norm_team(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("la ", "los angeles ")
    s = s.replace("l.a.", "los angeles")
    s = s.replace("ny ", "new york ")
    s = s.replace("n.y.", "new york")
    s = _RX_NONWORD.sub(" ", s)
    s = " ".join(s.split())

    # normalize common variants
    s = s.replace("trail blazers", "blazers")
    s = s.replace("76ers", "sixers")

    return s


@dataclass(frozen=True)
class PriorsRow:
    match_date_utc: datetime
    home: str
    away: str
    payload: dict


def _load_priors_jsonl(path: Path) -> list[PriorsRow]:
    rows: list[PriorsRow] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            dt = _parse_iso_utc(str(obj.get("match_date") or ""))
            if not dt:
                continue
            rows.append(
                PriorsRow(
                    match_date_utc=dt,
                    home=_norm_team(str(obj.get("home_team") or "")),
                    away=_norm_team(str(obj.get("away_team") or "")),
                    payload=obj,
                )
            )

    return rows


def _read_box_game_meta(box_path: Path) -> tuple[str, datetime | None, str, str]:
    obj = json.loads(box_path.read_text())
    game = obj.get("game") or {}
    gid = str(game.get("gameId") or box_path.stem)
    dt = _parse_iso_utc(str(game.get("gameTimeUTC") or ""))

    ht = game.get("homeTeam") or {}
    at = game.get("awayTeam") or {}

    # Use city + teamName to match OddsPortal.
    home = _norm_team(f"{ht.get('teamCity','')} {ht.get('teamName','')}")
    away = _norm_team(f"{at.get('teamCity','')} {at.get('teamName','')}")

    return gid, dt, home, away


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--date-tolerance-days",
        type=int,
        default=1,
        help="Allowed date drift between sources (default 1 day)",
    )
    args = ap.parse_args()

    priors_path = Path(args.priors).expanduser().resolve()
    box_dir = Path(args.box_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    priors = _load_priors_jsonl(priors_path)
    print(f"Loaded priors rows: {len(priors)}")

    # Index priors by (home, away), keep datetimes so we can choose the closest match.
    by_teams: dict[tuple[str, str], list[PriorsRow]] = {}
    for r in priors:
        by_teams.setdefault((r.home, r.away), []).append(r)

    for k in by_teams:
        by_teams[k].sort(key=lambda r: r.match_date_utc)

    tol = timedelta(days=int(args.date_tolerance_days))

    matched_games = 0
    written = 0
    missing_games = 0
    used_priors_links: set[str] = set()
    reused_priors_attempts = 0

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for box_path in sorted(box_dir.glob("*.json")):
            gid, dt, home, away = _read_box_game_meta(box_path)
            if not dt:
                continue

            candidates = by_teams.get((home, away), [])
            if not candidates:
                missing_games += 1
                continue

            # Choose the closest priors row in time within tolerance.
            scored: list[tuple[float, PriorsRow]] = []
            for r in candidates:
                delta = abs((dt - r.match_date_utc).total_seconds())
                if delta <= tol.total_seconds():
                    scored.append((delta, r))

            scored.sort(key=lambda t: t[0])

            found: PriorsRow | None = None
            for _, r in scored:
                link = str(r.payload.get("match_link") or "")
                if link and link in used_priors_links:
                    reused_priors_attempts += 1
                    continue
                found = r
                if link:
                    used_priors_links.add(link)
                break

            if not found:
                missing_games += 1
                continue

            matched_games += 1

            payload = {
                "game_id": gid,
                "game_time_utc": dt.isoformat(),
                "home": home,
                "away": away,
                "bookmaker": found.payload.get("bookmaker"),
                "total_line": found.payload.get("total_line"),
                "home_spread_line": found.payload.get("home_spread_line"),
                "home_team_total_line_implied": found.payload.get("home_team_total_line_implied"),
                "away_team_total_line_implied": found.payload.get("away_team_total_line_implied"),
                "source_match_date": found.payload.get("match_date"),
                "source_match_link": found.payload.get("match_link"),
            }
            out.write(json.dumps(payload) + "\n")
            written += 1

    print(f"Matched games: {matched_games}")
    print(f"Missing games: {missing_games}")
    print(f"Unique priors rows used: {len(used_priors_links)}")
    print(f"Skipped reuse attempts: {reused_priors_attempts}")
    print(f"Wrote: {written} -> {out_path}")


if __name__ == "__main__":
    main()
