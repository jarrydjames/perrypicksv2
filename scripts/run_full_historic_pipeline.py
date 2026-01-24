"""Run the full one-time historical pipeline.

This is the "press button, receive priors DB" script.

Steps:
1) Pull OddsPortal priors for each season (bet365.us) into JSONL.
2) Join priors to NBA game_id using your existing NBA box cache.
3) Merge all seasons into one combined game_id-keyed priors file.

This is meant to be called by a `.command` launcher.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), "\n")
    res = subprocess.run(cmd, cwd=str(cwd), check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def _merge_jsonl(inputs: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for p in inputs:
            if not p.exists():
                continue
            w.write(p.read_text(encoding="utf-8"))
            if not p.read_text(encoding="utf-8").endswith("\n"):
                w.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--odds-harvester",
        required=True,
        help="Path to local OddsHarvester repo",
    )
    ap.add_argument(
        "--box-dir",
        required=True,
        help="Path to NBA box cache dir (data/raw/box with *.json files)",
    )
    ap.add_argument(
        "--seasons",
        required=True,
        help="Comma-separated OddsPortal season strings, e.g. 2022-2023,2023-2024,2024-2025",
    )
    ap.add_argument(
        "--pages",
        type=int,
        default=60,
        help="Upper bound pages to scan per season (script stops early if no new links)",
    )
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out-dir", default="data/oddsportal")
    args = ap.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if not seasons:
        raise SystemExit("No seasons provided")

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_season_joined: list[Path] = []

    for season in seasons:
        priors_out = out_dir / f"historic_bet365_{season}.jsonl"
        joined_out = out_dir / f"priors_by_game_id_{season}.jsonl"

        # 1) Pull OddsPortal priors
        pull_cmd = [
            sys.executable,
            "scripts/oddsportal_historic_pull.py",
            "--odds-harvester",
            args.odds_harvester,
            "--season",
            season,
            "--pages",
            str(args.pages),
            "--out",
            str(priors_out),
            "--resume",
        ]
        if args.headless:
            pull_cmd.append("--headless")
        _run(pull_cmd, cwd=REPO_ROOT)

        if not priors_out.exists() or priors_out.stat().st_size == 0:
            print(f"\nWARNING: priors file is missing/empty for season {season}: {priors_out}")
            print("Skipping join for this season.\n")
            continue

        # 2) Join priors -> game_id
        join_cmd = [
            sys.executable,
            "scripts/join_oddsportal_priors_to_game_ids.py",
            "--priors",
            str(priors_out),
            "--box-dir",
            args.box_dir,
            "--out",
            str(joined_out),
        ]
        _run(join_cmd, cwd=REPO_ROOT)

        per_season_joined.append(joined_out)

    # 3) Merge
    merged_out = out_dir / "priors_by_game_id_ALL.jsonl"
    _merge_jsonl(per_season_joined, merged_out)

    print("\nAll done.")
    print("Per-season priors:")
    for s in seasons:
        print(" -", out_dir / f"historic_bet365_{s}.jsonl")
    print("Per-season joined:")
    for p in per_season_joined:
        print(" -", p)
    print("Merged:")
    print(" -", merged_out)


if __name__ == "__main__":
    main()
