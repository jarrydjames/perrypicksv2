from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass(frozen=True)
class SnapshotRow:
    game_id: str
    ts_utc: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class BetRow:
    bet_id: str
    game_id: str
    created_ts_utc: str
    bet_type: str
    side: str
    line: Optional[float]
    odds: Optional[int]
    payload: Dict[str, Any]


class SQLiteStore:
    """Tiny SQLite persistence layer.

    Streamlit Cloud note: local disk persistence is not guaranteed across redeploys.
    That’s why V2 should include Export/Import.

    Keep it boring and stdlib-only.
    """

    def __init__(self, db_path: str = "data/perrypicks.sqlite"):
        self.db_path = db_path
        _ensure_parent_dir(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS games (
                  game_id TEXT PRIMARY KEY,
                  created_ts_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bets (
                  bet_id TEXT PRIMARY KEY,
                  game_id TEXT NOT NULL,
                  created_ts_utc TEXT NOT NULL,
                  bet_type TEXT NOT NULL,
                  side TEXT NOT NULL,
                  line REAL,
                  odds INTEGER,
                  payload_json TEXT NOT NULL,
                  FOREIGN KEY(game_id) REFERENCES games(game_id)
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  game_id TEXT NOT NULL,
                  ts_utc TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  UNIQUE(game_id, ts_utc),
                  FOREIGN KEY(game_id) REFERENCES games(game_id)
                );

                CREATE INDEX IF NOT EXISTS idx_snapshots_game_ts ON snapshots(game_id, ts_utc);
                CREATE INDEX IF NOT EXISTS idx_bets_game_ts ON bets(game_id, created_ts_utc);
                """
            )

    def upsert_game(self, game_id: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO games(game_id, created_ts_utc) VALUES (?, ?)",
                (game_id, _utc_now_iso()),
            )

    def add_snapshot(self, game_id: str, payload: Dict[str, Any], ts_utc: Optional[str] = None) -> None:
        self.upsert_game(game_id)
        ts_utc = ts_utc or _utc_now_iso()
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO snapshots(game_id, ts_utc, payload_json) VALUES (?, ?, ?)",
                (game_id, ts_utc, json.dumps(payload, separators=(",", ":"))),
            )

    def list_snapshots(self, game_id: str, limit: int = 500) -> List[SnapshotRow]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT game_id, ts_utc, payload_json FROM snapshots WHERE game_id=? ORDER BY ts_utc ASC LIMIT ?",
                (game_id, int(limit)),
            ).fetchall()
        out: List[SnapshotRow] = []
        for r in rows:
            out.append(SnapshotRow(game_id=r["game_id"], ts_utc=r["ts_utc"], payload=json.loads(r["payload_json"])))
        return out

    def add_bet(
        self,
        *,
        bet_id: str,
        game_id: str,
        bet_type: str,
        side: str,
        line: Optional[float],
        odds: Optional[int],
        payload: Dict[str, Any],
        created_ts_utc: Optional[str] = None,
    ) -> None:
        self.upsert_game(game_id)
        created_ts_utc = created_ts_utc or _utc_now_iso()
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO bets(
                  bet_id, game_id, created_ts_utc, bet_type, side, line, odds, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id,
                    game_id,
                    created_ts_utc,
                    bet_type,
                    side,
                    line,
                    odds,
                    json.dumps(payload, separators=(",", ":")),
                ),
            )

    def list_bets(self, game_id: str, limit: int = 200) -> List[BetRow]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT bet_id, game_id, created_ts_utc, bet_type, side, line, odds, payload_json
                FROM bets
                WHERE game_id=?
                ORDER BY created_ts_utc DESC
                LIMIT ?
                """,
                (game_id, int(limit)),
            ).fetchall()
        out: List[BetRow] = []
        for r in rows:
            out.append(
                BetRow(
                    bet_id=r["bet_id"],
                    game_id=r["game_id"],
                    created_ts_utc=r["created_ts_utc"],
                    bet_type=r["bet_type"],
                    side=r["side"],
                    line=r["line"],
                    odds=r["odds"],
                    payload=json.loads(r["payload_json"]),
                )
            )
        return out

    def export_game(self, game_id: str) -> Dict[str, Any]:
        return {
            "game_id": game_id,
            "exported_ts_utc": _utc_now_iso(),
            "bets": [b.__dict__ for b in self.list_bets(game_id)],
            "snapshots": [s.__dict__ for s in self.list_snapshots(game_id)],
        }

    def import_game(self, exported: Dict[str, Any]) -> str:
        game_id = str(exported.get("game_id") or "")
        if not game_id:
            raise ValueError("Export missing game_id")

        self.upsert_game(game_id)

        for b in exported.get("bets", []) or []:
            self.add_bet(
                bet_id=str(b.get("bet_id")),
                game_id=game_id,
                bet_type=str(b.get("bet_type")),
                side=str(b.get("side")),
                line=b.get("line"),
                odds=b.get("odds"),
                payload=b.get("payload") or {},
                created_ts_utc=b.get("created_ts_utc"),
            )

        for s in exported.get("snapshots", []) or []:
            self.add_snapshot(
                game_id=game_id,
                payload=s.get("payload") or {},
                ts_utc=s.get("ts_utc"),
            )

        return game_id
