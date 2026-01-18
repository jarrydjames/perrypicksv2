from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import streamlit as st


@dataclass(frozen=True)
class LogFileInfo:
    path: str
    size_bytes: int
    mtime_epoch: float

    @property
    def mtime_local_str(self) -> str:
        try:
            return datetime.fromtimestamp(self.mtime_epoch).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "unknown"


def _latest_log(glob_pattern: str) -> Optional[LogFileInfo]:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        return None

    best_path = max(paths, key=lambda p: os.path.getmtime(p))
    try:
        stt = os.stat(best_path)
        return LogFileInfo(path=best_path, size_bytes=int(stt.st_size), mtime_epoch=float(stt.st_mtime))
    except Exception:
        return None


def _tail_text(path: str, max_lines: int) -> str:
    # Simple + safe (logs aren't huge). If this grows, implement a true tail.
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-int(max_lines) :])
    except Exception as e:
        return f"[error reading log] {e!r}"


def render_log_monitor(*, logs_dir: str = "logs", max_lines_default: int = 200, key_prefix: str = "logmon") -> None:
    """Tiny Streamlit UI to monitor the latest overnight backtest log.

    `key_prefix` is required if you render this component multiple times on the same page.
    """

    st.subheader("Overnight backtest log")

    c1, c2, c3 = st.columns([1.2, 1.0, 0.8])
    with c1:
        pattern = st.text_input(
            "Log file pattern",
            value=os.path.join(logs_dir, "nested_backtest_*.log"),
            help="We will display the newest file by modified time.",
            key=f"{key_prefix}:pattern",
        )
    with c2:
        max_lines = st.number_input(
            "Lines to show",
            min_value=50,
            max_value=2000,
            value=int(max_lines_default),
            step=50,
            key=f"{key_prefix}:max_lines",
        )
    with c3:
        auto = st.toggle(
            "Auto refresh",
            value=False,
            help="Refresh this log viewer every few seconds.",
            key=f"{key_prefix}:auto",
        )

    if auto:
        # No dependency on streamlit-autorefresh; Streamlit reruns on widget changes.
        # But we can force a rerun using a tiny sleep-less hack: st.experimental_rerun is too aggressive.
        # Instead: use a countdown progress with st.empty().
        import time

        placeholder = st.empty()
        for i in range(5, 0, -1):
            placeholder.caption(f"Refreshing in {i}s...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

    info = _latest_log(pattern)
    if info is None:
        st.info("No matching logs found yet. Start the overnight script to generate one.")
        return

    st.write(f"**Log:** `{info.path}`")
    st.caption(f"Last modified: {info.mtime_local_str}  ·  Size: {info.size_bytes/1024:.1f} KB")

    txt = _tail_text(info.path, int(max_lines))
    st.code(txt or "(log is empty)", language="text")
