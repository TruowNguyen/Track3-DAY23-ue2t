"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Memory and SQLite checkpoint backends are supported.
    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "SQLite persistence requires: pip install langgraph-checkpoint-sqlite"
            ) from exc
        raw_path = database_url or "outputs/checkpoints.sqlite"
        if raw_path.startswith("sqlite:///"):
            raw_path = raw_path.removeprefix("sqlite:///")
        db_path = Path(raw_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise ValueError(
            "Postgres is an optional deployment extension; use memory/sqlite for this lab build"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
