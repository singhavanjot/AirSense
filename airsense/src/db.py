"""SQLite helpers for AirSense."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATABASE_PATH


def _resolve_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection, creating the database directory if needed."""
    path = _resolve_path(db_path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def write_dataframe(df: pd.DataFrame, table_name: str, db_path: str | Path | None = None) -> None:
    """Write a DataFrame to a SQLite table, replacing any existing table."""
    with get_connection(db_path) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def read_dataframe(table_name: str, db_path: str | Path | None = None) -> pd.DataFrame:
    """Read a SQLite table into a DataFrame."""
    with get_connection(db_path) as connection:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        except Exception:
            return pd.DataFrame()


def save_metadata(key: str, value: Any, db_path: str | Path | None = None) -> None:
    """Persist a metadata value as JSON in SQLite."""
    payload = json.dumps(value, default=str)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, payload),
        )
        connection.commit()


def load_metadata(key: str, db_path: str | Path | None = None) -> Any:
    """Load a JSON metadata value from SQLite."""
    with get_connection(db_path) as connection:
        try:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
    if row is None:
        return None
    return json.loads(row[0])
