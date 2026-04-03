"""PostgreSQL / TimescaleDB connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.config import get_db_url
from src.utils.logger import get_logger

log = get_logger("db")

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_db_url()
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        log.info("Database engine created: %s", url.split("@")[-1])
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_sql(sql: str, params: dict | None = None) -> None:
    with get_engine().connect() as conn:
        conn.execute(text(sql), params or {})
        conn.commit()


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_columns: list[str],
    update_columns: list[str] | None = None,
) -> int:
    """Insert DataFrame rows into a table, updating on conflict.

    Uses PostgreSQL ON CONFLICT DO UPDATE for idempotent writes.
    """
    if df.empty:
        return 0

    cols = list(df.columns)
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    conflict_list = ", ".join(conflict_columns)

    if update_columns is None:
        update_columns = [c for c in cols if c not in conflict_columns]

    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)

    sql = f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_list}) DO UPDATE
        SET {update_clause}
    """

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    with get_engine().connect() as conn:
        for record in records:
            conn.execute(text(sql), record)
        conn.commit()

    log.info("Upserted %d rows into %s", len(records), table)
    return len(records)


def check_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.error("Database connection failed: %s", e)
        return False
