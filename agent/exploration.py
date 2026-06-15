"""Programmatic DB exploration before SQL generation.

Runs read-only introspection queries: per-table row counts, row samples,
and distinct values for low-cardinality (categorical) columns. Output is
rendered as compact text for prompt context.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from agent.schema import db_path, quote_ident

# Columns with at most this many distinct values are treated as categorical.
MAX_CATEGORICAL_DISTINCT = 25
# Cap how many distinct values we list per column.
MAX_VALUES_SHOWN = 20
# Rows sampled per table for the model.
SAMPLE_ROWS_PER_TABLE = 3
# Skip tables larger than this when scanning categoricals (safety valve).
MAX_ROWS_FOR_CATEGORICAL_SCAN = 500_000


@lru_cache(maxsize=32)
def _list_tables(db_id: str) -> tuple[str, ...]:
    path = db_path(db_id)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    return tuple(r[0] for r in rows)


@lru_cache(maxsize=32)
def _table_columns(db_id: str, table: str) -> tuple[tuple[str, str, int], ...]:
    """Return (name, declared_type, is_pk) per column."""
    path = db_path(db_id)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        info = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return tuple((name, ctype or "TEXT", int(pk)) for _cid, name, ctype, _nn, _d, pk in info)


def _run_scalar(conn: sqlite3.Connection, sql: str) -> int | None:
    try:
        row = conn.execute(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        return None


def explore_table_counts(db_id: str) -> str:
    """Row count for every user table."""
    path = db_path(db_id)
    lines = ["-- Table row counts"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for table in _list_tables(db_id):
            count = _run_scalar(conn, f"SELECT COUNT(*) FROM {quote_ident(table)}")
            if count is None:
                lines.append(f"{table}: (count unavailable)")
            else:
                lines.append(f"{table}: {count:,} rows")
    return "\n".join(lines)


def explore_table_samples(db_id: str, limit: int = SAMPLE_ROWS_PER_TABLE) -> str:
    """A few sample rows per table so the model sees real value shapes."""
    path = db_path(db_id)
    sections: list[str] = ["-- Sample rows (up to {limit} per table)".format(limit=limit)]

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for table in _list_tables(db_id):
            cols = [c[0] for c in _table_columns(db_id, table)]
            if not cols:
                continue
            col_list = ", ".join(quote_ident(c) for c in cols)
            try:
                rows = conn.execute(
                    f"SELECT {col_list} FROM {quote_ident(table)} LIMIT {int(limit)}"
                ).fetchall()
            except sqlite3.Error as e:
                sections.append(f"\n{table}: sample failed ({e})")
                continue

            if not rows:
                sections.append(f"\n{table}: (empty)")
                continue

            sections.append(f"\n{table}:")
            sections.append("  columns: " + ", ".join(cols))
            for i, row in enumerate(rows, 1):
                rendered = ", ".join(
                    f"{cols[j]}={row[j]!r}" for j in range(len(cols))
                )
                sections.append(f"  row {i}: {rendered}")

    return "\n".join(sections)


def _is_likely_identifier(col_name: str, col_type: str, is_pk: bool) -> bool:
    """Skip high-cardinality ID-like columns when hunting categoricals."""
    lower = col_name.lower()
    if is_pk and "int" in col_type.lower():
        return True
    if lower == "id" or lower.endswith("_id"):
        return "int" in col_type.lower() or col_type.upper() in ("INTEGER", "")
    return False


def explore_categorical_values(db_id: str) -> str:
    """Distinct values for low-cardinality columns (enums, statuses, labels)."""
    path = db_path(db_id)
    lines = [
        "-- Categorical column values "
        f"(columns with ≤{MAX_CATEGORICAL_DISTINCT} distinct non-null values)"
    ]

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for table in _list_tables(db_id):
            row_count = _run_scalar(conn, f"SELECT COUNT(*) FROM {quote_ident(table)}")
            if row_count is None or row_count > MAX_ROWS_FOR_CATEGORICAL_SCAN:
                continue

            table_hits: list[str] = []
            for col_name, col_type, is_pk in _table_columns(db_id, table):
                if _is_likely_identifier(col_name, col_type, is_pk):
                    continue

                distinct = _run_scalar(
                    conn,
                    f"SELECT COUNT(DISTINCT {quote_ident(col_name)}) FROM {quote_ident(table)} "
                    f"WHERE {quote_ident(col_name)} IS NOT NULL",
                )
                if distinct is None or distinct == 0 or distinct > MAX_CATEGORICAL_DISTINCT:
                    continue

                try:
                    values = conn.execute(
                        f"SELECT DISTINCT {quote_ident(col_name)} FROM {quote_ident(table)} "
                        f"WHERE {quote_ident(col_name)} IS NOT NULL "
                        f"ORDER BY {quote_ident(col_name)} "
                        f"LIMIT {MAX_VALUES_SHOWN}"
                    ).fetchall()
                except sqlite3.Error:
                    continue

                shown = [repr(v[0]) for v in values]
                suffix = ""
                if distinct > len(shown):
                    suffix = f" … (+{distinct - len(shown)} more)"
                table_hits.append(
                    f"  {col_name} ({distinct} distinct): {', '.join(shown)}{suffix}"
                )

            if table_hits:
                lines.append(f"\n{table}:")
                lines.extend(table_hits)

    if len(lines) == 1:
        lines.append("(no low-cardinality categorical columns detected)")
    return "\n".join(lines)


def build_db_context(
    schema: str,
    table_counts: str = "",
    table_samples: str = "",
    categorical_profile: str = "",
) -> str:
    """Merge schema + exploration sections for LLM prompts."""
    parts = [schema]
    if table_counts:
        parts.append("\n" + table_counts)
    if table_samples:
        parts.append("\n" + table_samples)
    if categorical_profile:
        parts.append("\n" + categorical_profile)
    return "\n".join(parts)
