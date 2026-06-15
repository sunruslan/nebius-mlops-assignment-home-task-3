"""Prompt templates for the agent nodes."""

from agent.exploration import build_db_context

GENERATE_SQL_SYSTEM = """You are an expert SQLite analyst for a text-to-SQL system.

Rules:
- Write ONE SQLite query that answers the user's question.
- Use ONLY tables and columns from the provided schema and exploration data.
- Double-quote all table and column identifiers (e.g. SELECT "col" FROM "table").
- Prefer exact literal values from the categorical profile when filtering text/enums.
- Use JOINs when the question spans multiple tables; check foreign keys in the schema.
- Do not use markdown except a single ```sql code block containing the query.
- No explanations outside the code block."""

# Placeholders: {db_context}, {question}
GENERATE_SQL_USER = """Database context:

{db_context}

Question: {question}

Return the SQL query in a ```sql block."""


VERIFY_SYSTEM = """You are a strict SQL answer verifier for a text-to-SQL agent.

Decide whether the executed query result plausibly answers the question.

Mark ok=false when ANY of these apply:
- SQL execution failed (ERROR in result).
- Zero rows returned but the question clearly expects data (counts, lists, "which", "who", "what", etc.).
- Returned columns cannot answer the question (wrong entity, wrong metric).
- Values look inconsistent with the database profile (e.g. filter on a value not in categorical lists).
- Obvious off-by-one interpretation errors when the question asks for a specific rank, superlative, or threshold.

Mark ok=true when the result reasonably answers the question, even if column names differ.

Respond with ONLY a JSON object (no markdown):
{{"ok": true|false, "issue": "<short explanation if ok is false, else empty string>"}}"""

# Placeholders: {question}, {sql}, {db_context}, {execution}
VERIFY_USER = """Question: {question}

SQL executed:
{sql}

Database profile (schema + exploration):
{db_context}

Execution result:
{execution}

JSON verdict:"""


REVISE_SYSTEM = """You are an expert SQLite analyst fixing a failed text-to-SQL attempt.

Rules:
- Produce a corrected SQLite query using the schema and exploration data.
- Address the verifier's issue directly.
- Double-quote all identifiers.
- Use categorical value lists for accurate filters.
- Return ONLY a ```sql code block with the revised query."""

# Placeholders: {db_context}, {question}, {sql}, {execution}, {issue}
REVISE_USER = """Database context:

{db_context}

Question: {question}

Previous SQL (failed verification):
{sql}

Execution result:
{execution}

Verifier issue: {issue}

Write a corrected SQL query in a ```sql block."""


def format_generate_user(
    *,
    schema: str,
    question: str,
    table_counts: str = "",
    table_samples: str = "",
    categorical_profile: str = "",
) -> str:
    db_context = build_db_context(schema, table_counts, table_samples, categorical_profile)
    return GENERATE_SQL_USER.format(db_context=db_context, question=question)


def format_verify_user(
    *,
    schema: str,
    question: str,
    sql: str,
    execution: str,
    table_counts: str = "",
    table_samples: str = "",
    categorical_profile: str = "",
) -> str:
    db_context = build_db_context(schema, table_counts, table_samples, categorical_profile)
    return VERIFY_USER.format(
        question=question,
        sql=sql,
        db_context=db_context,
        execution=execution,
    )


def format_revise_user(
    *,
    schema: str,
    question: str,
    sql: str,
    execution: str,
    issue: str,
    table_counts: str = "",
    table_samples: str = "",
    categorical_profile: str = "",
) -> str:
    db_context = build_db_context(schema, table_counts, table_samples, categorical_profile)
    return REVISE_USER.format(
        db_context=db_context,
        question=question,
        sql=sql,
        execution=execution,
        issue=issue,
    )
