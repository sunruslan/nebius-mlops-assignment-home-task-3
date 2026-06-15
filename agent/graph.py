"""LangGraph agent: text-to-SQL with DB exploration and verify+revise loop.

Graph shape:

    START -> attach_schema -> explore_counts -> explore_samples
          -> explore_categoricals -> generate_sql -> execute -> verify
                                                                  |
                                                      ok=true ----+----> END
                                                                  |
                                                      ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.exploration import (
    explore_categorical_values,
    explore_table_counts,
    explore_table_samples,
)
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
MAX_ITERATIONS = 4

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph."""

    question: str
    db_id: str
    schema: str = ""
    table_counts: str = ""
    table_samples: str = ""
    categorical_profile: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Parsing helpers --------------------------------------------------

def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    # Fallback: strip common prefixes and take the first statement-looking block.
    cleaned = re.sub(r"^(?:here(?:'s| is).*?:\s*)", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.strip().rstrip(";")


def _parse_verify_response(text: str) -> tuple[bool, str]:
    """Parse {"ok": bool, "issue": str} from model output."""
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    # Try full JSON parse first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "ok" in obj:
            return bool(obj["ok"]), str(obj.get("issue", ""))
    except json.JSONDecodeError:
        pass

    # Extract first {...} blob.
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "ok" in obj:
                return bool(obj["ok"]), str(obj.get("issue", ""))
        except json.JSONDecodeError:
            pass

    # Heuristic fallback.
    lower = raw.lower()
    if '"ok": true' in lower or '"ok":true' in lower or re.search(r"\bok\b\s*:\s*true", lower):
        return True, ""
    return False, raw[:500] or "verifier returned unparseable response"


def _history_entry(node: str, **fields: Any) -> dict[str, Any]:
    """Build a history record, truncating long text fields."""
    entry: dict[str, Any] = {"node": node}
    for key, value in fields.items():
        if isinstance(value, str) and len(value) > 800:
            entry[key] = value[:800] + "…"
        else:
            entry[key] = value
    return entry


# ---- Nodes ------------------------------------------------------------

def attach_schema_node(state: AgentState) -> dict:
    """Render CREATE TABLE DDL from sqlite metadata."""
    schema = render_schema(state.db_id)
    return {
        "schema": schema,
        "history": state.history + [_history_entry("attach_schema", tables=len(schema.split("CREATE TABLE")) - 1)],
    }


def explore_counts_node(state: AgentState) -> dict:
    """Count rows in every table."""
    counts = explore_table_counts(state.db_id)
    return {
        "table_counts": counts,
        "history": state.history + [_history_entry("explore_counts", preview=counts)],
    }


def explore_samples_node(state: AgentState) -> dict:
    """Sample a few rows per table."""
    samples = explore_table_samples(state.db_id)
    return {
        "table_samples": samples,
        "history": state.history + [_history_entry("explore_samples", preview=samples)],
    }


def explore_categoricals_node(state: AgentState) -> dict:
    """List distinct values for low-cardinality columns."""
    profile = explore_categorical_values(state.db_id)
    return {
        "categorical_profile": profile,
        "history": state.history + [_history_entry("explore_categoricals", preview=profile)],
    }


def generate_sql_node(state: AgentState) -> dict:
    """First LLM call: question + full DB profile -> SQL."""
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.format_generate_user(
            schema=state.schema,
            table_counts=state.table_counts,
            table_samples=state.table_samples,
            categorical_profile=state.categorical_profile,
            question=state.question,
        )),
    ])
    sql = _extract_sql(str(response.content))
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [_history_entry("generate_sql", sql=sql, iteration=state.iteration + 1)],
    }


def execute_node(state: AgentState) -> dict:
    """Run the SQL and store the result."""
    execution = execute_sql(state.db_id, state.sql)
    return {
        "execution": execution,
        "history": state.history + [_history_entry(
            "execute",
            ok=execution.ok,
            row_count=execution.row_count,
            error=execution.error,
        )],
    }


def verify_node(state: AgentState) -> dict:
    """Second LLM call: check whether results plausibly answer the question."""
    execution = state.execution
    if execution is None:
        return {
            "verify_ok": False,
            "verify_issue": "no execution result to verify",
            "history": state.history + [_history_entry("verify", ok=False, issue="no execution")],
        }

    # Fast path: hard SQL errors never pass verification.
    if not execution.ok:
        issue = execution.error or "SQL execution failed"
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history + [_history_entry("verify", ok=False, issue=issue, fast_path=True)],
        }

    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.format_verify_user(
            schema=state.schema,
            table_counts=state.table_counts,
            table_samples=state.table_samples,
            categorical_profile=state.categorical_profile,
            question=state.question,
            sql=state.sql,
            execution=execution.render(),
        )),
    ])
    verify_ok, verify_issue = _parse_verify_response(str(response.content))
    return {
        "verify_ok": verify_ok,
        "verify_issue": verify_issue,
        "history": state.history + [_history_entry("verify", ok=verify_ok, issue=verify_issue)],
    }


def revise_node(state: AgentState) -> dict:
    """Third LLM call (on failure): produce corrected SQL."""
    execution = state.execution
    execution_text = execution.render() if execution else "ERROR: no execution result"

    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.format_revise_user(
            schema=state.schema,
            table_counts=state.table_counts,
            table_samples=state.table_samples,
            categorical_profile=state.categorical_profile,
            question=state.question,
            sql=state.sql,
            execution=execution_text,
            issue=state.verify_issue or "result did not answer the question",
        )),
    ])
    sql = _extract_sql(str(response.content))
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [_history_entry(
            "revise",
            sql=sql,
            iteration=state.iteration + 1,
            prior_issue=state.verify_issue,
        )],
    }


def route_after_verify(state: AgentState) -> str:
    """Route to revise loop or terminate."""
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", attach_schema_node)
    g.add_node("explore_counts", explore_counts_node)
    g.add_node("explore_samples", explore_samples_node)
    g.add_node("explore_categoricals", explore_categoricals_node)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "explore_counts")
    g.add_edge("explore_counts", "explore_samples")
    g.add_edge("explore_samples", "explore_categoricals")
    g.add_edge("explore_categoricals", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
