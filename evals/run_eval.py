"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

AGENT_TIMEOUT_SECONDS = 300.0


def _extract_sql_snapshots(history: list[dict], fallback_sql: str = "") -> dict[int, str]:
    """Map revision index -> SQL from agent history.

    Revision 0 is the first generate_sql; revision k>0 comes from the k-th revise.
    """
    snapshots: dict[int, str] = {}
    for entry in history:
        node = entry.get("node")
        sql = entry.get("sql")
        if not sql:
            continue
        if node == "generate_sql":
            snapshots[0] = sql
        elif node == "revise":
            rev = int(entry.get("iteration", len(snapshots) + 1)) - 1
            snapshots[rev] = sql

    if not snapshots and fallback_sql:
        snapshots[0] = fallback_sql
    return snapshots


def _compare_to_gold(
    db_id: str,
    sql: str,
    gold_rows: list[tuple] | None,
    gold_ok: bool,
) -> dict:
    """Run pred SQL and compare canonicalized rows to gold."""
    if not gold_ok:
        return {
            "correct": False,
            "pred_ok": False,
            "pred_row_count": 0,
            "error": "gold SQL failed to execute",
        }
    if not sql.strip():
        return {
            "correct": False,
            "pred_ok": False,
            "pred_row_count": 0,
            "error": "empty SQL",
        }

    pred_ok, pred_rows, pred_err = run_sql(db_id, sql)
    if not pred_ok:
        return {
            "correct": False,
            "pred_ok": False,
            "pred_row_count": 0,
            "error": pred_err,
        }

    return {
        "correct": matches(gold_rows, pred_rows),
        "pred_ok": True,
        "pred_row_count": len(pred_rows or []),
        "error": None,
    }


def _correct_at_revision(result: dict, revision: int) -> bool:
    """Return correctness at revision, carrying forward past terminal_revision."""
    terminal = int(result.get("terminal_revision", 0))
    effective = min(revision, terminal)
    by_rev = {
        int(item["iteration"]): bool(item["correct"])
        for item in result.get("per_iteration", [])
    }
    return by_rev.get(effective, False)


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    qtext = question["question"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    agent: dict = {}
    agent_error: str | None = None
    try:
        with httpx.Client(timeout=AGENT_TIMEOUT_SECONDS) as client:
            response = client.post(
                agent_url,
                json={"question": qtext, "db": db_id},
            )
            response.raise_for_status()
            agent = response.json()
    except httpx.HTTPError as e:
        agent_error = f"{type(e).__name__}: {e}"

    history = agent.get("history", []) if agent else []
    snapshots = _extract_sql_snapshots(history, fallback_sql=agent.get("sql", ""))
    terminal_revision = max(snapshots) if snapshots else 0

    per_iteration: list[dict] = []
    for rev in sorted(snapshots):
        score = _compare_to_gold(db_id, snapshots[rev], gold_rows, gold_ok)
        per_iteration.append({"iteration": rev, "sql": snapshots[rev], **score})

    final_sql = snapshots.get(terminal_revision, agent.get("sql", ""))
    final_score = _compare_to_gold(db_id, final_sql, gold_rows, gold_ok)

    return {
        "question": qtext,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "agent_sql": agent.get("sql"),
        "agent_iterations": int(agent.get("iterations", 0)),
        "agent_ok": bool(agent.get("ok", False)),
        "agent_verified": bool(agent.get("verified", False)),
        "agent_error": agent_error or agent.get("error"),
        "terminal_revision": terminal_revision,
        "final_correct": bool(final_score["correct"]) if gold_ok and not agent_error else False,
        "per_iteration": per_iteration,
        "history_nodes": [entry.get("node") for entry in history],
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    count = len(results)
    if count == 0:
        return {
            "count": 0,
            "overall_pass_rate": 0.0,
            "overall_correct": 0,
            "per_iteration_pass_rate": {},
            "per_iteration_correct": {},
            "avg_agent_iterations": 0.0,
            "gold_failures": 0,
            "agent_request_failures": 0,
        }

    overall_correct = sum(1 for r in results if r.get("final_correct"))
    gold_failures = sum(1 for r in results if not r.get("gold_ok"))
    agent_request_failures = sum(1 for r in results if r.get("agent_error") and not r.get("agent_sql"))

    max_revision = 0
    for result in results:
        max_revision = max(max_revision, int(result.get("terminal_revision", 0)))
        for item in result.get("per_iteration", []):
            max_revision = max(max_revision, int(item["iteration"]))

    per_iteration_pass_rate: dict[str, float] = {}
    per_iteration_correct: dict[str, int] = {}
    for revision in range(max_revision + 1):
        correct = sum(1 for r in results if _correct_at_revision(r, revision))
        per_iteration_pass_rate[str(revision)] = correct / count
        per_iteration_correct[str(revision)] = correct

    return {
        "count": count,
        "overall_pass_rate": overall_correct / count,
        "overall_correct": overall_correct,
        "per_iteration_pass_rate": per_iteration_pass_rate,
        "per_iteration_correct": per_iteration_correct,
        "avg_agent_iterations": sum(int(r.get("agent_iterations", 0)) for r in results) / count,
        "gold_failures": gold_failures,
        "agent_request_failures": agent_request_failures,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
