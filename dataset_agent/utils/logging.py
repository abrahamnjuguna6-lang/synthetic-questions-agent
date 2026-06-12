"""Structured logging and node tracing for the dataset agent.

Usage in every LLM node:

    from utils.logging import logger, node_trace

    def my_node(state: AgentState, llm: BaseChatModel) -> dict:
        with node_trace("my_node") as entry:
            response = (PROMPT | llm).invoke(...)
            entry["model"] = _extract_model_name(response)
            entry["tokens"] = response.response_metadata.get("token_usage", {})
            return {"some_field": result, "node_telemetry": [entry]}

The `entry` dict is mutated by the context manager on exit — status, duration_ms,
and (on error) error_type/error are added automatically. Because Python list
references survive return-before-__exit__, the fully-populated entry is always
included in the returned state update.

Call `print_run_summary(final_state)` in main.py after graph.invoke() to emit
a human-readable performance and quality table.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator


# ── Logger setup ───────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line to stderr.

    If the message is already a JSON string, it is merged into the record;
    otherwise it is placed under the "message" key.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        try:
            body: dict = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            body = {"message": record.getMessage()}

        body["level"] = record.levelname
        body["ts"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        if record.exc_info and record.levelno >= logging.ERROR:
            body["traceback"] = traceback.format_exception(*record.exc_info)

        return json.dumps(body, default=str)


def _build_logger() -> logging.Logger:
    log = logging.getLogger("dataset_agent")
    if log.handlers:
        return log  # idempotent — safe to import multiple times

    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    log.addHandler(handler)
    log.propagate = False
    return log


logger = _build_logger()


# ── Node tracing context manager ───────────────────────────────────────────────

@contextmanager
def node_trace(node_name: str) -> Generator[dict, None, None]:
    """Time a node, log start/success/error, and yield a mutable telemetry entry.

    Callers annotate `entry` freely inside the `with` block
    (e.g. ``entry["model"] = "deepseek/..."``).  On context exit, `status` and
    `duration_ms` are added; on error, `error_type` and `error` are also added.

    Because Python evaluates the return expression *before* calling __exit__,
    the `[entry]` list in the return value holds a reference to the same dict
    object that __exit__ will mutate — so the fully-populated entry is always
    present in the state update that LangGraph receives.

    Example::

        with node_trace("synthesize_context") as entry:
            response = chain.invoke(...)
            entry["model"] = _extract_model_name(response)
            return {"context_summary": ..., "node_telemetry": [entry]}
    """
    entry: dict = {
        "node": node_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _t0 = time.perf_counter()

    logger.debug(json.dumps({"event": "node_start", "node": node_name}))

    try:
        yield entry

        entry["status"] = "ok"
        entry["duration_ms"] = round((time.perf_counter() - _t0) * 1000, 1)

        # Emit all annotated fields alongside the core event fields
        extras = {
            k: v for k, v in entry.items()
            if k not in {"node", "started_at", "status", "duration_ms"}
        }
        logger.info(json.dumps({
            "event": "node_done",
            "node": node_name,
            "duration_ms": entry["duration_ms"],
            **extras,
        }))

    except Exception as exc:
        entry["status"] = "error"
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)[:500]
        entry["duration_ms"] = round((time.perf_counter() - _t0) * 1000, 1)

        logger.error(
            json.dumps({
                "event": "node_error",
                "node": node_name,
                "error_type": entry["error_type"],
                "error": entry["error"],
                "duration_ms": entry["duration_ms"],
            }),
            exc_info=True,
        )
        raise


# ── Run summary ────────────────────────────────────────────────────────────────

def print_run_summary(state: dict) -> None:
    """Print a structured run summary from the final graph state.

    Reads node_telemetry, quality_history, research_results, and
    validation_feedback to surface timing, quality progression, and
    gap areas for future improvement.
    """
    telemetry: list[dict] = state.get("node_telemetry", [])
    quality_history: list[dict] = state.get("quality_history", [])
    research_results: list[dict] = state.get("research_results", [])
    validation_feedback: list[str] = state.get("validation_feedback", [])

    sep = "═" * 62
    print(f"\n{sep}")
    print("  DATASET AGENT — RUN SUMMARY")
    print(sep)

    # ── Node performance table ─────────────────────────────────────────────────
    if telemetry:
        print("\n  NODE PERFORMANCE")
        print(f"  {'Node':<26} {'Status':<8} {'ms':>8}  Notes")
        print("  " + "─" * 58)
        for t in telemetry:
            node = t.get("node", "?")
            status = t.get("status", "?")
            ms = t.get("duration_ms", 0)
            dur = f"{ms:,.0f}" if ms < 60_000 else f"{ms / 1000:.1f}s"
            icon = "✓" if status == "ok" else "✗"
            extras = {
                k: v for k, v in t.items()
                if k not in {"node", "started_at", "status", "duration_ms"}
            }
            notes_parts = []
            if "model" in extras:
                notes_parts.append(f"model={extras['model']}")
            if "questions_generated" in extras:
                notes_parts.append(f"questions={extras['questions_generated']}")
            if "topic" in extras:
                notes_parts.append(f"topic={extras['topic']}")
            if "error_type" in extras:
                notes_parts.append(f"error={extras['error_type']}: {extras.get('error','')[:40]}")
            notes = "  ".join(notes_parts)
            print(f"  {icon} {node:<25} {status:<8} {dur:>8}  {notes}")

    # ── Quality validation history ─────────────────────────────────────────────
    if quality_history:
        print("\n  QUALITY VALIDATION HISTORY")
        print(f"  {'Retry':<7} {'Rule':<8} {'LLM Score':<12} {'Issues'}")
        print("  " + "─" * 58)
        for q in quality_history:
            retry = q.get("retry", 0)
            rule = "PASS" if q.get("rule_passed") else "FAIL"
            score = q.get("llm_score")
            score_str = f"{score:.2f}" if score is not None else "n/a "
            issues = q.get("issues", [])
            first_issue = (issues[0][:44] + "…") if issues and len(issues[0]) > 44 else (issues[0] if issues else "—")
            print(f"  {retry:<7} {rule:<8} {score_str:<12} {first_issue}")

        if len(quality_history) > 1:
            scores = [q["llm_score"] for q in quality_history if q.get("llm_score") is not None]
            if len(scores) >= 2:
                delta = scores[-1] - scores[0]
                trend = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                print(f"\n  Score trend across retries: {trend}")

    # ── Research stats ─────────────────────────────────────────────────────────
    if research_results:
        total_snippets = sum(len(r.get("snippets", [])) for r in research_results)
        empty_queries = sum(
            1 for r in research_results
            if any("[Search failed" in s for s in r.get("snippets", []))
        )
        print(f"\n  RESEARCH: {len(research_results)} queries → {total_snippets} snippets")
        if empty_queries:
            print(f"  WARNING: {empty_queries} search(es) failed — consider checking Tavily key or DuckDuckGo availability")

    # ── Accumulated validation feedback (gaps for future improvement) ──────────
    if validation_feedback:
        print(f"\n  VALIDATION FEEDBACK — {len(validation_feedback)} issue(s) logged across retries")
        print("  (These identify gaps to improve prompts or model selection)")
        for fb in validation_feedback[-6:]:
            print(f"    • {fb[:80]}")
        if len(validation_feedback) > 6:
            print(f"    … and {len(validation_feedback) - 6} more (check node_telemetry for full list)")

    print(f"\n{sep}\n")
