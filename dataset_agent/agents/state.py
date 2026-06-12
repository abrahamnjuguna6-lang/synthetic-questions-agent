import operator
from typing import Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input (set once at graph entry) ──────────────────────────────────────
    sample_questions: list[str]
    num_questions: int
    extra_instructions: str

    # ── Topic analysis ────────────────────────────────────────────────────────
    topic: str
    subtopics: list[str]
    difficulty_level: str        # "beginner" | "intermediate" | "advanced"
    question_patterns: list[str] # e.g. ["conceptual", "application", "comparison"]

    # ── Research ──────────────────────────────────────────────────────────────
    research_queries: list[str]
    # operator.add reducer: parallel research_worker results accumulate safely
    research_results: Annotated[list[dict], operator.add]
    context_summary: str

    # ── Generation ────────────────────────────────────────────────────────────
    generated_questions: list[dict]  # [{"question": str, "type": str, "difficulty": str}]
    # Questions that survived dedup validation — carried forward into repair_questions
    accepted_questions: list[dict]
    # operator.add reducer: feedback accumulates across retry passes
    validation_feedback: Annotated[list[str], operator.add]
    retry_count: int

    # ── Output ────────────────────────────────────────────────────────────────
    final_dataset: list[dict]
    output_path: str
    error: str | None

    # ── Observability (operator.add: entries from all nodes accumulate) ───────
    # Each node appends one dict via node_trace(); used by print_run_summary().
    node_telemetry: Annotated[list[dict], operator.add]
    # Each validate_quality pass appends one scoring dict; tracks quality trend
    # across retries so future runs can identify where the loop breaks down.
    quality_history: Annotated[list[dict], operator.add]
