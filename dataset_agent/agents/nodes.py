"""One function per LangGraph node.

Nodes only return partial state updates (dicts), never the full state object.
validate_quality uses Command to combine state updates with dynamic routing.

Every LLM node is wrapped with `node_trace()` which:
  - Emits structured JSON logs (debug/info/error) to stderr
  - Captures timing (duration_ms) and model metadata
  - Returns a telemetry entry appended to state.node_telemetry via operator.add

The feedback loop works at two levels:
  1. validate_quality → generate_questions (existing Command retry loop)
  2. quality_history accumulates per-retry scores so run_summary can surface
     quality trends and identify where the loop breaks down across future runs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agents.prompts import (
    CONTEXT_SYNTHESIS_PROMPT,
    QUALITY_SCORING_PROMPT,
    QUESTION_GENERATION_PROMPT,
    QUESTION_REPAIR_PROMPT,
    RESEARCH_QUERY_PROMPT,
    TOPIC_ANALYSIS_PROMPT,
)
from agents.state import AgentState
from config.settings import DatasetConfig
from output.formatters import write_dataset
from utils.logging import logger, node_trace
from validation.quality import run_all_checks

if TYPE_CHECKING:
    pass


# ── Node 1: parse_input ────────────────────────────────────────────────────────

def parse_input(state: AgentState, dataset_config: DatasetConfig) -> dict:
    """Normalise user-supplied arguments. No LLM call."""
    num = state.get("num_questions") or dataset_config.default_num_questions
    logger.info(json.dumps({
        "event": "pipeline_start",
        "node": "parse_input",
        "num_questions": num,
        "sample_questions_count": len(state.get("sample_questions", [])),
    }))
    return {
        "num_questions": num,
        "extra_instructions": state.get("extra_instructions", ""),
        "retry_count": 0,
        "validation_feedback": [],
        "accepted_questions": [],
        "research_results": [],
        "node_telemetry": [],
        "quality_history": [],
        "error": None,
    }


# ── Node 2: analyze_topic ──────────────────────────────────────────────────────

def analyze_topic(state: AgentState, llm: BaseChatModel) -> dict:
    """LLM call — extract topic, subtopics, difficulty, question patterns."""
    with node_trace("analyze_topic") as entry:
        sample_qs = _format_sample_questions(state["sample_questions"])

        chain = TOPIC_ANALYSIS_PROMPT | llm
        response = chain.invoke({"sample_questions": sample_qs})
        entry["model"] = _extract_model_name(response)

        parsed = _parse_json_response(response.content)

        if not parsed or not isinstance(parsed, dict) or not parsed.get("topic"):
            logger.warning(json.dumps({
                "event": "json_parse_fallback",
                "node": "analyze_topic",
                "reason": "primary response was not parseable JSON — retrying with simpler prompt",
            }))
            parsed = _analyze_topic_fallback(llm, sample_qs)

        topic = parsed.get("topic") or "General Knowledge"
        entry["topic"] = topic
        entry["difficulty"] = parsed.get("difficulty_level", "intermediate")

        return {
            "topic": topic,
            "subtopics": parsed.get("subtopics") or [],
            "difficulty_level": parsed.get("difficulty_level") or "intermediate",
            "question_patterns": parsed.get("question_patterns") or ["conceptual"],
            "node_telemetry": [entry],
        }


def _analyze_topic_fallback(llm: BaseChatModel, sample_qs: str) -> dict:
    """Simpler prompt for models that struggle with structured JSON output."""
    from langchain_core.messages import HumanMessage

    prompt = (
        "Look at these questions and answer with a single JSON object.\n\n"
        f"Questions:\n{sample_qs}\n\n"
        'Reply with ONLY this JSON, no other text:\n'
        '{"topic": "subject name", "subtopics": ["sub1", "sub2"], '
        '"difficulty_level": "intermediate", "question_patterns": ["conceptual"]}'
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return _parse_json_response(response.content) or {}


# ── Node 3: plan_research ──────────────────────────────────────────────────────

def plan_research(
    state: AgentState, llm: BaseChatModel, research_config: "ResearchConfig"
) -> dict:
    """Generate targeted search queries from the identified topic."""
    with node_trace("plan_research") as entry:
        chain = RESEARCH_QUERY_PROMPT | llm
        response = chain.invoke({
            "topic": state["topic"],
            "subtopics": ", ".join(state["subtopics"]),
            "difficulty_level": state["difficulty_level"],
            "max_queries": research_config.max_research_queries,
        })
        entry["model"] = _extract_model_name(response)

        parsed = _parse_json_response(response.content)
        queries: list[str] = parsed if isinstance(parsed, list) else []
        queries = queries[: research_config.max_research_queries]

        entry["queries_planned"] = len(queries)

        if not queries:
            logger.warning(json.dumps({
                "event": "no_research_queries",
                "node": "plan_research",
                "topic": state["topic"],
                "action": "routing directly to synthesize_context",
            }))

        return {"research_queries": queries, "node_telemetry": [entry]}


# ── Node 4: conduct_research (fan-out entry point) ────────────────────────────

def conduct_research(state: AgentState) -> list:
    """Fan out one Send per research query. Results accumulate via reducer.

    If plan_research produced no queries, route directly to synthesize_context
    so the graph doesn't stall with no outgoing edges.
    """
    from langgraph.types import Send

    queries = state.get("research_queries", [])
    if not queries:
        return [Send("synthesize_context", {})]
    return [Send("research_worker", {"query": q}) for q in queries]


# ── Node 4b: research_worker ───────────────────────────────────────────────────

def research_worker(state: dict, search_tool: BaseTool) -> dict:
    """Execute a single search query. Returns partial state with one result entry."""
    query = state["query"]
    snippets: list[str] = []

    with node_trace("research_worker") as entry:
        entry["query"] = query[:80]
        try:
            raw = search_tool.invoke(query)

            # ── Normalise provider response into a flat list of text snippets ──
            #
            # TavilySearch v0.2+ with include_answer and include_raw_content returns:
            #   {
            #     "answer":  "<AI-synthesised paragraph>",   ← most valuable
            #     "results": [
            #       { "title", "url", "content", "raw_content", "score" },
            #       ...
            #     ]
            #   }
            # DuckDuckGo returns a plain string.
            # Older Tavily versions / other providers may return list[dict].

            if isinstance(raw, dict):
                # Tavily structured response
                answer = raw.get("answer", "")
                if answer:
                    snippets.append(f"[Summary]\n{answer}")

                for r in raw.get("results", []):
                    if not isinstance(r, dict):
                        continue
                    title = r.get("title", "")
                    url   = r.get("url", "")
                    header = f"[{title}]({url})\n" if title else ""
                    # raw_content (full page markdown) is far richer than content snippet
                    text = r.get("raw_content") or r.get("content") or ""
                    if text:
                        snippets.append(f"{header}{str(text)[:2500]}")

            elif isinstance(raw, str):
                snippets = [raw[:4000]]

            elif isinstance(raw, list):
                for r in raw[:5]:
                    if isinstance(r, dict):
                        text = r.get("raw_content") or r.get("content") or str(r)
                        snippets.append(str(text)[:2500])
                    else:
                        snippets.append(str(r)[:1000])

            entry["snippets_retrieved"] = len(snippets)
            entry["answer_included"] = snippets[0].startswith("[Summary]") if snippets else False
        except Exception as exc:
            entry["search_error"] = str(exc)[:200]
            logger.warning(json.dumps({
                "event": "search_failed",
                "node": "research_worker",
                "query": query[:80],
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            }))
            snippets = [f"[Search failed: {type(exc).__name__}: {str(exc)[:100]}]"]

        return {
            "research_results": [{"query": query, "snippets": snippets}],
            "node_telemetry": [entry],
        }


# ── Node 5: synthesize_context ─────────────────────────────────────────────────

def synthesize_context(state: AgentState, llm: BaseChatModel) -> dict:
    """Collapse all research results into a concise context summary."""
    with node_trace("synthesize_context") as entry:
        if not state.get("research_results"):
            logger.warning(json.dumps({
                "event": "no_research_results",
                "node": "synthesize_context",
                "action": "using empty context — topic analysis only",
            }))
            return {
                "context_summary": "No research context available.",
                "node_telemetry": [entry],
            }

        formatted = "\n\n".join(
            f"Query: {r['query']}\nSnippets:\n" + "\n".join(r.get("snippets", []))
            for r in state["research_results"]
        )
        chain = CONTEXT_SYNTHESIS_PROMPT | llm
        response = chain.invoke({"topic": state["topic"], "research_results": formatted})
        entry["model"] = _extract_model_name(response)
        entry["context_length"] = len(response.content)

        return {
            "context_summary": response.content.strip(),
            "node_telemetry": [entry],
        }


# ── Node 6: generate_questions ─────────────────────────────────────────────────

def generate_questions(
    state: AgentState, llm: BaseChatModel, dataset_config: "DatasetConfig"
) -> dict:
    """Generate questions in small batches to stay within model output limits.

    Batching keeps each LLM response short, avoiding truncation on models with
    low output-token caps (e.g. free-tier reasoning models).
    """
    with node_trace("generate_questions") as entry:
        feedback = state.get("validation_feedback", [])
        feedback_section = (
            "IMPORTANT — previous attempt was rejected. Fix these issues:\n"
            + "\n".join(f"- {fb}" for fb in feedback)
            + "\n\n"
            if feedback
            else ""
        )

        extra = state.get("extra_instructions", "").strip()
        extra_section = f"Additional instructions: {extra}\n" if extra else ""

        total = state["num_questions"]
        batch_size = dataset_config.generation_batch_size
        all_questions: list[dict] = []
        truncated_batches = 0

        chain = QUESTION_GENERATION_PROMPT | llm
        base_invoke_kwargs = {
            "topic": state["topic"],
            "subtopics": ", ".join(state["subtopics"]),
            "difficulty_level": state["difficulty_level"],
            "question_patterns": ", ".join(state["question_patterns"]),
            "context_summary": state.get("context_summary", ""),
            "sample_questions": _format_sample_questions(state["sample_questions"]),
            "feedback_section": feedback_section,
            "extra_instructions_section": extra_section,
        }

        remaining = total
        while remaining > 0:
            batch_n = min(batch_size, remaining)
            response = chain.invoke({**base_invoke_kwargs, "num_questions": batch_n})

            # Anthropic uses stop_reason="max_tokens"; OpenAI uses finish_reason="length"
            meta = response.response_metadata or {}
            finish = meta.get("finish_reason") or meta.get("stop_reason", "unknown")
            if finish in ("length", "max_tokens"):
                truncated_batches += 1
                logger.warning(json.dumps({
                    "event": "response_truncated",
                    "node": "generate_questions",
                    "batch_size": batch_n,
                    "stop_signal": finish,
                    "tip": "Lower DATASET_GENERATION_BATCH_SIZE or increase LLM_MAX_TOKENS",
                }))

            parsed = _parse_json_response(response.content)
            batch_qs: list[dict] = (
                parsed.get("questions", []) if isinstance(parsed, dict) else []
            )
            all_questions.extend(batch_qs)
            remaining -= batch_n

        entry["model"] = _extract_model_name(response)  # last batch model
        entry["questions_generated"] = len(all_questions)
        entry["retry_pass"] = state.get("retry_count", 0)
        if truncated_batches:
            entry["truncated_batches"] = truncated_batches

        return {"generated_questions": all_questions, "node_telemetry": [entry]}


# ── Node 7: validate_quality ───────────────────────────────────────────────────

def validate_quality(
    state: AgentState,
    llm: BaseChatModel,
    dataset_config: DatasetConfig,
) -> Command[Literal["generate_questions", "repair_questions", "format_output", "handle_error"]]:
    """Two-phase validation. Returns Command to route + update state atomically.

    Phase 1 — rule-based checks (fast, no LLM):
      count_check, format_check, uniqueness_check (TF-IDF cosine similarity)

    Phase 2 — LLM scoring (relevance, diversity, clarity, difficulty_alignment):
      Runs only if Phase 1 passes.

    Each pass appends a quality_history entry so the run summary can show
    score progression across retries, making it easy to spot improvement gaps.
    """
    with node_trace("validate_quality") as entry:
        questions = state.get("generated_questions", [])
        retry_count = state.get("retry_count", 0)

        entry["retry"] = retry_count
        entry["questions_to_validate"] = len(questions)

        # Phase 1 — rule-based (fast)
        rule_result = run_all_checks(
            questions=questions,
            expected_count=state["num_questions"],
            topic=state["topic"],
            subtopics=state["subtopics"],
        )
        entry["rule_passed"] = rule_result.passed

        if not rule_result.passed:
            logger.warning(json.dumps({
                "event": "rule_validation_failed",
                "node": "validate_quality",
                "retry": retry_count,
                "issues": rule_result.issues[:5],
            }))
            quality_entry = _make_quality_entry(
                retry=retry_count,
                rule_passed=False,
                llm_score=None,
                issues=rule_result.issues,
            )
            # Structural failure (count/format) — regenerate from scratch
            return _handle_validation_failure(
                issues=rule_result.issues,
                retry_count=retry_count,
                max_retries=dataset_config.max_retries,
                quality_entry=quality_entry,
                telemetry_entry=entry,
                accepted_questions=None,
            )

        # Phase 2 — LLM scoring
        chain = QUALITY_SCORING_PROMPT | llm
        response = chain.invoke({
            "topic": state["topic"],
            "subtopics": ", ".join(state["subtopics"]),
            "difficulty_level": state["difficulty_level"],
            "generated_questions": json.dumps(questions, indent=2),
        })
        entry["model"] = _extract_model_name(response)

        score_data = _parse_json_response(response.content)
        overall_score: float = score_data.get("overall_score", 0.0) if isinstance(score_data, dict) else 0.0
        issues: list[str] = score_data.get("issues", []) if isinstance(score_data, dict) else []

        entry["llm_score"] = overall_score
        entry["score_dimensions"] = {
            k: score_data.get(k)
            for k in ("relevance", "diversity", "clarity", "difficulty_alignment")
            if isinstance(score_data, dict) and k in score_data
        }

        quality_entry = _make_quality_entry(
            retry=retry_count,
            rule_passed=True,
            llm_score=overall_score,
            issues=issues,
        )

        if overall_score >= dataset_config.quality_threshold:
            logger.info(json.dumps({
                "event": "quality_passed",
                "node": "validate_quality",
                "score": overall_score,
                "threshold": dataset_config.quality_threshold,
                "retry": retry_count,
            }))
            return Command(
                update={
                    "final_dataset": questions,
                    "quality_history": [quality_entry],
                    "node_telemetry": [entry],
                },
                goto="format_output",
            )

        # Identify duplicate questions to drop; keep the rest for surgical repair
        duplicate_indices: set[int] = set(
            score_data.get("duplicate_indices", [])
            if isinstance(score_data, dict) else []
        )
        accepted_qs = [
            q for i, q in enumerate(questions)
            if i not in duplicate_indices
        ]

        summary = (
            f"LLM quality score {overall_score:.2f} < threshold "
            f"{dataset_config.quality_threshold:.2f}. Issues: {'; '.join(issues)}"
        )
        logger.warning(json.dumps({
            "event": "llm_score_below_threshold",
            "node": "validate_quality",
            "score": overall_score,
            "threshold": dataset_config.quality_threshold,
            "retry": retry_count,
            "duplicate_count": len(duplicate_indices),
            "accepted_count": len(accepted_qs),
            "repair_needed": len(questions) - len(accepted_qs),
            "issues": issues[:3],
        }))
        # Duplicate/quality failure — keep good questions, repair only the bad ones
        return _handle_validation_failure(
            issues=[summary],
            retry_count=retry_count,
            max_retries=dataset_config.max_retries,
            quality_entry=quality_entry,
            telemetry_entry=entry,
            accepted_questions=accepted_qs,
        )


def _make_quality_entry(
    retry: int,
    rule_passed: bool,
    llm_score: float | None,
    issues: list[str],
) -> dict:
    return {
        "retry": retry,
        "rule_passed": rule_passed,
        "llm_score": llm_score,
        "issues": issues,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _handle_validation_failure(
    issues: list[str],
    retry_count: int,
    max_retries: int,
    quality_entry: dict,
    telemetry_entry: dict,
    accepted_questions: list[dict] | None = None,
) -> Command:
    if retry_count < max_retries:
        if accepted_questions is not None:
            # Surgical repair: carry forward good questions, regenerate only duplicates
            return Command(
                update={
                    "retry_count": retry_count + 1,
                    "validation_feedback": issues,
                    "accepted_questions": accepted_questions,
                    "quality_history": [quality_entry],
                    "node_telemetry": [telemetry_entry],
                },
                goto="repair_questions",
            )
        # Structural failure: regenerate everything from scratch
        return Command(
            update={
                "retry_count": retry_count + 1,
                "validation_feedback": issues,
                "accepted_questions": [],
                "quality_history": [quality_entry],
                "node_telemetry": [telemetry_entry],
            },
            goto="generate_questions",
        )

    error_msg = (
        f"Quality threshold not met after {max_retries} retries. "
        f"Last issues: {'; '.join(issues[:3])}"
    )
    logger.error(json.dumps({
        "event": "max_retries_exhausted",
        "node": "validate_quality",
        "max_retries": max_retries,
        "last_issues": issues,
    }))
    return Command(
        update={
            "error": error_msg,
            "quality_history": [quality_entry],
            "node_telemetry": [telemetry_entry],
        },
        goto="handle_error",
    )


# ── Node 7b: repair_questions ─────────────────────────────────────────────────

def repair_questions(
    state: AgentState, llm: BaseChatModel, dataset_config: "DatasetConfig"
) -> dict:
    """Generate only the questions needed to replace duplicates.

    Carries forward accepted_questions (validated non-duplicates from the previous
    validate_quality pass) and generates repair_count new questions whose scenarios
    are explicitly forbidden from repeating any accepted scenario.
    """
    with node_trace("repair_questions") as entry:
        accepted = state.get("accepted_questions", [])
        total = state["num_questions"]
        repair_count = max(0, total - len(accepted))

        entry["accepted_carried_forward"] = len(accepted)
        entry["repair_count"] = repair_count
        entry["retry_pass"] = state.get("retry_count", 0)

        if repair_count == 0:
            # Nothing to repair — pass accepted through to validate_quality
            entry["questions_generated"] = len(accepted)
            return {"generated_questions": accepted, "node_telemetry": [entry]}

        # Build a concise "do not repeat" reference from accepted question stems
        accepted_text = "\n".join(
            f"{i + 1}. {q.get('question', '')[:150]}"
            for i, q in enumerate(accepted)
        ) or "None accepted yet."

        # Use only the most recent validation issues to keep the prompt tight
        feedback = state.get("validation_feedback", [])
        recent_issues = feedback[-2:] if feedback else []
        feedback_section = (
            "PREVIOUS ISSUES TO ADDRESS:\n"
            + "\n".join(f"- {fb[:250]}" for fb in recent_issues)
            + "\n\n"
            if recent_issues else ""
        )

        chain = QUESTION_REPAIR_PROMPT | llm
        all_new_qs: list[dict] = []
        last_response = None

        batch_size = dataset_config.generation_batch_size
        remaining = repair_count
        while remaining > 0:
            batch_n = min(batch_size, remaining)
            last_response = chain.invoke({
                "topic": state["topic"],
                "num_questions": total,
                "accepted_count": len(accepted),
                "repair_count": batch_n,
                "accepted_questions_text": accepted_text,
                "subtopics": ", ".join(state["subtopics"]),
                "difficulty_level": state["difficulty_level"],
                "context_summary": state.get("context_summary", ""),
                "feedback_section": feedback_section,
            })
            parsed = _parse_json_response(last_response.content)
            batch_qs = parsed.get("questions", []) if isinstance(parsed, dict) else []
            all_new_qs.extend(batch_qs)
            remaining -= batch_n

        if last_response:
            entry["model"] = _extract_model_name(last_response)

        combined = accepted + all_new_qs
        entry["questions_generated"] = len(combined)

        return {"generated_questions": combined, "node_telemetry": [entry]}


# ── Node 8: format_output ──────────────────────────────────────────────────────

def format_output(
    state: AgentState, dataset_config: DatasetConfig
) -> dict:
    """Write the final dataset to disk. No LLM call."""
    with node_trace("format_output") as entry:
        path = write_dataset(
            questions=state["final_dataset"],
            output_dir=dataset_config.output_dir,
            output_format=dataset_config.output_format,
            topic=state["topic"],
        )
        entry["output_path"] = path
        entry["questions_written"] = len(state["final_dataset"])

        return {"output_path": path, "node_telemetry": [entry]}


# ── Node 9: handle_error ───────────────────────────────────────────────────────

def handle_error(state: AgentState, dataset_config: DatasetConfig) -> dict:
    """Log the error and write partial results if generation produced anything."""
    with node_trace("handle_error") as entry:
        error_msg = state.get("error", "Unknown error")
        entry["error_message"] = error_msg

        logger.error(json.dumps({
            "event": "pipeline_error",
            "node": "handle_error",
            "error": error_msg,
            "retry_count": state.get("retry_count", 0),
        }))

        partial = state.get("generated_questions", [])
        if partial:
            try:
                path = write_dataset(
                    questions=partial,
                    output_dir=dataset_config.output_dir,
                    output_format=dataset_config.output_format,
                    topic=state.get("topic", "unknown"),
                )
                entry["partial_output_path"] = path
                logger.info(json.dumps({
                    "event": "partial_results_saved",
                    "node": "handle_error",
                    "path": path,
                    "questions_saved": len(partial),
                }))
                return {"output_path": path, "node_telemetry": [entry]}
            except Exception as exc:
                logger.error(json.dumps({
                    "event": "partial_save_failed",
                    "node": "handle_error",
                    "error": str(exc)[:200],
                }))

        return {"node_telemetry": [entry]}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _extract_model_name(response) -> str:
    """Extract model ID from OpenRouter (or other provider) response metadata.

    OpenRouter echoes the model name back in response_metadata. Different
    providers use slightly different keys, so we try the most common ones.
    """
    meta = getattr(response, "response_metadata", {}) or {}
    return (
        meta.get("model_name")
        or meta.get("model")
        or "unknown"
    )


def _parse_json_response(text: str) -> dict | list:
    """Extract and parse the first valid JSON object or array in an LLM response.

    Handles:
    - Chain-of-thought / reasoning models that emit <think>...</think> blocks
      before their actual answer (e.g. nemotron, deepseek-r1)
    - Markdown code fences (```json ... ```)
    - JSON embedded in surrounding prose

    Uses json.JSONDecoder.raw_decode so it finds the first *complete* JSON
    value without requiring the rest of the string to be empty.
    """
    # 1. Strip chain-of-thought reasoning blocks
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*", "", clean).replace("```", "").strip()

    # 3. Fast path — the whole cleaned string is valid JSON
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 4. Scan for the first { or [ that begins a complete, valid JSON value.
    #    raw_decode(s, idx) parses from position idx and returns (obj, end_idx),
    #    tolerating arbitrary trailing text — exactly what we need.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(clean):
        if ch not in ("{", "["):
            continue
        try:
            obj, _ = decoder.raw_decode(clean, i)
            return obj
        except json.JSONDecodeError:
            continue

    return {}


def _format_sample_questions(questions: list[str]) -> str:
    """Format sample questions for inclusion in prompts.

    Multi-line MCQ blocks are preserved as-is and separated by blank lines.
    Single-line questions are numbered.
    """
    parts = []
    for i, q in enumerate(questions, start=1):
        if "\n" in q.strip():
            parts.append(f"[{i}]\n{q.strip()}")
        else:
            parts.append(f"{i}. {q.strip()}")
    return "\n\n".join(parts)
