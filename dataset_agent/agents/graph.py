"""Construct and compile the LangGraph StateGraph.

Graph topology:
  START
    → parse_input
    → analyze_topic
    → plan_research
    → conduct_research  (conditional fan-out via Send API)
        → research_worker (×N parallel)
    → synthesize_context   (all workers converge here)
    → generate_questions
    → validate_quality     (Command-based dynamic routing)
        ↓ pass        ↓ LLM-score fail    ↓ rule fail      ↓ exhausted
    format_output  repair_questions  generate_questions  handle_error
        ↓               ↓                    ↓                ↓
       END         validate_quality    validate_quality       END

repair_questions carries forward accepted (non-duplicate) questions and
generates only the N replacements with an explicit "do not repeat" list.
validate_quality uses Command — no static edges leave it.

Node-affinity LLM routing (when a ModelRouter is supplied):
  reasoning_llm  → analyze_topic, validate_quality
  code_llm       → plan_research, generate_questions
  balanced_llm   → synthesize_context
"""

from __future__ import annotations

import functools
from typing import Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agents import nodes
from agents.state import AgentState
from config.settings import DatasetConfig, ResearchConfig


def build_graph(
    llm_or_router: "Union[BaseChatModel, object]",
    search_tool: BaseTool,
    dataset_config: DatasetConfig,
    research_config: ResearchConfig,
):
    """Return a compiled LangGraph graph ready to invoke.

    `llm_or_router` accepts either:
    - A plain BaseChatModel (single-model mode, e.g. --model CLI flag)
    - A ModelRouter instance (multi-model rotation, default on OpenRouter)

    When a ModelRouter is supplied, each node group is bound to the LLM
    that best matches its workload (reasoning vs. code vs. balanced).
    """
    # Detect ModelRouter via duck typing so we avoid a circular import
    if hasattr(llm_or_router, "reasoning_llm"):
        router = llm_or_router
        reasoning_llm: BaseChatModel = router.reasoning_llm
        code_llm: BaseChatModel = router.code_llm
        balanced_llm: BaseChatModel = router.balanced_llm
    else:
        # Single-model mode — all nodes share the same LLM
        reasoning_llm = code_llm = balanced_llm = llm_or_router

    # Bind dependencies into each node via functools.partial so node signatures
    # stay clean (state-only) from the graph's perspective.
    parse_input_node = functools.partial(nodes.parse_input, dataset_config=dataset_config)
    analyze_topic_node = functools.partial(nodes.analyze_topic, llm=reasoning_llm)
    plan_research_node = functools.partial(
        nodes.plan_research, llm=code_llm, research_config=research_config
    )
    research_worker_node = functools.partial(nodes.research_worker, search_tool=search_tool)
    synthesize_context_node = functools.partial(nodes.synthesize_context, llm=balanced_llm)
    generate_questions_node = functools.partial(
        nodes.generate_questions, llm=code_llm, dataset_config=dataset_config
    )
    repair_questions_node = functools.partial(
        nodes.repair_questions, llm=code_llm, dataset_config=dataset_config
    )
    validate_quality_node = functools.partial(
        nodes.validate_quality, llm=reasoning_llm, dataset_config=dataset_config
    )
    format_output_node = functools.partial(nodes.format_output, dataset_config=dataset_config)
    handle_error_node = functools.partial(nodes.handle_error, dataset_config=dataset_config)

    # Transient errors (network blips, provider overload) are retried at the
    # LangGraph level. The ModelRouter's with_fallbacks() handles sustained
    # rate limit exhaustion by switching models before these retries trigger.
    transient_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)

    builder = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node("parse_input", parse_input_node)
    builder.add_node("analyze_topic", analyze_topic_node, retry_policy=transient_retry)
    builder.add_node("plan_research", plan_research_node, retry_policy=transient_retry)
    # conduct_research is a conditional edge function (returns Send list), not a node
    builder.add_node("research_worker", research_worker_node, retry_policy=transient_retry)
    builder.add_node("synthesize_context", synthesize_context_node, retry_policy=transient_retry)
    # No retry policy on generate/repair — a 429 retry burns daily quota;
    # quality retries are handled at the graph level by validate_quality.
    builder.add_node("generate_questions", generate_questions_node)
    builder.add_node("repair_questions", repair_questions_node)
    builder.add_node(
        "validate_quality",
        validate_quality_node,
        retry_policy=transient_retry,
    )
    builder.add_node("format_output", format_output_node)
    builder.add_node("handle_error", handle_error_node)

    # ── Static edges ───────────────────────────────────────────────────────────
    builder.add_edge(START, "parse_input")
    builder.add_edge("parse_input", "analyze_topic")
    builder.add_edge("analyze_topic", "plan_research")

    # Fan-out: plan_research → [Send("research_worker", {query}) for each query]
    # Also allow direct route to synthesize_context when no queries are produced.
    builder.add_conditional_edges(
        "plan_research",
        nodes.conduct_research,
        ["research_worker", "synthesize_context"],
    )

    # All research_worker instances converge on synthesize_context
    builder.add_edge("research_worker", "synthesize_context")
    builder.add_edge("synthesize_context", "generate_questions")

    # validate_quality routes via Command — no static edges needed from it.
    builder.add_edge("generate_questions", "validate_quality")
    builder.add_edge("repair_questions", "validate_quality")

    builder.add_edge("format_output", END)
    builder.add_edge("handle_error", END)

    return builder.compile()
