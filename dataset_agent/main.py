#!/usr/bin/env python3
"""CLI entry point for the AI Question Dataset Generator.

Usage examples:
  python main.py --questions questions.txt --num 30
  python main.py --questions "q1.txt" --num 10 --instructions "Focus on exam-style questions"
  python main.py --questions "q1.txt" --model "anthropic/claude-3.5-sonnet" --format jsonl
  python main.py --questions "q1.txt" --no-router   # force single-model mode

Multi-model router mode (default when LLM_PROVIDER=openrouter and LLM_USE_ROUTER=true):
  Rotates across three free OpenRouter models using LangChain's with_fallbacks().
  Node affinity: reasoning nodes → DeepSeek V4 Flash, code nodes → Qwen3 Coder,
  synthesis → DeepSeek V4 Flash, fallback everywhere → Gemma 4 31B.

Single-model mode (--model flag or LLM_USE_ROUTER=false):
  Uses the specified model for all nodes — matches the original behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_questions(source: str) -> list[str]:
    """Load questions from a file or inline string.

    Supports two formats:

    Multi-line MCQ format (preferred):
        Question: <stem>
        A. <option>
        B. <option>
        C. <option>
        D. <option>

    Plain format (one question per line):
        What is the capital of France?
        Who invented the telephone?

    Returns each question as a single formatted string block.
    """
    p = Path(source)
    raw = p.read_text(encoding="utf-8") if p.exists() else source
    return _parse_question_blocks(raw)


def _parse_question_blocks(text: str) -> list[str]:
    """Parse text into question blocks."""
    import re

    if "Question:" in text:
        raw_blocks = re.split(r"\n(?=Question:)", text.strip())
        blocks = []
        for block in raw_blocks:
            block = block.strip()
            if block:
                blocks.append(block)
        return blocks

    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a question dataset from sample questions using AI."
    )
    parser.add_argument(
        "--questions", "-q",
        required=True,
        help="Path to a text file (one question per line) or inline question string.",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of questions to generate (overrides DATASET_DEFAULT_NUM_QUESTIONS).",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=(
            "Single model identifier, e.g. 'openai/gpt-4o' "
            "(bypasses multi-model router; overrides LLM_MODEL)."
        ),
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        default=False,
        help="Force single-model mode even if LLM_USE_ROUTER=true.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl", "csv"],
        default=None,
        help="Output format (overrides DATASET_OUTPUT_FORMAT).",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Output directory (overrides DATASET_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--instructions", "-i",
        default="",
        help="Additional generation instructions, e.g. 'Focus on exam-style questions'.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file (default: .env in current directory).",
    )
    args = parser.parse_args()

    # Load environment variables before importing settings
    from dotenv import load_dotenv
    load_dotenv(args.env, override=False)

    from config.settings import DatasetConfig, LLMConfig, ModelRouterConfig, ResearchConfig
    from tools.search import create_search_tool
    from agents.graph import build_graph

    # Build configs; apply CLI overrides
    llm_config = LLMConfig()
    if args.model:
        llm_config.model = args.model

    research_config = ResearchConfig()
    dataset_config = DatasetConfig()

    if args.num:
        dataset_config.default_num_questions = args.num
    if args.format:
        dataset_config.output_format = args.format
    if args.out:
        dataset_config.output_dir = args.out

    # Load sample questions
    sample_questions = _load_questions(args.questions)
    if not sample_questions:
        print("[ERROR] No sample questions found. Provide at least one question.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(sample_questions)} sample question(s).")

    # ── Build LLM / router ────────────────────────────────────────────────────
    use_router = (
        llm_config.use_router
        and not args.model       # explicit --model bypasses router
        and not args.no_router   # explicit --no-router flag bypasses router
    )

    if use_router:
        from providers.llm import create_model_router
        router_config = ModelRouterConfig()
        llm_or_router = create_model_router(llm_config, router_config)
        print(
            f"[INFO] Multi-model router: {llm_or_router.model_summary}\n"
            f"       Rate-limit failover: primary → secondary → fallback (LangChain with_fallbacks)"
        )
    else:
        from providers.llm import create_llm
        llm_or_router = create_llm(llm_config)
        print(f"[INFO] Single-model mode: {llm_config.model}")

    print(
        f"[INFO] Target: {dataset_config.default_num_questions} questions | "
        f"Search: {research_config.search_provider} | "
        f"Format: {dataset_config.output_format}"
    )

    search_tool = create_search_tool(research_config)
    graph = build_graph(llm_or_router, search_tool, dataset_config, research_config)

    # ── Run the graph ─────────────────────────────────────────────────────────
    initial_state: dict = {
        "sample_questions": sample_questions,
        "num_questions": dataset_config.default_num_questions,
        "extra_instructions": args.instructions,
    }

    print("[INFO] Running dataset generation pipeline…\n")
    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        _handle_fatal_exception(exc)
        sys.exit(1)

    # ── Print run summary (timing, quality history, research stats) ───────────
    from utils.logging import print_run_summary
    print_run_summary(result)

    if result.get("error"):
        print(f"[ERROR] {result['error']}", file=sys.stderr)
        if result.get("output_path"):
            print(f"[INFO] Partial results saved to: {result['output_path']}")
        sys.exit(1)

    output_path = result.get("output_path", "")
    topic = result.get("topic", "Unknown")
    count = len(result.get("final_dataset", []))

    print(f"[OK] Dataset generated successfully.")
    print(f"     Topic    : {topic}")
    print(f"     Questions: {count}")
    print(f"     Output   : {output_path}")


def _handle_fatal_exception(exc: Exception) -> None:
    """Print a clean error message for known failure modes."""
    msg = str(exc)
    if "429" in msg or "Rate limit" in msg or "rate_limit" in msg.lower():
        import re
        reset_ts = re.search(r"X-RateLimit-Reset['\": ]+(\d+)", msg)
        reset_note = ""
        if reset_ts:
            import datetime
            ts_ms = int(reset_ts.group(1))
            dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
            reset_note = f" Quota resets at {dt.strftime('%Y-%m-%d %H:%M UTC')}."
        print(
            f"\n[ERROR] All models rate-limited (free tier exhausted).{reset_note}\n"
            "       Options:\n"
            "         1. Wait for the daily quota to reset\n"
            "         2. Add credits at https://openrouter.ai/credits\n"
            "         3. Try again later — quotas reset at different times per model\n"
            "         4. Switch to a paid model: --model openai/gpt-4o-mini --no-router",
            file=sys.stderr,
        )
    else:
        print(f"\n[ERROR] Unexpected error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
