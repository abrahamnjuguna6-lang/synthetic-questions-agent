"""Provider-aware multi-model router with LangChain fallback chains.

Supports two providers:
  - "anthropic"  — Claude 4.x family via ChatAnthropic
  - "openrouter" — Free-tier models via ChatOpenAI + base_url override

Node affinity maps each pipeline stage to the right model tier:

  Anthropic (Haiku-preferred for cost):
  ┌─────────────────────────┬──────────────────┬────────────────────────────────┐
  │ Node(s)                 │ Affinity         │ Primary → Fallback chain       │
  ├─────────────────────────┼──────────────────┼────────────────────────────────┤
  │ analyze_topic           │ reasoning_llm    │ Sonnet 4.6 → Haiku 4.5 → Opus  │
  │ validate_quality        │ reasoning_llm    │ (needs reliable scoring)       │
  ├─────────────────────────┼──────────────────┼────────────────────────────────┤
  │ plan_research           │ code_llm         │ Haiku 4.5 → Sonnet 4.6         │
  │ generate_questions      │ code_llm         │ (bulk tasks — cheapest first)  │
  ├─────────────────────────┼──────────────────┼────────────────────────────────┤
  │ synthesize_context      │ balanced_llm     │ Haiku 4.5 → Sonnet 4.6         │
  └─────────────────────────┴──────────────────┴────────────────────────────────┘

  Config mapping (primary=Sonnet, secondary=Haiku, fallback=Opus):
    reasoning_llm : primary(Sonnet) → secondary(Haiku) → fallback(Opus)
    code_llm      : secondary(Haiku) → primary(Sonnet)          [no Opus — overkill]
    balanced_llm  : secondary(Haiku) → primary(Sonnet)          [cheapest first]

  Why this order?
    - Haiku ($1/$5 per 1M) handles ~70% of pipeline work (research, generation, synthesis)
    - Sonnet ($3/$15) guards the two reasoning-critical nodes
    - Opus ($5/$25) is the last resort, only activates if both others are rate-limited

  OpenRouter (unchanged from prior implementation):
    reasoning_llm : DeepSeek → Qwen3 → Gemma
    code_llm      : Qwen3 → DeepSeek → Gemma
    balanced_llm  : DeepSeek → Gemma → Qwen3
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

# ── OpenRouter constants (used only when provider != "anthropic") ──────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/dataset-agent",
    "X-Title": "Dataset Agent",
}

# Free model IDs on OpenRouter (May 2026)
MODEL_DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash:free"
MODEL_QWEN3_CODER       = "qwen/qwen3-coder:free"
MODEL_GEMMA4_31B        = "google/gemma-4-31b-it:free"

# Anthropic Claude 4.x model IDs (May 2026)
MODEL_CLAUDE_OPUS    = "claude-opus-4-7"
MODEL_CLAUDE_SONNET  = "claude-sonnet-4-6"
MODEL_CLAUDE_HAIKU   = "claude-haiku-4-5-20251001"


class ModelRouter:
    """Provider-aware model router with LangChain with_fallbacks() cascade chains.

    Each affinity property (reasoning_llm / code_llm / balanced_llm) returns a
    fully LCEL-compatible BaseChatModel: ``PROMPT | router.reasoning_llm``.

    The fallback chain triggers on the correct rate-limit exception for each
    provider (anthropic.RateLimitError or openai.RateLimitError), so the model
    switch is invisible to all calling nodes.
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "openrouter",
        primary_model: str = MODEL_DEEPSEEK_V4_FLASH,
        secondary_model: str = MODEL_QWEN3_CODER,
        fallback_model: str = MODEL_GEMMA4_31B,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._provider = provider.lower()

        def _make(model_id: str) -> BaseChatModel:
            if self._provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=model_id,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,
                )
            else:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=OPENROUTER_BASE_URL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=1,
                    default_headers=_OPENROUTER_HEADERS,
                )

        self._primary   = _make(primary_model)
        self._secondary = _make(secondary_model)
        self._fallback  = _make(fallback_model)

        self.primary_model_id   = primary_model
        self.secondary_model_id = secondary_model
        self.fallback_model_id  = fallback_model

    # ── Internal helper ────────────────────────────────────────────────────────

    def _cascade(self, *models: BaseChatModel) -> BaseChatModel:
        """Build a with_fallbacks() chain from an ordered list of models.

        Catches the provider-specific rate-limit exception so only genuine
        429s trigger fallback — other errors surface normally.
        """
        first, *rest = models
        if not rest:
            return first

        exceptions: tuple
        if self._provider == "anthropic":
            try:
                import anthropic
                exceptions = (anthropic.RateLimitError,)
            except ImportError:
                exceptions = (Exception,)
        else:
            try:
                from openai import RateLimitError
                exceptions = (RateLimitError,)
            except ImportError:
                exceptions = (Exception,)

        return first.with_fallbacks(list(rest), exceptions_to_handle=exceptions)

    # ── Public LLM properties ─────────────────────────────────────────────────

    @property
    def reasoning_llm(self) -> BaseChatModel:
        """For: analyze_topic, validate_quality — highest-stakes reasoning.

        Anthropic:   Sonnet 4.6 → Haiku 4.5 → Opus 4.7
          Sonnet handles reliable topic analysis and quality scoring.
          Falls back to Haiku on rate limit; Opus is the last resort.

        OpenRouter:  DeepSeek → Qwen3 Coder → Gemma 4 31B
        """
        if self._provider == "anthropic":
            return self._cascade(self._primary, self._secondary, self._fallback)
        return self._cascade(self._primary, self._secondary, self._fallback)

    @property
    def code_llm(self) -> BaseChatModel:
        """For: plan_research, generate_questions — bulk / routine LLM work.

        Anthropic:   Haiku 4.5 → Sonnet 4.6
          Haiku is fast and cheap for query generation and batch MCQ output.
          Sonnet fallback only if Haiku is rate-limited. Opus excluded (overkill).

        OpenRouter:  Qwen3 Coder → DeepSeek → Gemma 4 31B
        """
        if self._provider == "anthropic":
            return self._cascade(self._secondary, self._primary)
        return self._cascade(self._secondary, self._primary, self._fallback)

    @property
    def balanced_llm(self) -> BaseChatModel:
        """For: synthesize_context — prose summarisation of research snippets.

        Anthropic:   Haiku 4.5 → Sonnet 4.6
          Synthesis is routine prose work — Haiku excels and costs ~5× less.
          Sonnet steps in only if Haiku is rate-limited.

        OpenRouter:  DeepSeek → Gemma 4 31B → Qwen3 Coder
        """
        if self._provider == "anthropic":
            return self._cascade(self._secondary, self._primary)
        return self._cascade(self._primary, self._fallback, self._secondary)

    @property
    def model_summary(self) -> str:
        return (
            f"provider={self._provider} | "
            f"primary={self.primary_model_id} | "
            f"secondary={self.secondary_model_id} | "
            f"fallback={self.fallback_model_id}"
        )
