from langchain_core.language_models import BaseChatModel
from config.settings import LLMConfig, ModelRouterConfig


def create_model_router(llm_config: LLMConfig, router_config: ModelRouterConfig):
    """Create a provider-aware ModelRouter with LangChain with_fallbacks() chains.

    For Anthropic: reasoning nodes get Sonnet 4.6, routine nodes get Haiku 4.5,
    Opus 4.7 is the last-resort fallback. See config/model_router.py for the
    full cascade table.

    For OpenRouter: DeepSeek / Qwen3 / Gemma free-tier rotation (unchanged).
    """
    from config.model_router import ModelRouter

    return ModelRouter(
        api_key=llm_config.api_key,
        provider=llm_config.provider,
        primary_model=router_config.primary_model,
        secondary_model=router_config.secondary_model,
        fallback_model=router_config.fallback_model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Return a configured chat model.

    OpenRouter is OpenAI-API-compatible, so ChatOpenAI handles it via base_url.
    Add new provider branches here without touching any agent logic.
    """
    if config.provider in ("openrouter", "openai"):
        from langchain_openai import ChatOpenAI

        # OpenRouter requires HTTP-Referer and recommends X-Title for routing
        # and analytics. Without these some free models may return 403.
        extra_headers: dict = {}
        if config.provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://github.com/dataset-agent",
                "X-Title": "Dataset Agent",
            }

        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            default_headers=extra_headers,
        )

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    raise ValueError(
        f"Unknown provider: {config.provider!r}. "
        "Supported: 'openrouter', 'openai', 'anthropic', 'google'."
    )
