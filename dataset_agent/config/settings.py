from pydantic_settings import BaseSettings
from pydantic import Field


class LLMConfig(BaseSettings):
    provider: str = "openrouter"
    api_key: str = Field(..., description="OpenRouter (or other provider) API key")
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    # Enable multi-model rotation on OpenRouter free tier (see ModelRouterConfig).
    # Set LLM_USE_ROUTER=false to force single-model mode, or use --model CLI flag.
    use_router: bool = True

    model_config = {"env_prefix": "LLM_", "extra": "ignore"}


class ModelRouterConfig(BaseSettings):
    """Free-tier model IDs for the three rotation slots.

    Defaults are the top three free models on OpenRouter as of May 2026,
    ordered by reasoning quality and code-analysis capability.
    """
    primary_model: str = "deepseek/deepseek-v4-flash:free"   # 1M ctx, best reasoning
    secondary_model: str = "qwen/qwen3-coder:free"           # 1M ctx, best coding
    fallback_model: str = "google/gemma-4-31b-it:free"       # 256K ctx, dense/reliable

    model_config = {"env_prefix": "ROUTER_", "extra": "ignore"}


class ResearchConfig(BaseSettings):
    search_provider: str = "duckduckgo"  # "tavily" | "duckduckgo" | "wikipedia"
    tavily_api_key: str = ""
    max_results_per_query: int = 5       # increased: Tavily advanced returns richer docs
    max_research_queries: int = 3
    # Tavily-specific settings for richer context (only active when provider=tavily)
    # search_depth: "basic" | "advanced" | "fast" | "ultra-fast"
    tavily_search_depth: str = "advanced"
    # include_raw_content: "markdown" = full page as structured markdown (best for LLMs)
    #                      "text"     = full page as plain text
    #                      True/False = raw_content field included/excluded (older API)
    tavily_include_raw_content: str = "markdown"
    # include_answer: "advanced" = rich AI-synthesised paragraph (most useful)
    #                 "basic"    = short AI answer
    #                 True/False = legacy bool (older API)
    tavily_include_answer: str = "advanced"

    model_config = {"env_prefix": "RESEARCH_", "extra": "ignore"}


class DatasetConfig(BaseSettings):
    default_num_questions: int = 20
    max_retries: int = 3
    quality_threshold: float = 0.75
    output_format: str = "json"  # "json" | "jsonl" | "csv"
    output_dir: str = "./output"
    generation_batch_size: int = 5  # questions per LLM call; keeps responses short

    model_config = {"env_prefix": "DATASET_", "extra": "ignore"}
