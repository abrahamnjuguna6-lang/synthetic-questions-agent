from langchain_core.tools import BaseTool
from config.settings import ResearchConfig


def create_search_tool(config: ResearchConfig) -> BaseTool:
    """Return a search tool based on configured provider.

    Priority:
      1. Tavily  — richest results, requires RESEARCH_TAVILY_API_KEY
      2. DuckDuckGo — no API key, good fallback
      3. Wikipedia — guaranteed fallback, topic-level summaries only
    """
    provider = config.search_provider.lower()

    if provider == "tavily":
        if not config.tavily_api_key:
            raise ValueError(
                "RESEARCH_SEARCH_PROVIDER=tavily requires RESEARCH_TAVILY_API_KEY to be set."
            )
        import os
        from langchain_tavily import TavilySearch

        # TavilySearch reads TAVILY_API_KEY from env; bridge our custom var name.
        os.environ.setdefault("TAVILY_API_KEY", config.tavily_api_key)

        # advanced depth crawls deeper pages and returns more structured content;
        # include_raw_content gives full page text (not just snippets) — richer
        # context for question generation; include_answer adds Tavily's own AI
        # synthesis which acts as a free context summary.
        return TavilySearch(
            max_results=config.max_results_per_query,
            search_depth=config.tavily_search_depth,
            include_raw_content=config.tavily_include_raw_content,
            include_answer=config.tavily_include_answer,
        )

    if provider == "duckduckgo":
        from langchain_community.tools import DuckDuckGoSearchRun

        return DuckDuckGoSearchRun()

    if provider == "wikipedia":
        from tools.wikipedia import create_wikipedia_tool

        return create_wikipedia_tool()

    raise ValueError(
        f"Unknown search provider: {provider!r}. "
        "Supported: 'tavily', 'duckduckgo', 'wikipedia'."
    )
