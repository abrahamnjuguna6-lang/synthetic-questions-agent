from langchain_core.tools import BaseTool


def create_wikipedia_tool() -> BaseTool:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    )
