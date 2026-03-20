"""
core/web_search.py
Tavily web search fallback — returns results as LangChain Documents.
"""
from langchain_core.documents import Document
from config import TAVILY_API_KEY, WEB_SEARCH_ENABLED


def web_search_fallback(query: str, max_results: int = 5) -> list[Document]:
    """
    Query Tavily and return results as a list of Documents.
    Returns [] if web search is disabled or the API call fails.
    """
    if not WEB_SEARCH_ENABLED:
        print("⚠️  Web search disabled — add TAVILY_API_KEY to .env")
        return []

    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=max_results)
        docs    = []
        for item in results.get("results", []):
            content = item.get("content") or item.get("snippet") or ""
            url     = item.get("url", "web")
            title   = item.get("title", "Web Result")
            if content:
                docs.append(Document(
                    page_content=f"[Web: {title}]\n{content}",
                    metadata={"source": url, "type": "web"},
                ))
        print(f"🌐 Tavily returned {len(docs)} results")
        return docs
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []
