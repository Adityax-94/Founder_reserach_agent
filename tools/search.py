"""
Web search tool.

Primary:  Tavily Search API  (TAVILY_API_KEY env var)
Fallback: DuckDuckGo via duckduckgo-search library (no key required)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web for `query` and return a list of result dicts:
      [{url, title, snippet, source}]
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        return _tavily_search(query, num_results, api_key)
    else:
        logger.warning("TAVILY_API_KEY not set – falling back to DuckDuckGo")
        return _ddg_search(query, num_results)


# ── Tavily ────────────────────────────────────────────────────────────────────

def _tavily_search(query: str, num_results: int, api_key: str) -> list[dict]:
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=num_results,
            search_depth="advanced",
            include_raw_content=False,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "url":     r.get("url", ""),
                "title":   r.get("title", ""),
                "snippet": r.get("content", ""),
                "source":  "tavily",
            })
        logger.debug("[search:tavily] %d results for: %s", len(results), query)
        return results
    except Exception as exc:
        logger.error("[search:tavily] error: %s", exc)
        return []


# ── DuckDuckGo fallback ───────────────────────────────────────────────────────

def _ddg_search(query: str, num_results: int) -> list[dict]:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=num_results))
        results = []
        for r in hits:
            results.append({
                "url":     r.get("href", ""),
                "title":   r.get("title", ""),
                "snippet": r.get("body", ""),
                "source":  "duckduckgo",
            })
        logger.debug("[search:ddg] %d results for: %s", len(results), query)
        return results
    except Exception as exc:
        logger.error("[search:ddg] error: %s", exc)
        return []
