"""
Shared state that flows through every node in the research graph.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    # ── inputs ───────────────────────────────────────────────────────────────
    target_name:      str          # e.g. "Elon Musk"
    target_context:   str          # e.g. "CEO of Tesla and SpaceX"
    max_iterations:   int          # safety cap on search/scrape loops

    # ── planning ─────────────────────────────────────────────────────────────
    research_plan:    list[str]    # ordered list of research goals
    pending_queries:  list[str]    # search queries yet to be executed

    # ── raw data ─────────────────────────────────────────────────────────────
    search_results:   list[dict]   # [{query, url, title, snippet}]
    scraped_pages:    list[dict]   # [{url, title, content, source_type}]

    # ── extracted facts ──────────────────────────────────────────────────────
    extracted_facts:  list[dict]   # [{fact, source_url, confidence, category}]

    # ── memory / dedup ───────────────────────────────────────────────────────
    seen_urls:        list[str]
    memory_summary:   str          # running LLM summary of findings so far

    # ── loop control ─────────────────────────────────────────────────────────
    iterations:       int

    # ── final outputs ────────────────────────────────────────────────────────
    profile:          dict         # structured founder profile
    report_path:      str          # path to generated markdown/JSON report
    report_markdown:  str
