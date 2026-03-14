"""
LangGraph node implementations.

Each node receives the full ResearchState dict and returns a partial dict
with only the keys it mutates – LangGraph merges them automatically.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.state import ResearchState
from memory.store import MemoryStore
from tools.scraper import scrape_url
from tools.search import web_search

logger = logging.getLogger(__name__)

# ── shared LLM ───────────────────────────────────────────────────────────────
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)

_memory_store = MemoryStore()


# ── helpers ──────────────────────────────────────────────────────────────────

def _call_llm_json(system: str, user: str) -> dict | list:
    """Call the LLM and parse its response as JSON."""
    resp = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = resp.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def _call_llm_text(system: str, user: str) -> str:
    resp = _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()


# ── nodes ────────────────────────────────────────────────────────────────────

def plan_node(state: ResearchState) -> dict:
    """Generate a structured research plan and initial search queries."""
    logger.info("[plan] Building research plan for: %s", state["target_name"])

    system = (
        "You are an expert research strategist. Given a founder/CEO name, "
        "output a JSON object with two keys:\n"
        "  'research_goals': list of 6-8 distinct research objectives\n"
        "  'search_queries': list of 10-15 targeted search queries\n"
        "Cover: background, education, career history, companies founded/led, "
        "funding rounds, vision/philosophy, controversies, recent news, interviews.\n"
        "Return ONLY valid JSON – no prose."
    )
    user = (
        f"Target: {state['target_name']}\n"
        f"Context: {state.get('target_context', 'No additional context provided.')}"
    )

    result = _call_llm_json(system, user)

    return {
        "research_plan":   result.get("research_goals", []),
        "pending_queries": result.get("search_queries", []),
        "search_results":  [],
        "scraped_pages":   [],
        "extracted_facts": [],
        "seen_urls":       [],
        "memory_summary":  "",
        "iterations":      0,
        "max_iterations":  state.get("max_iterations", 5),
    }


def search_node(state: ResearchState) -> dict:
    """Execute the next batch of search queries."""
    pending   = list(state.get("pending_queries", []))
    seen_urls = list(state.get("seen_urls", []))
    existing  = list(state.get("search_results", []))

    # Take up to 3 queries per iteration
    batch, remaining = pending[:3], pending[3:]

    logger.info("[search] Running %d queries (remaining after: %d)", len(batch), len(remaining))

    new_results: list[dict] = []
    new_urls:    list[str]  = []

    for query in batch:
        hits = web_search(query, num_results=5)
        for hit in hits:
            if hit["url"] not in seen_urls:
                seen_urls.append(hit["url"])
                new_urls.append(hit["url"])
                new_results.append({**hit, "query": query})

    return {
        "search_results":  existing + new_results,
        "pending_queries": remaining,
        "seen_urls":       seen_urls,
    }


def scrape_node(state: ResearchState) -> dict:
    """Scrape the URLs discovered in this iteration."""
    search_results = state.get("search_results", [])
    scraped_pages  = list(state.get("scraped_pages", []))
    scraped_urls   = {p["url"] for p in scraped_pages}

    # Only scrape URLs from results we haven't scraped yet
    to_scrape = [
        r for r in search_results
        if r["url"] not in scraped_urls
    ][:8]  # cap per iteration

    logger.info("[scrape] Scraping %d new URLs", len(to_scrape))

    for result in to_scrape:
        page = scrape_url(result["url"])
        if page:
            scraped_pages.append(page)

    return {"scraped_pages": scraped_pages}


def analyse_node(state: ResearchState) -> dict:
    """Extract structured facts from newly scraped pages."""
    scraped_pages  = state.get("scraped_pages", [])
    extracted_facts = list(state.get("extracted_facts", []))
    target_name    = state["target_name"]
    memory_summary = state.get("memory_summary", "")

    # Only analyse pages not yet processed (simple heuristic: compare counts)
    already_processed = len(extracted_facts)  # rough proxy
    new_pages = scraped_pages[already_processed // 3:]  # approx new pages

    if not new_pages:
        logger.info("[analyse] No new pages to analyse")
        return {}

    logger.info("[analyse] Extracting facts from %d pages", len(new_pages))

    combined_text = "\n\n---\n\n".join(
        f"SOURCE: {p['url']}\nTITLE: {p.get('title','')}\n\n{p['content'][:3000]}"
        for p in new_pages[:5]
    )

    system = (
        "You are an expert researcher extracting structured facts about a person. "
        "Return a JSON array of fact objects. Each object must have:\n"
        "  'fact': concise factual statement\n"
        "  'category': one of [background, education, career, companies, "
        "               funding, philosophy, controversy, achievement, recent_news, quote]\n"
        "  'source_url': URL where this was found\n"
        "  'confidence': high | medium | low\n"
        "Deduplicate against already-known facts. Return ONLY valid JSON array."
    )
    user = (
        f"Person: {target_name}\n\n"
        f"Already known facts summary:\n{memory_summary or 'None yet'}\n\n"
        f"New source content:\n{combined_text}"
    )

    try:
        new_facts = _call_llm_json(system, user)
        if isinstance(new_facts, list):
            extracted_facts.extend(new_facts)
    except Exception as exc:
        logger.warning("[analyse] JSON parse error: %s", exc)

    # Also generate follow-up queries based on gaps
    gap_system = (
        "Given the facts gathered so far, identify 3-5 specific gaps in the research "
        "about this person. Return a JSON array of search query strings to fill those gaps. "
        "Return ONLY a JSON array of strings."
    )
    gap_user = (
        f"Person: {target_name}\n"
        f"Facts so far: {json.dumps(extracted_facts[-20:], indent=2)}"
    )

    follow_up_queries: list[str] = []
    try:
        follow_up_queries = _call_llm_json(gap_system, gap_user)  # type: ignore[assignment]
    except Exception:
        pass

    pending = list(state.get("pending_queries", []))
    pending.extend(follow_up_queries)

    return {
        "extracted_facts":  extracted_facts,
        "pending_queries":  pending,
        "iterations":       state.get("iterations", 0) + 1,
    }


def memory_write_node(state: ResearchState) -> dict:
    """Summarise and persist findings to the memory store."""
    target_name    = state["target_name"]
    extracted_facts = state.get("extracted_facts", [])

    logger.info("[memory] Writing %d facts to memory", len(extracted_facts))

    # Store individual facts
    for fact in extracted_facts:
        _memory_store.upsert(
            key=fact.get("fact", "")[:120],
            value=fact,
            namespace=target_name,
        )

    # Rolling LLM summary
    system = (
        "Produce a concise running summary (max 400 words) of everything known "
        "about this person based on the facts provided. Be factual and structured."
    )
    user = (
        f"Person: {target_name}\n\n"
        f"All facts:\n{json.dumps(extracted_facts, indent=2)}"
    )
    summary = _call_llm_text(system, user)

    return {"memory_summary": summary}


def finalise_node(state: ResearchState) -> dict:
    """Synthesise all facts into a canonical structured profile."""
    target_name    = state["target_name"]
    extracted_facts = state.get("extracted_facts", [])
    memory_summary = state.get("memory_summary", "")

    logger.info("[finalise] Building structured profile for %s", target_name)

    system = (
        "You are an expert analyst. Synthesise all research into a structured JSON profile "
        "for a founder/CEO. The JSON must include these top-level keys:\n"
        "  name, title, summary (2-3 sentence bio),\n"
        "  education (list of {institution, degree, year}),\n"
        "  career_timeline (list of {year, role, organisation, description}),\n"
        "  companies (list of {name, role, founded_year, status, description}),\n"
        "  funding_highlights (list of {round, amount, year, company}),\n"
        "  key_achievements (list of strings),\n"
        "  philosophy_and_vision (list of key belief/theme strings),\n"
        "  notable_quotes (list of {quote, source, year}),\n"
        "  controversies (list of {topic, description, year}),\n"
        "  recent_news (list of {headline, date, url}),\n"
        "  sources (list of unique URLs used)\n"
        "Return ONLY valid JSON – no prose."
    )
    user = (
        f"Person: {target_name}\n\n"
        f"Research summary:\n{memory_summary}\n\n"
        f"All extracted facts:\n{json.dumps(extracted_facts, indent=2)}"
    )

    try:
        profile = _call_llm_json(system, user)
    except Exception as exc:
        logger.error("[finalise] Profile build failed: %s", exc)
        profile = {"name": target_name, "error": str(exc)}

    return {"profile": profile}


def report_node(state: ResearchState) -> dict:
    """Render the profile as a human-readable Markdown report and save it."""
    import os
    from datetime import datetime

    profile = state.get("profile", {})
    target_name = state["target_name"]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Markdown rendering ────────────────────────────────────────────────────
    lines: list[str] = []

    def h(level: int, text: str):
        lines.append(f"{'#' * level} {text}")
        lines.append("")

    def ul(items: list):
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    h(1, f"Founder Research Report: {profile.get('name', target_name)}")
    lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    h(2, "Summary")
    lines.append(profile.get("summary", "_No summary available._"))
    lines.append("")

    if profile.get("education"):
        h(2, "Education")
        for edu in profile["education"]:
            lines.append(f"- **{edu.get('institution', '')}** – {edu.get('degree', '')} ({edu.get('year', '')})")
        lines.append("")

    if profile.get("career_timeline"):
        h(2, "Career Timeline")
        for item in sorted(profile["career_timeline"], key=lambda x: str(x.get("year", "")), reverse=True):
            lines.append(f"- **{item.get('year', '?')}** · {item.get('role', '')} @ {item.get('organisation', '')}  ")
            if item.get("description"):
                lines.append(f"  _{item['description']}_")
        lines.append("")

    if profile.get("companies"):
        h(2, "Companies")
        for co in profile["companies"]:
            status_badge = f"({co.get('status', '')})" if co.get("status") else ""
            lines.append(f"### {co.get('name', '')} {status_badge}")
            lines.append(f"**Role:** {co.get('role', '')}  ")
            lines.append(f"**Founded:** {co.get('founded_year', 'N/A')}  ")
            lines.append(co.get("description", ""))
            lines.append("")

    if profile.get("funding_highlights"):
        h(2, "Funding Highlights")
        for fh in profile["funding_highlights"]:
            lines.append(f"- {fh.get('company', '')} · {fh.get('round', '')} · **{fh.get('amount', '')}** ({fh.get('year', '')})")
        lines.append("")

    if profile.get("key_achievements"):
        h(2, "Key Achievements")
        ul(profile["key_achievements"])

    if profile.get("philosophy_and_vision"):
        h(2, "Philosophy & Vision")
        ul(profile["philosophy_and_vision"])

    if profile.get("notable_quotes"):
        h(2, "Notable Quotes")
        for q in profile["notable_quotes"]:
            lines.append(f"> \"{q.get('quote', '')}\"")
            lines.append(f"> — *{q.get('source', '')}*, {q.get('year', '')}")
            lines.append("")

    if profile.get("controversies"):
        h(2, "Controversies")
        for c in profile["controversies"]:
            lines.append(f"- **{c.get('topic', '')}** ({c.get('year', '')}): {c.get('description', '')}")
        lines.append("")

    if profile.get("recent_news"):
        h(2, "Recent News")
        for n in profile["recent_news"]:
            url = n.get("url", "")
            headline = n.get("headline", "")
            date = n.get("date", "")
            if url:
                lines.append(f"- [{headline}]({url}) — {date}")
            else:
                lines.append(f"- {headline} — {date}")
        lines.append("")

    if profile.get("sources"):
        h(2, "Sources")
        for src in profile["sources"]:
            lines.append(f"- {src}")
        lines.append("")

    markdown = "\n".join(lines)

    # ── Save files ─────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    slug = target_name.lower().replace(" ", "_")
    md_path   = f"output/{slug}_{ts}.md"
    json_path = f"output/{slug}_{ts}.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"profile": profile, "facts": state.get("extracted_facts", [])}, f, indent=2)

    logger.info("[report] Saved → %s and %s", md_path, json_path)

    return {
        "report_path":     md_path,
        "report_markdown": markdown,
    }
