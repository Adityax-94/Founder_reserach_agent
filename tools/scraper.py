"""
Web scraper tool.

Strategy:
  1. Detect source type (LinkedIn, Crunchbase, news, generic).
  2. Use requests + BeautifulSoup for most pages.
  3. Apply source-specific content selectors where possible.
  4. Return clean plain text + metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 12  # seconds


# ── source-type detection ─────────────────────────────────────────────────────

def _classify_url(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "linkedin.com" in domain:
        return "linkedin"
    if "crunchbase.com" in domain:
        return "crunchbase"
    if "wikipedia.org" in domain:
        return "wikipedia"
    if "techcrunch.com" in domain or "forbes.com" in domain or "bloomberg.com" in domain:
        return "news"
    if "twitter.com" in domain or "x.com" in domain:
        return "twitter"
    return "generic"


# ── content extractors ────────────────────────────────────────────────────────

def _extract_wikipedia(soup: BeautifulSoup) -> str:
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return _extract_generic(soup)
    # Remove nav boxes, references, edit links
    for tag in content_div.find_all(["table", "sup", "span.mw-editsection"]):
        tag.decompose()
    paragraphs = content_div.find_all("p")
    return "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs[:30])


def _extract_news(soup: BeautifulSoup) -> str:
    # Try common article containers
    for selector in ["article", "[class*='article-body']", "[class*='story-body']",
                     "[class*='post-content']", "main"]:
        container = soup.select_one(selector)
        if container:
            return "\n\n".join(
                p.get_text(" ", strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) > 40
            )
    return _extract_generic(soup)


def _extract_generic(soup: BeautifulSoup) -> str:
    # Remove boilerplate
    for tag in soup.find_all(["nav", "footer", "header", "aside",
                               "script", "style", "noscript"]):
        tag.decompose()
    paragraphs = soup.find_all("p")
    text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
    return text or soup.get_text(" ", strip=True)[:5000]


# ── main scraper ──────────────────────────────────────────────────────────────

def scrape_url(url: str) -> Optional[dict]:
    """
    Scrape a URL and return:
      {url, title, content, source_type, word_count}
    Returns None on failure.
    """
    source_type = _classify_url(url)

    # LinkedIn blocks scrapers – skip gracefully
    if source_type == "linkedin":
        logger.info("[scrape] Skipping LinkedIn URL (bot-blocked): %s", url)
        return None

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("[scrape] Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    if source_type == "wikipedia":
        content = _extract_wikipedia(soup)
    elif source_type == "news":
        content = _extract_news(soup)
    else:
        content = _extract_generic(soup)

    # Normalise whitespace
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    if len(content) < 100:
        logger.info("[scrape] Too little content at %s (%d chars)", url, len(content))
        return None

    logger.debug("[scrape] %s → %d chars (%s)", url, len(content), source_type)

    return {
        "url":         url,
        "title":       title,
        "content":     content[:8000],   # cap per page
        "source_type": source_type,
        "word_count":  len(content.split()),
    }
