"""
Unit tests for Founder Research Agent tools and memory.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Memory Store tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_upsert_and_get(self, tmp_path):
        from memory.store import MemoryStore
        store = MemoryStore(persist_path=str(tmp_path / "mem.json"))
        store.upsert("fact1", {"text": "Founded OpenAI"}, namespace="Sam Altman")
        val = store.get("fact1", namespace="Sam Altman")
        assert val == {"text": "Founded OpenAI"}

    def test_get_all(self, tmp_path):
        from memory.store import MemoryStore
        store = MemoryStore(persist_path=str(tmp_path / "mem.json"))
        store.upsert("k1", "v1", namespace="ns")
        store.upsert("k2", "v2", namespace="ns")
        all_vals = store.get_all(namespace="ns")
        assert "v1" in all_vals
        assert "v2" in all_vals

    def test_persistence(self, tmp_path):
        from memory.store import MemoryStore
        path = str(tmp_path / "mem.json")
        store1 = MemoryStore(persist_path=path)
        store1.upsert("key", "persistent_value", namespace="test")
        store1.save()

        store2 = MemoryStore(persist_path=path)
        assert store2.get("key", namespace="test") == "persistent_value"

    def test_clear_namespace(self, tmp_path):
        from memory.store import MemoryStore
        store = MemoryStore(persist_path=str(tmp_path / "mem.json"))
        store.upsert("k", "v", namespace="temp")
        store.clear_namespace("temp")
        assert store.get_all(namespace="temp") == []

    def test_stats(self, tmp_path):
        from memory.store import MemoryStore
        store = MemoryStore(persist_path=str(tmp_path / "mem.json"))
        store.upsert("a", 1, namespace="ns")
        store.upsert("b", 2, namespace="ns")
        stats = store.stats("ns")
        assert stats["entry_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Scraper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScraper:
    def test_classify_linkedin(self):
        from tools.scraper import _classify_url
        assert _classify_url("https://www.linkedin.com/in/samaltman") == "linkedin"

    def test_classify_wikipedia(self):
        from tools.scraper import _classify_url
        assert _classify_url("https://en.wikipedia.org/wiki/Sam_Altman") == "wikipedia"

    def test_classify_generic(self):
        from tools.scraper import _classify_url
        assert _classify_url("https://example.com/article") == "generic"

    def test_linkedin_skipped(self):
        from tools.scraper import scrape_url
        result = scrape_url("https://www.linkedin.com/in/anyone")
        assert result is None

    @patch("tools.scraper.requests.get")
    def test_scrape_generic_page(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = """
        <html>
          <head><title>Test Article</title></head>
          <body>
            <p>Sam Altman is the CEO of OpenAI. He previously led Y Combinator.</p>
            <p>He was born in 1985 in Chicago, Illinois.</p>
            <p>Under his leadership, OpenAI launched GPT-4 and ChatGPT.</p>
          </body>
        </html>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from tools.scraper import scrape_url
        result = scrape_url("https://example.com/sam-altman")

        assert result is not None
        assert result["url"] == "https://example.com/sam-altman"
        assert result["title"] == "Test Article"
        assert "OpenAI" in result["content"]
        assert result["source_type"] == "generic"

    @patch("tools.scraper.requests.get")
    def test_scrape_returns_none_on_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        from tools.scraper import scrape_url
        assert scrape_url("https://bad-url.example.com") is None

    @patch("tools.scraper.requests.get")
    def test_scrape_returns_none_on_thin_content(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Hi</p></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        from tools.scraper import scrape_url
        assert scrape_url("https://thin.example.com") is None


# ─────────────────────────────────────────────────────────────────────────────
# Search tool tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:
    @patch("tools.search.os.getenv")
    @patch("tools.search._ddg_search")
    def test_fallback_to_ddg_when_no_tavily_key(self, mock_ddg, mock_getenv):
        mock_getenv.return_value = None
        mock_ddg.return_value = [{"url": "http://x.com", "title": "X", "snippet": "Y", "source": "ddg"}]
        from tools.search import web_search
        results = web_search("Sam Altman CEO")
        mock_ddg.assert_called_once()
        assert len(results) == 1

    @patch("tools.search.os.getenv")
    @patch("tools.search._tavily_search")
    def test_uses_tavily_when_key_present(self, mock_tavily, mock_getenv):
        mock_getenv.return_value = "tvly-fake"
        mock_tavily.return_value = [{"url": "http://a.com", "title": "A", "snippet": "B", "source": "tavily"}]
        from tools.search import web_search
        results = web_search("Sam Altman founder")
        mock_tavily.assert_called_once()
        assert results[0]["source"] == "tavily"


# ─────────────────────────────────────────────────────────────────────────────
# Agent state tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState:
    def test_state_typeddict_keys(self):
        from agent.state import ResearchState
        # TypedDict should have all expected keys in annotations
        keys = ResearchState.__annotations__.keys()
        for expected in ["target_name", "extracted_facts", "profile", "report_markdown"]:
            assert expected in keys, f"Missing key: {expected}"
