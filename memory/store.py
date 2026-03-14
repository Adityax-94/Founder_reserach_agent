"""
Simple key-value memory store with:
  - In-process dict for speed
  - Optional JSON file persistence between runs
  - Namespace support (one namespace per research target)
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    A lightweight persistent memory store.

    Usage:
        store = MemoryStore(persist_path="output/memory.json")
        store.upsert("key", {"fact": "..."}, namespace="Elon Musk")
        all_facts = store.get_all(namespace="Elon Musk")
    """

    def __init__(self, persist_path: str = "output/memory.json"):
        self._path = persist_path
        # {namespace: {key: {value, updated_at}}}
        self._data: dict[str, dict[str, dict]] = defaultdict(dict)
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def upsert(self, key: str, value: Any, namespace: str = "default") -> None:
        """Insert or update a memory entry."""
        self._data[namespace][key] = {
            "value":      value,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        entry = self._data.get(namespace, {}).get(key)
        return entry["value"] if entry else None

    def get_all(self, namespace: str = "default") -> list[Any]:
        """Return all values for a namespace."""
        return [entry["value"] for entry in self._data.get(namespace, {}).values()]

    def list_namespaces(self) -> list[str]:
        return list(self._data.keys())

    def delete(self, key: str, namespace: str = "default") -> None:
        self._data.get(namespace, {}).pop(key, None)

    def clear_namespace(self, namespace: str) -> None:
        self._data.pop(namespace, None)

    def save(self) -> None:
        """Persist memory to disk."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(dict(self._data), f, indent=2, default=str)
        logger.debug("[memory] Saved to %s", self._path)

    def stats(self, namespace: str = "default") -> dict:
        ns = self._data.get(namespace, {})
        return {"namespace": namespace, "entry_count": len(ns)}

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for ns, entries in loaded.items():
                    self._data[ns].update(entries)
                logger.debug("[memory] Loaded from %s", self._path)
            except Exception as exc:
                logger.warning("[memory] Could not load memory file: %s", exc)

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())
