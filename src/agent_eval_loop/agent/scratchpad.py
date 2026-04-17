"""Scratchpad: persistent cross-turn working memory for agents.

The scratchpad gives the agent a place to store structured state
across turns — extracted entities, conversation summary, current
task status, etc. It's injected into the system prompt each turn.
"""

from __future__ import annotations

from typing import Any


class Scratchpad:
    """Key-value working memory that persists across conversation turns.

    Design principle: the scratchpad is append-friendly and read-cheap.
    The agent can write structured notes that accumulate across turns.
    On each turn, the full scratchpad is rendered into the context window.

    For long conversations, implement compaction (summarize older entries)
    to manage context window usage.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []  # audit trail

    def set(self, key: str, value: Any) -> None:
        """Set a scratchpad entry. Overwrites if key exists."""
        old = self._entries.get(key)
        self._entries[key] = value
        self._history.append({
            "action": "set",
            "key": key,
            "old_value": old,
            "new_value": value,
        })

    def get(self, key: str, default: Any = None) -> Any:
        """Get a scratchpad entry."""
        return self._entries.get(key, default)

    def append_to(self, key: str, value: Any) -> None:
        """Append to a list-type entry. Creates the list if it doesn't exist."""
        if key not in self._entries:
            self._entries[key] = []
        if not isinstance(self._entries[key], list):
            self._entries[key] = [self._entries[key]]
        self._entries[key].append(value)

    def render(self) -> str:
        """Render scratchpad contents for injection into the system prompt.

        Returns empty string if scratchpad is empty (so we don't waste tokens).
        """
        if not self._entries:
            return ""

        lines = []
        for key, value in self._entries.items():
            if isinstance(value, list):
                items = "\n".join(f"  - {item}" for item in value)
                lines.append(f"{key}:\n{items}")
            elif isinstance(value, dict):
                import json
                lines.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._history.clear()

    def compact(self, summarizer=None) -> None:
        """Compact the scratchpad to reduce token usage.

        If a summarizer function is provided, it will be called with
        the current entries and should return a condensed version.
        Otherwise, only keeps the most recent value for each key.
        """
        if summarizer:
            self._entries = summarizer(self._entries)
        # Default: no-op (entries are already deduplicated by key)

    @property
    def entries(self) -> dict[str, Any]:
        return self._entries.copy()

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history.copy()
