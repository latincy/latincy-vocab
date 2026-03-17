"""Deduplication of vocabulary entries."""

from __future__ import annotations

from vocabbuilder.core.models import VocabEntry


class Deduplicator:
    """Merge VocabEntry instances that share the same lemma+POS."""

    def deduplicate(self, entries: list[VocabEntry]) -> list[VocabEntry]:
        """Merge entries with the same (lemma, pos) key."""
        merged: dict[tuple[str, str], VocabEntry] = {}

        for entry in entries:
            if entry.key in merged:
                merged[entry.key].merge(entry)
            else:
                merged[entry.key] = entry

        return list(merged.values())
