"""Data models for vocabulary building."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from vocabbuilder.utils.normalization import upos_to_abbrev


@dataclass
class TokenInfo:
    """Per-token annotation from spaCy."""

    text: str
    lemma: str
    pos: str
    morph: dict[str, str] = field(default_factory=dict)
    sent_idx: int = 0


@dataclass
class ProcessedPassage:
    """A passage after spaCy processing."""

    text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)


@dataclass
class VocabEntry:
    """A single vocabulary entry."""

    lemma: str
    display_lemma: str
    pos: str
    glosses: list[str] = field(default_factory=list)
    forms_seen: set[str] = field(default_factory=set)
    frequency: int = 1
    morphology: list[dict[str, str]] = field(default_factory=list)
    passage_indices: list[int] = field(default_factory=list)
    first_index: int = 0  # rank of first appearance in the passage (0-based)
    # Textbook citation form from latincy-lexicon (principal parts / gen+gender /
    # -a,-um), when an upstream lexicon pipe set ``token._.lexicon``. None → fall
    # back to ``display_lemma``.
    citation_form: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Unique key for deduplication: (lemma, pos)."""
        return (self.lemma, self.pos)

    @property
    def headword(self) -> str:
        """Citation form if available, else the v-form display lemma."""
        return self.citation_form or self.display_lemma

    @property
    def pos_marker(self) -> str:
        """POS tag for display. Empty when the citation form already carries a
        gender (noun-shaped, e.g. ``amicus, amici, m.``) — covers true nouns and
        words the lexicon mistypes as nouns (possessives), so we never emit a
        contradictory ``m., adj.``."""
        if self.pos == "NOUN":
            return ""
        if (self.citation_form or "").rstrip().endswith(("m.", "f.", "n.")):
            return ""
        return upos_to_abbrev(self.pos)

    def formatted(self) -> str:
        """One-line glossary entry: ``headword, marker, gloss`` (empties omitted)."""
        gloss = "; ".join(self.glosses)
        return ", ".join(p for p in (self.headword, self.pos_marker, gloss) if p)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (``forms_seen`` set → sorted list)."""
        return {
            "lemma": self.lemma,
            "display_lemma": self.display_lemma,
            "citation_form": self.citation_form,
            "pos": self.pos,
            "glosses": self.glosses,
            "forms_seen": sorted(self.forms_seen),
            "frequency": self.frequency,
            "morphology": self.morphology,
            "passage_indices": self.passage_indices,
            "first_index": self.first_index,
        }

    def merge(self, other: VocabEntry) -> None:
        """Merge another entry (same lemma+pos) into this one."""
        self.forms_seen |= other.forms_seen
        self.frequency += other.frequency
        self.morphology.extend(other.morphology)
        self.passage_indices.extend(other.passage_indices)
        if not self.glosses and other.glosses:
            self.glosses = other.glosses
        if not self.display_lemma and other.display_lemma:
            self.display_lemma = other.display_lemma


@dataclass
class VocabList:
    """Collection of vocabulary entries with filtering and sorting."""

    entries: list[VocabEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VocabEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> VocabEntry:
        return self.entries[index]

    def by_frequency(self, descending: bool = True) -> VocabList:
        """Return a new VocabList sorted by frequency."""
        sorted_entries = sorted(self.entries, key=lambda e: e.frequency, reverse=descending)
        return VocabList(entries=sorted_entries)

    def by_alpha(self) -> VocabList:
        """Return a new VocabList sorted alphabetically by display_lemma."""
        sorted_entries = sorted(self.entries, key=lambda e: e.display_lemma.lower())
        return VocabList(entries=sorted_entries)

    def by_first_occurrence(self) -> VocabList:
        """Return a new VocabList in passage reading order (first appearance)."""
        sorted_entries = sorted(self.entries, key=lambda e: e.first_index)
        return VocabList(entries=sorted_entries)

    def to_dicts(self) -> list[dict[str, Any]]:
        """JSON-safe list of entry dicts."""
        return [e.to_dict() for e in self.entries]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the list to a JSON string."""
        return json.dumps(self.to_dicts(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Render a Markdown glossary: ``- **headword**, marker, gloss``."""
        lines = []
        for e in self.entries:
            tail = ", ".join(p for p in (e.pos_marker, "; ".join(e.glosses)) if p)
            suffix = f", {tail}" if tail else ""
            lines.append(f"- **{e.headword}**{suffix}")
        return "\n".join(lines)

    def filter_pos(self, pos_tags: set[str]) -> VocabList:
        """Return entries matching the given POS tags."""
        return VocabList(entries=[e for e in self.entries if e.pos in pos_tags])

    def filter_min_frequency(self, min_freq: int) -> VocabList:
        """Return entries with frequency >= min_freq."""
        return VocabList(entries=[e for e in self.entries if e.frequency >= min_freq])
