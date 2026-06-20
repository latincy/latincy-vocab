"""Passage processing with spaCy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import spacy

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import ProcessedPassage, VocabEntry
from vocabbuilder.processors.vocab_core import passage_from_doc

if TYPE_CHECKING:
    from spacy.language import Language


class PassageProcessor:
    """Process Latin text through spaCy to extract token annotations."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = spacy.load(
                self._config.spacy_model,
                disable=self._config.spacy_disable,
            )
        return self._nlp

    def process(self, text: str) -> ProcessedPassage:
        """Run spaCy on text and return structured passage data.

        The model run lives here; the Doc → ``ProcessedPassage`` mapping is the
        shared Doc-pure :func:`~vocabbuilder.processors.vocab_core.passage_from_doc`
        so the spaCy component reuses the identical filtering rules.
        """
        doc = self.nlp(text)
        passage = passage_from_doc(doc, self._config)
        # Preserve the caller's original text (doc.text is spaCy-normalized).
        passage.text = text
        return passage

    def extract_vocab(self, passage: ProcessedPassage) -> list[VocabEntry]:
        """Group tokens by (lemma, pos) into VocabEntry list."""
        groups: dict[tuple[str, str], VocabEntry] = {}

        for token in passage.tokens:
            key = (token.lemma, token.pos)
            if key not in groups:
                groups[key] = VocabEntry(
                    lemma=token.lemma,
                    display_lemma="",  # filled in by pipeline with GlossProvider
                    pos=token.pos,
                    forms_seen={token.text},
                    frequency=1,
                    morphology=[token.morph] if token.morph else [],
                    passage_indices=[token.sent_idx],
                )
            else:
                entry = groups[key]
                entry.forms_seen.add(token.text)
                entry.frequency += 1
                if token.morph:
                    entry.morphology.append(token.morph)
                entry.passage_indices.append(token.sent_idx)

        return list(groups.values())
