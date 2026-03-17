"""Passage processing with spaCy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import spacy

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import ProcessedPassage, TokenInfo, VocabEntry

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
        """Run spaCy on text and return structured passage data."""
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]
        tokens = []
        for sent_idx, sent in enumerate(doc.sents):
            for token in sent:
                if token.pos_ in self._config.exclude_pos:
                    continue
                morph = {
                    k: v
                    for k, v in (
                        feat.split("=") for feat in str(token.morph).split("|") if "=" in feat
                    )
                }
                tokens.append(
                    TokenInfo(
                        text=token.text,
                        lemma=token.lemma_,
                        pos=token.pos_,
                        morph=morph,
                        sent_idx=sent_idx,
                    )
                )

        return ProcessedPassage(text=text, tokens=tokens, sentences=sentences)

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
