"""Doc-pure core for vocabulary-list building.

Shared by both :class:`~vocabbuilder.processors.passage.PassageProcessor` (text
path) and the ``latincy_vocab`` spaCy component (Doc path). This module never
loads a spaCy model and never touches gloss/lexicon files: it consumes an
already-parsed ``Doc`` and the token annotations upstream pipes have set
(``token._.gloss`` from latincy-lexicon's ``whitakers_words``, and later
citation forms). Imports stay limited to models/config/normalization so there is
no import cycle with ``pipeline``/``component``.
"""

from __future__ import annotations

from typing import Iterator

from spacy.tokens import Doc, Token

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import ProcessedPassage, TokenInfo, VocabEntry, VocabList
from vocabbuilder.processors.deduplicator import Deduplicator
from vocabbuilder.utils.normalization import to_v_form


def _parse_morph(morph) -> dict[str, str]:
    """Parse spaCy ``str(token.morph)`` (``Case=Nom|Number=Sing``) into a dict."""
    return {
        k: v
        for k, v in (
            feat.split("=") for feat in str(morph).split("|") if "=" in feat
        )
    }


def _sentence_index(doc: Doc) -> Iterator[tuple[int, "spacy.tokens.Span | Doc"]]:
    """Yield ``(sent_idx, span)`` pairs, falling back to the whole doc.

    A hand-built or senter-less ``Doc`` has no ``SENT_START`` annotation, so
    ``doc.sents`` would raise; in that case treat the doc as a single sentence.
    """
    if doc.has_annotation("SENT_START"):
        yield from enumerate(doc.sents)
    else:
        yield 0, doc


def _keep(token: Token, config: PipelineConfig) -> bool:
    """A token contributes to the vocab list iff it is a content word.

    Drops excluded POS (PROPN routes to NER/NEL) and standalone enclitics left
    behind by tokenization (``populusque`` → ``populus`` + ``que``).
    """
    if token.pos_ in config.exclude_pos:
        return False
    if config.drop_enclitics and token.lemma_ in config.enclitic_lemmas:
        return False
    return True


def _resolve_display_lemma(entry: VocabEntry, config: PipelineConfig) -> str:
    """The display headword. Today this is u→v normalization only; the
    in-progress latincy-lexicon citation-form formatter plugs in here later.
    """
    return to_v_form(entry.lemma)


def passage_from_doc(doc: Doc, config: PipelineConfig) -> ProcessedPassage:
    """The Doc-pure half of ``PassageProcessor.process`` (no model run)."""
    sentences = [span.text for _idx, span in _sentence_index(doc)]
    tokens: list[TokenInfo] = []
    for sent_idx, span in _sentence_index(doc):
        for token in span:
            if not _keep(token, config):
                continue
            tokens.append(
                TokenInfo(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    morph=_parse_morph(token.morph),
                    sent_idx=sent_idx,
                )
            )
    return ProcessedPassage(text=doc.text, tokens=tokens, sentences=sentences)


def build_vocab_list(doc: Doc, config: PipelineConfig) -> VocabList:
    """Aggregate an already-parsed ``Doc`` into a :class:`VocabList`.

    Groups content tokens by ``(lemma, pos)`` in first-occurrence order, seeds
    glosses from ``token._.gloss`` when an upstream pipe provided it, dedups, and
    fills display lemmas. Runs no model and loads no files.
    """
    has_gloss = Token.has_extension("gloss")
    groups: dict[tuple[str, str], VocabEntry] = {}

    for sent_idx, span in _sentence_index(doc):
        for token in span:
            if not _keep(token, config):
                continue
            morph = _parse_morph(token.morph)
            gloss = token._.gloss if has_gloss else None
            key = (token.lemma_, token.pos_)
            if key not in groups:
                groups[key] = VocabEntry(
                    lemma=token.lemma_,
                    display_lemma="",
                    pos=token.pos_,
                    glosses=[gloss] if gloss else [],
                    forms_seen={token.text},
                    frequency=1,
                    morphology=[morph] if morph else [],
                    passage_indices=[sent_idx],
                    first_index=len(groups),  # rank of first appearance
                )
            else:
                entry = groups[key]
                entry.forms_seen.add(token.text)
                entry.frequency += 1
                if morph:
                    entry.morphology.append(morph)
                entry.passage_indices.append(sent_idx)
                if not entry.glosses and gloss:
                    entry.glosses = [gloss]

    entries = Deduplicator().deduplicate(list(groups.values()))
    for entry in entries:
        if not entry.display_lemma:
            entry.display_lemma = _resolve_display_lemma(entry, config)
    return VocabList(entries=entries)
