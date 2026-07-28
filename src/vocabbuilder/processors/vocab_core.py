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
from vocabbuilder.utils.normalization import to_u_form, to_v_form


def _fold_lemma(text: str) -> str:
    """Case- and u/v/j-folded lemma key for matching config lemma sets."""
    return to_u_form(text).lower()


def _effective_pos(token: Token, keep_propn_folded: frozenset[str]) -> str:
    """The POS this token contributes under, rescuing mis-tagged PROPN nouns.

    The LatinCy model tags some common nouns as PROPN because they double as
    proper names (``Musa`` the Muse vs. common ``musa, musae, f.``). A lemma in
    ``keep_propn_folded`` (already u/v/j+case-folded) is reclassified PROPN→NOUN
    so it survives the PROPN exclusion and flows through the noun formatting path
    (citation form, empty pos-marker) instead of being dropped as a proper name.
    """
    if token.pos_ == "PROPN" and _fold_lemma(token.lemma_) in keep_propn_folded:
        return "NOUN"
    return token.pos_


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


def _keep(token: Token, config: PipelineConfig, effective_pos: str) -> bool:
    """A token contributes to the vocab list iff it is a content word.

    Drops whitespace tokens, excluded POS (PROPN routes to NER/NEL, unless the
    lemma is rescued via ``keep_propn_lemmas`` — see :func:`_effective_pos`), and
    standalone enclitics left behind by tokenization (``populusque`` →
    ``populus`` + ``que``). ``effective_pos`` is the rescued-aware POS, so a
    PROPN token whose lemma is rescued arrives here as NOUN and is kept.
    """
    if token.is_space:
        return False
    if effective_pos in config.exclude_pos:
        return False
    if config.drop_enclitics and token.lemma_ in config.enclitic_lemmas:
        return False
    return True


def _keep_propn_folded(config: PipelineConfig) -> frozenset[str]:
    """The config's PROPN-rescue lemma set, u/v/j+case-folded for matching."""
    return frozenset(_fold_lemma(w) for w in config.keep_propn_lemmas)


def _resolve_display_lemma(entry: VocabEntry, config: PipelineConfig) -> str:
    """The fallback display headword (u→v normalization). Used when no upstream
    lexicon citation form is available."""
    return to_v_form(entry.lemma)


def _format_citation(lexicon, lemma: str | None = None) -> str | None:
    """Textbook citation form for a token.

    Prefers a lemma-keyed pronominal citation (demonstratives/relatives, which
    the lexicon stores inconsistently — ``hic`` arrives as PRON headword ``"h"``),
    then falls back to ``format_principal_parts`` on the top-ranked lexicon entry.
    Returns ``None`` when nothing applies or the library is unavailable.
    """
    try:
        from latincy_lexicon import format_principal_parts, pronominal_citation
    except Exception:
        return None
    try:
        pron = pronominal_citation(lemma)
        if pron:
            return pron
    except Exception:
        pass
    if not lexicon:
        return None
    try:
        return format_principal_parts(lexicon[0])
    except Exception:
        return None


def passage_from_doc(doc: Doc, config: PipelineConfig) -> ProcessedPassage:
    """The Doc-pure half of ``PassageProcessor.process`` (no model run)."""
    sentences = [span.text for _idx, span in _sentence_index(doc)]
    keep_propn = _keep_propn_folded(config)
    tokens: list[TokenInfo] = []
    for sent_idx, span in _sentence_index(doc):
        for token in span:
            pos = _effective_pos(token, keep_propn)
            if not _keep(token, config, pos):
                continue
            tokens.append(
                TokenInfo(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=pos,
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
    has_lexicon = Token.has_extension("lexicon")
    keep_propn = _keep_propn_folded(config)
    groups: dict[tuple[str, str], VocabEntry] = {}

    for sent_idx, span in _sentence_index(doc):
        for token in span:
            pos = _effective_pos(token, keep_propn)
            if not _keep(token, config, pos):
                continue
            morph = _parse_morph(token.morph)
            gloss = token._.gloss if has_gloss else None
            key = (token.lemma_, pos)
            if key not in groups:
                lexicon = token._.lexicon if has_lexicon else None
                groups[key] = VocabEntry(
                    lemma=token.lemma_,
                    display_lemma="",
                    pos=pos,
                    glosses=[gloss] if gloss else [],
                    forms_seen={token.text},
                    frequency=1,
                    morphology=[morph] if morph else [],
                    passage_indices=[sent_idx],
                    first_index=len(groups),  # rank of first appearance
                    citation_form=_format_citation(lexicon, token.lemma_),
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
