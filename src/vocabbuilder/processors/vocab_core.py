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

from functools import lru_cache
from typing import Iterator

from spacy.tokens import Doc, Token

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import ProcessedPassage, TokenInfo, VocabEntry, VocabList
from vocabbuilder.data.lemma_overrides import resolve_lemma_override
from vocabbuilder.processors.deduplicator import Deduplicator
from vocabbuilder.utils.normalization import to_v_form


def _effective_lemma(token: Token, morph: dict[str, str], config: PipelineConfig) -> str:
    """``token.lemma_``, corrected via ``config.lemma_overrides`` if a rule fires."""
    corrected = resolve_lemma_override(token.lemma_, token.pos_, morph, config.lemma_overrides)
    return corrected or token.lemma_


@lru_cache(maxsize=1)
def _lexicon_dict() -> dict[str, list[dict]] | None:
    """latincy-lexicon's full lemma->entries dict, loaded once per process.

    Only used when an override actually fires (see ``build_vocab_list``): the
    corrected lemma needs a fresh dictionary lookup because ``token._.lexicon``
    was ranked by latincy-lexicon's ``whitakers_words`` pipe against the
    ORIGINAL, uncorrected lemma and cannot be reused for the corrected one.
    """
    try:
        from latincy_lexicon import build_lexicon

        return build_lexicon()
    except Exception:
        return None


def _token_gloss(token: Token) -> str | None:
    """The upstream Whitaker's Words gloss on a token, or ``None``.

    Reads ``token._.gloss`` (set by latincy-lexicon's ``whitakers_words`` pipe)
    when the extension exists, normalizing an empty string to ``None``.
    """
    if Token.has_extension("gloss"):
        return token._.gloss or None
    return None


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

    Drops whitespace tokens, excluded POS, and standalone enclitics left behind
    by tokenization (``populusque`` → ``populus`` + ``que``).

    PROPN is excluded by default so bare proper names route to a separate NER/NEL
    channel — but a PROPN that Whitaker's Words actually glosses (``Musa`` →
    "muse", ``Roma`` → "Rome") is a real lexical item, not a bare name, so it is
    rescued into the list when ``config.keep_glossed_propn`` is set (the default).
    An unglossed PROPN (a name WW does not cover, e.g. ``Aquitani``) still drops.
    Only PROPN is gloss-rescued; ``PUNCT``/``X`` are always dropped.
    """
    if token.is_space:
        return False
    if token.pos_ in config.exclude_pos:
        if not (
            config.keep_glossed_propn
            and token.pos_ == "PROPN"
            and _token_gloss(token)
        ):
            return False
    if config.drop_enclitics and token.lemma_ in config.enclitic_lemmas:
        return False
    return True


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
    tokens: list[TokenInfo] = []
    for sent_idx, span in _sentence_index(doc):
        for token in span:
            if not _keep(token, config):
                continue
            morph = _parse_morph(token.morph)
            tokens.append(
                TokenInfo(
                    text=token.text,
                    lemma=_effective_lemma(token, morph, config),
                    pos=token.pos_,
                    morph=morph,
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
    groups: dict[tuple[str, str], VocabEntry] = {}

    for sent_idx, span in _sentence_index(doc):
        for token in span:
            if not _keep(token, config):
                continue
            morph = _parse_morph(token.morph)
            lemma = _effective_lemma(token, morph, config)
            gloss = token._.gloss if has_gloss else None
            if lemma != token.lemma_:
                # Override fired: token._.lexicon was ranked against the OLD
                # lemma by upstream whitakers_words and cannot be reused --
                # re-look-up gloss + citation entry against the corrected
                # lemma, for both the new-group and merge branches below.
                lex_dict = _lexicon_dict()
                lexicon = lex_dict.get(lemma) if lex_dict else None
                if lexicon:
                    gloss = "; ".join(lexicon[0].get("glosses", [])) or gloss
            else:
                lexicon = token._.lexicon if has_lexicon else None
            key = (lemma, token.pos_)
            if key not in groups:
                groups[key] = VocabEntry(
                    lemma=lemma,
                    display_lemma="",
                    pos=token.pos_,
                    glosses=[gloss] if gloss else [],
                    forms_seen={token.text},
                    frequency=1,
                    morphology=[morph] if morph else [],
                    passage_indices=[sent_idx],
                    first_index=len(groups),  # rank of first appearance
                    citation_form=_format_citation(lexicon, lemma),
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
    # ``has_gloss`` (a gloss pipe registered ``token._.gloss``) is the signal that
    # gloss-less entries are coverage gaps rather than the intended lexicon-free
    # output — so rendered views hide them by default. See VocabList.missing_gloss.
    return VocabList(entries=entries, glosses_expected=has_gloss)
