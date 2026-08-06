"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from vocabbuilder.data.lemma_overrides import DEFAULT_LEMMA_OVERRIDES, LemmaOverrideRule


@dataclass
class PipelineConfig:
    """Configuration for the vocabulary pipeline."""

    spacy_model: str = "la_core_web_lg"
    spacy_disable: list[str] = field(default_factory=lambda: ["lookup_lemmatizer"])

    # Glosses + POS-aware citation forms come from latincy-lexicon's
    # ``whitakers_words`` pipe, which PassageProcessor appends to the model when
    # this is True. False → lexicon-free: display lemmas via u→v only, no
    # dictionary glosses or citation forms.
    use_glosses: bool = True
    min_frequency: int = 1
    # PROPN is excluded by design: proper names route to a separate NER/NEL
    # channel, not the vocabulary list. Whitespace tokens are dropped
    # structurally (token.is_space), not via a pseudo-POS tag.
    exclude_pos: set[str] = field(default_factory=lambda: {"PROPN", "PUNCT", "X"})

    # Rescue a PROPN token into the vocabulary list when Whitaker's Words
    # actually glosses its lemma. The model tags some real vocabulary as PROPN
    # because it doubles as a name — ``Musa`` (the Muse) vs. the noun ``musa,
    # musae, f.`` "muse" — and the blanket PROPN exclusion above then drops it. A
    # WW gloss is the signal that the token is a genuine lexical item rather than
    # a bare name: a glossed PROPN (``Musa``, ``Roma``, ``Gallia``) is kept, an
    # unglossed one (``Aquitani``, ``Celtae``) still routes to NER/NEL. Set False
    # to restore strict drop-all-PROPN behavior.
    keep_glossed_propn: bool = True

    # Standalone enclitics left behind by tokenization (e.g. 'populusque' →
    # 'populus' + 'que') are dropped; the host word is already lemmatized.
    drop_enclitics: bool = True
    enclitic_lemmas: set[str] = field(default_factory=lambda: {"que"})

    # Curated corrections for known spaCy/LatinCy homograph mis-lemmas (e.g.
    # 'latus' the fero-participle mislemmatized as the noun 'latus, lateris').
    # See vocabbuilder.data.lemma_overrides for the rule format and the
    # current small, hand-curated rule set. Pass () to disable entirely.
    lemma_overrides: tuple[LemmaOverrideRule, ...] = DEFAULT_LEMMA_OVERRIDES
