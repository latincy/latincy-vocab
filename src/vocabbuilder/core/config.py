"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    # Common nouns that the LatinCy model systematically mis-tags as PROPN
    # because they double as proper names — e.g. ``Musa`` (the Muse) vs. the
    # dictionary common noun ``musa, musae, f.`` "muse". Without this rescue such
    # a token would be swept up by the PROPN exclusion above and vanish from the
    # vocabulary list, even though it is a genuine vocabulary word. A lemma listed
    # here is treated as a NOUN when it arrives tagged PROPN, so it appears with
    # its citation form and noun formatting. Matching folds u/v/j and case
    # (``Musa``/``musa``/``MVSA`` all match); a token the model already tags NOUN
    # is unaffected. Extend the set to rescue additional lemmas.
    keep_propn_lemmas: set[str] = field(default_factory=lambda: {"musa"})

    # Standalone enclitics left behind by tokenization (e.g. 'populusque' →
    # 'populus' + 'que') are dropped; the host word is already lemmatized.
    drop_enclitics: bool = True
    enclitic_lemmas: set[str] = field(default_factory=lambda: {"que"})
