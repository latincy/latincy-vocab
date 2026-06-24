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

    # Standalone enclitics left behind by tokenization (e.g. 'populusque' →
    # 'populus' + 'que') are dropped; the host word is already lemmatized.
    drop_enclitics: bool = True
    enclitic_lemmas: set[str] = field(default_factory=lambda: {"que"})
