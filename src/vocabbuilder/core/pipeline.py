"""Main vocabulary pipeline orchestrator."""

from __future__ import annotations

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.processors.passage import PassageProcessor
from vocabbuilder.processors.vocab_core import build_vocab_list


class VocabPipeline:
    """Convenience wrapper: text → spaCy (+ ``whitakers_words``) → VocabList.

    Owns the spaCy model and shares the Doc-pure
    :func:`~vocabbuilder.processors.vocab_core.build_vocab_list` core with the
    ``latincy_vocab`` component. Glosses and POS-aware citation forms come from
    latincy-lexicon's ``whitakers_words`` pipe, which
    :class:`~vocabbuilder.processors.passage.PassageProcessor` appends to the
    model when ``config.use_glosses`` is set (the default) — so a pip-installed
    user gets them with no prebuilt data on disk. With ``use_glosses=False`` the
    pipeline runs lexicon-free (display lemmas via u→v only, no glosses or
    citation forms).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config if config is not None else PipelineConfig()
        self._passage_processor = PassageProcessor(self._config)

    def process(self, text: str) -> VocabList:
        """Process Latin text into a vocabulary list."""
        doc = self._passage_processor.nlp(text)
        return build_vocab_list(doc, self._config)
