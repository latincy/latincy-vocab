"""Main vocabulary pipeline orchestrator."""

from __future__ import annotations

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.data.gloss_provider import GlossProvider
from vocabbuilder.processors.passage import PassageProcessor
from vocabbuilder.processors.vocab_core import build_vocab_list


class VocabPipeline:
    """Convenience wrapper: text → spaCy → ``build_vocab_list`` → (glosses) → VocabList.

    Owns the spaCy model and shares the Doc-pure
    :func:`~vocabbuilder.processors.vocab_core.build_vocab_list` core with the
    ``latincy_vocab`` component. When gloss data resolves, dictionary glosses are
    attached from the latincy-words :class:`GlossProvider`; otherwise the pipeline
    runs lexicon-free (display lemmas via u→v only, no glosses).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        if config is None:
            config = PipelineConfig()
        config.resolve_data_paths()

        self._config = config
        self._passage_processor = PassageProcessor(config)
        # Glosses are optional. With no gloss data resolved, the pipeline runs
        # lexicon-free and entries carry display lemmas but no dictionary glosses.
        self._gloss_provider = (
            GlossProvider(config.glosses_path) if config.glosses_path else None
        )

    def process(self, text: str) -> VocabList:
        """Process Latin text into a vocabulary list."""
        doc = self._passage_processor.nlp(text)
        vocab_list = build_vocab_list(doc, self._config)
        if self._gloss_provider is None:
            return vocab_list

        # Attach dictionary glosses + curated display lemmas from latincy-words.
        for entry in vocab_list:
            result = self._gloss_provider.lookup(
                entry.lemma, entry.pos, self._config.max_glosses
            )
            if result:
                entry.glosses = result.glosses
                entry.display_lemma = result.display_lemma
            else:
                entry.display_lemma = self._gloss_provider.get_display_lemma(entry.lemma)

        return vocab_list
