"""Main vocabulary pipeline orchestrator."""

from __future__ import annotations

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.data.gloss_provider import GlossProvider
from vocabbuilder.processors.deduplicator import Deduplicator
from vocabbuilder.processors.passage import PassageProcessor


class VocabPipeline:
    """Orchestrates: text → spaCy → extract → deduplicate → gloss → VocabList."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        if config is None:
            config = PipelineConfig()
        config.resolve_data_paths()

        self._config = config
        self._passage_processor = PassageProcessor(config)
        self._deduplicator = Deduplicator()
        self._gloss_provider = GlossProvider(config.glosses_path)

    def process(self, text: str) -> VocabList:
        """Process Latin text into a vocabulary list with glosses.

        Pipeline: spaCy NLP → token extraction → deduplication → gloss attachment.
        """
        # 1. Run spaCy
        passage = self._passage_processor.process(text)

        # 2. Extract vocab entries grouped by (lemma, pos)
        entries = self._passage_processor.extract_vocab(passage)

        # 3. Deduplicate (merges same lemma+pos from different contexts)
        entries = self._deduplicator.deduplicate(entries)

        # 4. Attach glosses and display lemmas
        for entry in entries:
            result = self._gloss_provider.lookup(
                entry.lemma, entry.pos, self._config.max_glosses
            )
            if result:
                entry.glosses = result.glosses
                entry.display_lemma = result.display_lemma
            else:
                entry.display_lemma = self._gloss_provider.get_display_lemma(entry.lemma)

        return VocabList(entries=entries)
