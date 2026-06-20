"""The pipeline must produce a structural vocab list with no lexicon contact.

`use_glosses=False` runs the lexicon-free path: tokenize → lemmatize → extract →
dedup → order, with display lemmas (u→v normalization only) but no dictionary
glosses. This is the backbone that works without latincy-words / latincy-lexicon.
"""

import pytest

from vocabbuilder import PipelineConfig, VocabList, VocabPipeline


@pytest.fixture(scope="module")
def lexicon_free_pipeline() -> VocabPipeline:
    return VocabPipeline(PipelineConfig(use_glosses=False))


class TestLexiconFreePipeline:
    def test_returns_vocab_list(self, lexicon_free_pipeline: VocabPipeline, caesar_passage: str):
        result = lexicon_free_pipeline.process(caesar_passage)
        assert isinstance(result, VocabList)
        assert len(result) > 0

    def test_no_glosses_attached(self, lexicon_free_pipeline: VocabPipeline, caesar_passage: str):
        result = lexicon_free_pipeline.process(caesar_passage)
        assert all(entry.glosses == [] for entry in result)

    def test_display_lemma_still_set(self, lexicon_free_pipeline: VocabPipeline, caesar_passage: str):
        """Even without glosses, every entry has a v-form display lemma."""
        result = lexicon_free_pipeline.process(caesar_passage)
        for entry in result:
            assert entry.display_lemma, f"missing display_lemma for {entry.lemma}"
