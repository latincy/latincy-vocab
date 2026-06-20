"""Tests for PassageProcessor (integration — needs spaCy model)."""

import pytest

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.processors.passage import PassageProcessor


@pytest.fixture(scope="module")
def processor() -> PassageProcessor:
    return PassageProcessor(PipelineConfig())


class TestPassageProcessor:
    def test_process_returns_tokens(self, processor: PassageProcessor, caesar_passage: str):
        result = processor.process(caesar_passage)
        assert len(result.tokens) > 0
        assert result.text == caesar_passage

    def test_process_returns_sentences(self, processor: PassageProcessor, caesar_passage: str):
        result = processor.process(caesar_passage)
        assert len(result.sentences) >= 1

    def test_tokens_have_lemma_and_pos(self, processor: PassageProcessor, caesar_passage: str):
        result = processor.process(caesar_passage)
        for token in result.tokens:
            assert token.lemma
            assert token.pos

    def test_punct_excluded(self, processor: PassageProcessor, caesar_passage: str):
        result = processor.process(caesar_passage)
        assert not any(t.pos == "PUNCT" for t in result.tokens)

    def test_propn_excluded(self, processor: PassageProcessor, caesar_passage: str):
        """Proper names route to the NER/NEL channel, not the vocab list."""
        result = processor.process(caesar_passage)
        assert not any(t.pos == "PROPN" for t in result.tokens)

    def test_enclitic_que_dropped(self, processor: PassageProcessor):
        """A split enclitic '-que' must not surface as its own 'que' entry."""
        result = processor.process("senatus populusque")
        lemmas = {t.lemma for t in result.tokens}
        assert "que" not in lemmas

    def test_known_lemma_present(self, processor: PassageProcessor, caesar_passage: str):
        """'est' should lemmatize to 'sum'."""
        result = processor.process(caesar_passage)
        lemmas = {t.lemma for t in result.tokens}
        assert "sum" in lemmas

    def test_extract_vocab(self, processor: PassageProcessor, caesar_passage: str):
        passage = processor.process(caesar_passage)
        entries = processor.extract_vocab(passage)
        assert len(entries) > 0
        # Each entry should have at least one form
        for entry in entries:
            assert len(entry.forms_seen) > 0
            assert entry.frequency >= 1

    def test_extract_vocab_groups_by_lemma_pos(self, processor: PassageProcessor):
        """Two occurrences of same lemma should be grouped."""
        result = processor.process("partes et partes")
        entries = processor.extract_vocab(result)
        pars_entries = [e for e in entries if e.lemma == "pars"]
        # Should be grouped into one entry
        if pars_entries:
            assert pars_entries[0].frequency >= 2
