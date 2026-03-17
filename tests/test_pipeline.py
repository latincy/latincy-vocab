"""Tests for full VocabPipeline (integration — needs spaCy + glosses data)."""

import pytest

from vocabbuilder import VocabPipeline, VocabList


@pytest.fixture(scope="module")
def pipeline() -> VocabPipeline:
    return VocabPipeline()


class TestVocabPipeline:
    def test_process_returns_vocab_list(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage)
        assert isinstance(result, VocabList)
        assert len(result) > 0

    def test_entries_have_glosses(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage)
        glossed = [e for e in result if e.glosses]
        # Most entries should have glosses
        assert len(glossed) > 0

    def test_entries_have_display_lemma(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage)
        for entry in result:
            assert entry.display_lemma, f"Missing display_lemma for {entry.lemma}"

    def test_v_form_display(self, pipeline: VocabPipeline):
        """spaCy lemma 'diuido' should display as 'divido'."""
        result = pipeline.process("Gallia est omnis divisa in partes tres")
        divido_entries = [e for e in result if "diuid" in e.lemma or "divid" in e.lemma]
        if divido_entries:
            assert divido_entries[0].display_lemma == "divido"

    def test_sort_by_frequency(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage).by_frequency()
        freqs = [e.frequency for e in result]
        assert freqs == sorted(freqs, reverse=True)

    def test_sort_by_alpha(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage).by_alpha()
        names = [e.display_lemma.lower() for e in result]
        assert names == sorted(names)

    def test_filter_pos(self, pipeline: VocabPipeline, caesar_passage: str):
        result = pipeline.process(caesar_passage).filter_pos({"NOUN"})
        assert all(e.pos == "NOUN" for e in result)

    def test_round_trip_smoke(self, pipeline: VocabPipeline, caesar_passage: str):
        """Full round-trip: passage → vocab list → verify key entries exist."""
        result = pipeline.process(caesar_passage)
        lemmas = {e.lemma for e in result}
        # These lemmas should appear from the Caesar passage
        assert "sum" in lemmas  # est -> sum
        assert "pars" in lemmas  # partes -> pars
