"""Tests for core data models."""

from vocabbuilder.core.models import VocabEntry, VocabList


class TestVocabEntry:
    def test_key(self):
        entry = VocabEntry(lemma="bellum", display_lemma="bellum", pos="NOUN")
        assert entry.key == ("bellum", "NOUN")

    def test_merge(self):
        e1 = VocabEntry(
            lemma="pars",
            display_lemma="pars",
            pos="NOUN",
            forms_seen={"partes"},
            frequency=2,
            morphology=[{"Case": "Acc"}],
            passage_indices=[0],
        )
        e2 = VocabEntry(
            lemma="pars",
            display_lemma="pars",
            pos="NOUN",
            forms_seen={"partis"},
            frequency=1,
            morphology=[{"Case": "Gen"}],
            passage_indices=[1],
        )
        e1.merge(e2)
        assert e1.frequency == 3
        assert e1.forms_seen == {"partes", "partis"}
        assert len(e1.morphology) == 2
        assert e1.passage_indices == [0, 1]

    def test_merge_fills_empty_glosses(self):
        e1 = VocabEntry(lemma="sum", display_lemma="sum", pos="VERB", glosses=[])
        e2 = VocabEntry(lemma="sum", display_lemma="sum", pos="VERB", glosses=["to be"])
        e1.merge(e2)
        assert e1.glosses == ["to be"]


class TestVocabList:
    def _make_entries(self):
        return [
            VocabEntry(lemma="bellum", display_lemma="bellum", pos="NOUN", frequency=3),
            VocabEntry(lemma="ager", display_lemma="ager", pos="NOUN", frequency=1),
            VocabEntry(lemma="capio", display_lemma="capio", pos="VERB", frequency=5),
        ]

    def test_len_and_iter(self):
        vl = VocabList(entries=self._make_entries())
        assert len(vl) == 3
        assert [e.lemma for e in vl] == ["bellum", "ager", "capio"]

    def test_getitem(self):
        vl = VocabList(entries=self._make_entries())
        assert vl[1].lemma == "ager"

    def test_by_frequency(self):
        vl = VocabList(entries=self._make_entries()).by_frequency()
        assert [e.frequency for e in vl] == [5, 3, 1]

    def test_by_alpha(self):
        vl = VocabList(entries=self._make_entries()).by_alpha()
        assert [e.display_lemma for e in vl] == ["ager", "bellum", "capio"]

    def test_filter_pos(self):
        vl = VocabList(entries=self._make_entries()).filter_pos({"NOUN"})
        assert len(vl) == 2
        assert all(e.pos == "NOUN" for e in vl)

    def test_filter_min_frequency(self):
        vl = VocabList(entries=self._make_entries()).filter_min_frequency(3)
        assert len(vl) == 2
        assert all(e.frequency >= 3 for e in vl)
