"""Model-free tests for the Doc-pure vocab core (no spaCy model needed).

Docs are hand-built with ``spacy.blank("la")`` and per-token ``pos_``/``lemma_``/
``_.gloss`` set directly, so these run without ``la_core_web_*``.
"""

from __future__ import annotations

import pytest
import spacy
from spacy.tokens import Doc, Token

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabList
from vocabbuilder.processors.vocab_core import build_vocab_list


@pytest.fixture(scope="module")
def blank_nlp():
    return spacy.blank("la")


def _ensure_gloss_ext() -> None:
    if not Token.has_extension("gloss"):
        Token.set_extension("gloss", default=None)


def make_doc(nlp, specs):
    """Build a Doc from (text, lemma, pos, gloss|None) tuples."""
    _ensure_gloss_ext()
    doc = Doc(nlp.vocab, words=[s[0] for s in specs])
    for tok, (_text, lemma, pos, gloss) in zip(doc, specs):
        tok.lemma_ = lemma
        tok.pos_ = pos
        if gloss is not None:
            tok._.gloss = gloss
    return doc


class TestBuildVocabList:
    def test_returns_vocab_list(self, blank_nlp):
        doc = make_doc(blank_nlp, [("toga", "toga", "NOUN", None)])
        result = build_vocab_list(doc, PipelineConfig())
        assert isinstance(result, VocabList)
        assert len(result) == 1

    def test_respects_exclude_pos(self, blank_nlp):
        doc = make_doc(blank_nlp, [
            ("Roma", "Roma", "PROPN", None),
            ("toga", "toga", "NOUN", None),
        ])
        lemmas = {e.lemma for e in build_vocab_list(doc, PipelineConfig())}
        assert "Roma" not in lemmas
        assert "toga" in lemmas

    def test_rescues_mistagged_propn_noun(self, blank_nlp):
        """A common noun mis-tagged PROPN (``musa``) is kept, not dropped."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", "muse")])
        vl = build_vocab_list(doc, PipelineConfig())
        assert [e.lemma for e in vl] == ["musa"]

    def test_rescued_propn_becomes_noun(self, blank_nlp):
        """The rescued token flows through the NOUN path (pos, empty marker)."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", "muse")])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.pos == "NOUN"
        assert entry.pos_marker == ""  # gender lives in the citation form

    def test_rescue_is_case_and_uv_folded(self, blank_nlp):
        """Matching folds case and u/v, so ``MVSA`` still rescues ``musa``."""
        doc = make_doc(blank_nlp, [("MVSA", "Musa", "PROPN", None)])
        vl = build_vocab_list(doc, PipelineConfig())
        assert len(vl) == 1

    def test_genuine_propn_still_excluded(self, blank_nlp):
        """A proper name not on the rescue list is still dropped."""
        doc = make_doc(blank_nlp, [
            ("Roma", "Roma", "PROPN", None),
            ("Musa", "musa", "PROPN", None),
        ])
        lemmas = {e.lemma for e in build_vocab_list(doc, PipelineConfig())}
        assert lemmas == {"musa"}

    def test_rescue_list_is_configurable(self, blank_nlp):
        """An empty keep_propn_lemmas restores the drop-all-PROPN behavior."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", None)])
        config = PipelineConfig(keep_propn_lemmas=set())
        assert len(build_vocab_list(doc, config)) == 0

    def test_rescued_and_noun_forms_merge(self, blank_nlp):
        """A rescued PROPN token and a NOUN token of the same lemma group as one."""
        doc = make_doc(blank_nlp, [
            ("Musa", "musa", "PROPN", None),
            ("musam", "musa", "NOUN", None),
        ])
        vl = build_vocab_list(doc, PipelineConfig())
        assert len(vl) == 1
        assert vl.entries[0].frequency == 2

    def test_enclitic_dropped(self, blank_nlp):
        doc = make_doc(blank_nlp, [
            ("populus", "populus", "NOUN", None),
            ("que", "que", "CCONJ", None),
        ])
        lemmas = {e.lemma for e in build_vocab_list(doc, PipelineConfig())}
        assert "que" not in lemmas

    def test_seeds_gloss_from_token(self, blank_nlp):
        doc = make_doc(blank_nlp, [("divisa", "diuido", "VERB", "divide, separate")])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.glosses == ["divide, separate"]

    def test_no_gloss_extension_safe(self, blank_nlp):
        doc = make_doc(blank_nlp, [("toga", "toga", "NOUN", None)])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.glosses == []

    def test_groups_by_lemma_pos(self, blank_nlp):
        doc = make_doc(blank_nlp, [
            ("toga", "toga", "NOUN", None),
            ("togam", "toga", "NOUN", None),
        ])
        result = build_vocab_list(doc, PipelineConfig())
        assert len(result) == 1
        entry = result.entries[0]
        assert entry.frequency == 2
        assert entry.forms_seen == {"toga", "togam"}

    def test_display_lemma_v_form(self, blank_nlp):
        doc = make_doc(blank_nlp, [("divisa", "diuido", "VERB", None)])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.display_lemma == "divido"

    def test_first_index_tracks_reading_order(self, blank_nlp):
        doc = make_doc(blank_nlp, [
            ("toga", "toga", "NOUN", None),
            ("virilis", "uirilis", "ADJ", None),
        ])
        vl = build_vocab_list(doc, PipelineConfig())
        assert [e.first_index for e in vl.entries] == [0, 1]

    def test_builds_on_model_free_doc(self, blank_nlp):
        """Building works on a hand-built Doc (no model, no senter required)."""
        doc = make_doc(blank_nlp, [
            ("toga", "toga", "NOUN", None),
            ("virilis", "uirilis", "ADJ", None),
        ])
        result = build_vocab_list(doc, PipelineConfig())
        assert len(result) == 2
