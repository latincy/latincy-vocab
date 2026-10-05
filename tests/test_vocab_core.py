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


def make_doc_with_morph(nlp, specs):
    """Build a Doc from (text, lemma, pos, gloss|None, morph|"") tuples."""
    _ensure_gloss_ext()
    doc = Doc(nlp.vocab, words=[s[0] for s in specs])
    for tok, (_text, lemma, pos, gloss, morph) in zip(doc, specs):
        tok.lemma_ = lemma
        tok.pos_ = pos
        if gloss is not None:
            tok._.gloss = gloss
        if morph:
            tok.set_morph(morph)
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

    def test_glossed_propn_rescued(self, blank_nlp):
        """A PROPN that WW glosses (``Musa`` → "muse") is kept, not dropped."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", "muse")])
        vl = build_vocab_list(doc, PipelineConfig())
        assert [e.lemma for e in vl] == ["musa"]

    def test_glossed_propn_stays_propn(self, blank_nlp):
        """The rescued token keeps its PROPN pos (for a later NER/NEL channel)."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", "muse")])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.pos == "PROPN"
        assert entry.glosses == ["muse"]

    def test_unglossed_propn_dropped(self, blank_nlp):
        """A PROPN with no WW gloss (a bare name) still drops out of the list."""
        doc = make_doc(blank_nlp, [
            ("Aquitani", "Aquitanus", "PROPN", None),
            ("Musa", "musa", "PROPN", "muse"),
        ])
        lemmas = {e.lemma for e in build_vocab_list(doc, PipelineConfig())}
        assert lemmas == {"musa"}

    def test_keep_glossed_propn_configurable(self, blank_nlp):
        """keep_glossed_propn=False restores strict drop-all-PROPN behavior."""
        doc = make_doc(blank_nlp, [("Musa", "musa", "PROPN", "muse")])
        config = PipelineConfig(keep_glossed_propn=False)
        assert len(build_vocab_list(doc, config)) == 0

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


class TestLemmaOverrides:
    def test_latus_participle_corrected_to_fero(self, blank_nlp):
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", None, "VerbForm=Part|Voice=Pass"),
        ])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.lemma == "fero"
        assert entry.pos == "VERB"

    def test_latus_noun_not_corrected(self, blank_nlp):
        """latus tagged NOUN (no VerbForm=Part) -- the genuine 'side' sense --
        must NOT be touched by the fero override."""
        doc = make_doc_with_morph(blank_nlp, [
            ("latere", "latus", "NOUN", None, "Case=Abl|Gender=Neut|Number=Sing"),
        ])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.lemma == "latus"
        assert entry.pos == "NOUN"

    def test_latus_adj_not_corrected(self, blank_nlp):
        """latus tagged ADJ (the 'wide, broad' sense) must NOT be touched."""
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "ADJ", None, "Case=Nom|Gender=Masc|Number=Sing"),
        ])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.lemma == "latus"
        assert entry.pos == "ADJ"

    def test_contemplo_deponent_corrected_to_contemplor(self, blank_nlp):
        doc = make_doc_with_morph(blank_nlp, [
            ("contemplamur", "contemplo", "VERB", None,
             "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin|Voice=Pass"),
        ])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.lemma == "contemplor"

    def test_overrides_disabled_via_config(self, blank_nlp):
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", None, "VerbForm=Part"),
        ])
        entry = build_vocab_list(doc, PipelineConfig(lemma_overrides=())).entries[0]
        assert entry.lemma == "latus"  # override opted out

    def test_overridden_tokens_merge_with_correct_lemma_group(self, blank_nlp):
        """A corrected 'latus' participle and a directly-correct 'tuli'->fero
        token land in ONE (fero, VERB) group, not two."""
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", None, "VerbForm=Part"),
            ("tulit", "fero", "VERB", None, ""),
        ])
        vl = build_vocab_list(doc, PipelineConfig())
        assert len(vl) == 1
        assert vl.entries[0].frequency == 2


class TestOverrideLookup:
    """The override path re-looks-up the corrected lemma; it must never keep the
    pre-override lemma's gloss, and should mirror upstream gloss selection."""

    def test_failed_lookup_drops_stale_gloss_and_warns(self, blank_nlp, monkeypatch):
        from vocabbuilder.processors import vocab_core

        monkeypatch.setattr(vocab_core, "_lexicon_dict", lambda: {})
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", "side, flank", "VerbForm=Part"),
        ])
        with pytest.warns(RuntimeWarning, match="fero"):
            entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.lemma == "fero"
        assert entry.glosses == []
        assert entry.citation_form is None

    def test_gloss_is_pos_ranked_first_sense_without_usage_note(self, blank_nlp, monkeypatch):
        from vocabbuilder.processors import vocab_core

        lex = {"fero": [
            {"ud_pos": ["NOUN"], "glosses": ["wrong pos"]},
            {"ud_pos": ["VERB"], "glosses": ["bear, carry (w/DAT)", "bring"]},
        ]}
        monkeypatch.setattr(vocab_core, "_lexicon_dict", lambda: lex)
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", "side, flank", "VerbForm=Part"),
        ])
        entry = build_vocab_list(doc, PipelineConfig()).entries[0]
        assert entry.glosses == ["bear, carry"]

    def test_failed_lexicon_load_is_not_cached(self, monkeypatch):
        import latincy_lexicon
        from vocabbuilder.processors import vocab_core

        monkeypatch.setattr(vocab_core, "_LEXICON_DICT", None)
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient")
            return {"x": []}

        monkeypatch.setattr(latincy_lexicon, "build_lexicon", flaky)
        with pytest.warns(RuntimeWarning):
            assert vocab_core._lexicon_dict() is None
        assert vocab_core._lexicon_dict() == {"x": []}


class TestOverrideLookupGating:
    def test_lexicon_free_build_never_loads_lexicon(self, blank_nlp, monkeypatch):
        from vocabbuilder.processors import vocab_core

        def boom():
            raise AssertionError("lexicon must not load on a lexicon-free build")

        monkeypatch.setattr(vocab_core, "_lexicon_dict", boom)
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", None, "VerbForm=Part"),
        ])
        vl = build_vocab_list(doc, PipelineConfig(use_glosses=False), glosses_expected=False)
        assert vl.entries[0].lemma == "fero"  # still corrected, just unglossed
        assert vl.entries[0].glosses == []

    def test_lookup_memoized_per_build(self, blank_nlp, monkeypatch):
        from vocabbuilder.processors import vocab_core

        calls = []
        monkeypatch.setattr(
            vocab_core, "_corrected_lookup",
            lambda lemma, pos: calls.append((lemma, pos)) or ("bear", []),
        )
        doc = make_doc_with_morph(blank_nlp, [
            ("latus", "latus", "VERB", None, "VerbForm=Part"),
            ("lata", "latus", "VERB", None, "VerbForm=Part"),
            ("latum", "latus", "VERB", None, "VerbForm=Part"),
        ])
        build_vocab_list(doc, PipelineConfig())
        assert calls == [("fero", "VERB")]
