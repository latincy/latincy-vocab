"""Tests for the ``latincy_vocab`` spaCy pipeline component.

Mostly model-free: Docs are hand-built and the component is exercised directly.
One integration test loads ``la_core_web_lg``.
"""

from __future__ import annotations

import pytest
import spacy
import srsly
from spacy.tokens import Doc, Token

import vocabbuilder  # noqa: F401 — registers the latincy_vocab factory on import
from vocabbuilder import VocabList


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


@pytest.fixture
def nlp_with_vocab():
    nlp = spacy.blank("la")
    component = nlp.add_pipe("latincy_vocab")
    return nlp, component


class TestRegistration:
    def test_factory_discoverable_after_import(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latincy_vocab")
        assert "latincy_vocab" in nlp.pipe_names

    def test_doc_extension_registered(self, nlp_with_vocab):
        assert Doc.has_extension("vocab_list")

    def test_double_add_is_idempotent(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latincy_vocab")
        nlp.add_pipe("latincy_vocab", name="latincy_vocab2")  # must not raise
        assert "latincy_vocab2" in nlp.pipe_names

    def test_default_config_json_serializable(self, nlp_with_vocab):
        nlp, _ = nlp_with_vocab
        srsly.json_dumps(nlp.config)  # must not raise (no sets/Paths leaked)


class TestComponentBehavior:
    def test_sets_vocab_list(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = make_doc(nlp, [("toga", "toga", "NOUN", None)])
        out = component(doc)
        assert out is doc
        assert isinstance(doc._.vocab_list, VocabList)

    def test_propn_excluded(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [
            ("Roma", "Roma", "PROPN", None),
            ("toga", "toga", "NOUN", None),
        ]))
        lemmas = {e.lemma for e in doc._.vocab_list}
        assert "Roma" not in lemmas and "toga" in lemmas

    def test_glossed_propn_rescued(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [
            ("Musa", "musa", "PROPN", "muse"),      # glossed → kept
            ("Aquitani", "Aquitanus", "PROPN", None),  # unglossed → dropped
        ]))
        lemmas = {e.lemma for e in doc._.vocab_list}
        assert "musa" in lemmas and "Aquitanus" not in lemmas

    def test_keep_glossed_propn_configurable(self):
        nlp = spacy.blank("la")
        component = nlp.add_pipe(
            "latincy_vocab", config={"keep_glossed_propn": False}
        )
        doc = component(make_doc(nlp, [("Musa", "musa", "PROPN", "muse")]))
        assert len(doc._.vocab_list) == 0

    def test_enclitic_que_dropped(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [
            ("populus", "populus", "NOUN", None),
            ("que", "que", "CCONJ", None),
        ]))
        assert "que" not in {e.lemma for e in doc._.vocab_list}

    def test_punct_excluded(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [
            ("toga", "toga", "NOUN", None),
            (".", ".", "PUNCT", None),
        ]))
        assert len(doc._.vocab_list) == 1

    def test_gloss_consumed(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [("divisa", "diuido", "VERB", "divide")]))
        assert doc._.vocab_list.entries[0].glosses == ["divide"]

    def test_display_lemma_v_form(self, nlp_with_vocab):
        nlp, component = nlp_with_vocab
        doc = component(make_doc(nlp, [("divisa", "diuido", "VERB", None)]))
        assert doc._.vocab_list.entries[0].display_lemma == "divido"


class TestSerialization:
    def test_to_from_disk_roundtrip(self, tmp_path):
        nlp = spacy.blank("la")
        c1 = nlp.add_pipe("latincy_vocab")
        c1.to_disk(tmp_path)

        nlp2 = spacy.blank("la")
        c2 = nlp2.add_pipe(
            "latincy_vocab",
            config={"drop_enclitics": False, "enclitic_lemmas": ["ne"], "exclude_pos": ["X"]},
        )
        c2.from_disk(tmp_path)
        assert c2._config.drop_enclitics is True
        assert c2._config.enclitic_lemmas == {"que"}
        assert c2._config.exclude_pos == {"PROPN", "PUNCT", "X"}
        assert c2._config.keep_glossed_propn is True


class TestIntegration:
    def test_add_pipe_real_model(self, caesar_passage):
        nlp = spacy.load("la_core_web_lg", disable=["lookup_lemmatizer"])
        nlp.add_pipe("latincy_vocab")
        doc = nlp(caesar_passage)
        vl = doc._.vocab_list
        assert isinstance(vl, VocabList) and len(vl) > 0
        lemmas = {e.lemma for e in vl}
        assert "sum" in lemmas and "pars" in lemmas
        # PROPN is now allowed only when Whitaker's Words glossed it — no bare
        # (unglossed) proper names slip into the list.
        assert all(e.glosses for e in vl if e.pos == "PROPN")
