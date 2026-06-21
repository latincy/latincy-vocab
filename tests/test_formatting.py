"""POS-aware citation-form formatting (model-free)."""

from __future__ import annotations

import pytest
import spacy
from spacy.tokens import Doc, Token

from latincy_lexicon import format_principal_parts

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabEntry
from vocabbuilder.processors.vocab_core import build_vocab_list
from vocabbuilder.utils.normalization import upos_to_abbrev


def make_entry(lemma, pos, *, citation=None, display=None, glosses=None):
    return VocabEntry(
        lemma=lemma,
        display_lemma=display or lemma,
        pos=pos,
        citation_form=citation,
        glosses=list(glosses or []),
    )


class TestPosAbbrev:
    def test_known_tags(self):
        assert upos_to_abbrev("VERB") == "v."
        assert upos_to_abbrev("ADJ") == "adj."
        assert upos_to_abbrev("DET") == "adj."
        assert upos_to_abbrev("ADV") == "adv."
        assert upos_to_abbrev("ADP") == "prep."
        assert upos_to_abbrev("CCONJ") == "conj."


class TestVocabEntryFormatting:
    def test_headword_prefers_citation(self):
        e = make_entry("duco", "VERB", citation="duco, ducere, duxi, ductum")
        assert e.headword == "duco, ducere, duxi, ductum"

    def test_headword_falls_back_to_display(self):
        e = make_entry("duco", "VERB", display="duco")
        assert e.headword == "duco"

    def test_pos_marker_noun_is_empty(self):
        """Nouns carry gender in the citation form, so no extra POS tag."""
        e = make_entry("amicus", "NOUN", citation="amicus, amici, m.")
        assert e.pos_marker == ""

    def test_pos_marker_verb(self):
        assert make_entry("duco", "VERB").pos_marker == "v."

    def test_pos_marker_suppressed_when_citation_carries_gender(self):
        """If the citation already shows a gender (noun-shaped form), don't also
        tag a part of speech — avoids 'suus, sui, m., adj.' for mistyped words."""
        e = make_entry("suus", "DET", citation="suus, sui, m.")
        assert e.pos_marker == ""

    def test_formatted_verb(self):
        e = make_entry("duco", "VERB", citation="duco, ducere, duxi, ductum", glosses=["to lead"])
        assert e.formatted() == "duco, ducere, duxi, ductum, v., to lead"

    def test_formatted_noun_uses_gender_not_pos(self):
        e = make_entry("amicus", "NOUN", citation="amicus, amici, m.", glosses=["friend"])
        assert e.formatted() == "amicus, amici, m., friend"

    def test_formatted_omits_empty_pieces(self):
        e = make_entry("et", "CCONJ")  # no citation, no gloss
        assert e.formatted() == "et, conj."


def _ensure_lexicon_ext():
    if not Token.has_extension("lexicon"):
        Token.set_extension("lexicon", default=None)


class TestCoreWiresCitation:
    def test_citation_form_from_token_lexicon(self):
        nlp = spacy.blank("la")
        _ensure_lexicon_ext()
        doc = Doc(nlp.vocab, words=["narrare"])
        tok = doc[0]
        tok.lemma_, tok.pos_ = "narro", "VERB"
        entry = {"pos": "V", "headword": "narro",
                 "principal_parts": ["narr", "narr", "narrav", "narrat"]}
        tok._.lexicon = [entry]
        vl = build_vocab_list(doc, PipelineConfig())
        assert vl.entries[0].citation_form == format_principal_parts(entry)
        assert vl.entries[0].citation_form == "narro, narrare, narravi, narratum"

    def test_no_lexicon_means_no_citation(self):
        nlp = spacy.blank("la")
        _ensure_lexicon_ext()
        doc = Doc(nlp.vocab, words=["toga"])
        doc[0].lemma_, doc[0].pos_ = "toga", "NOUN"
        # token._.lexicon defaults to None
        vl = build_vocab_list(doc, PipelineConfig())
        assert vl.entries[0].citation_form is None
        assert vl.entries[0].headword == "toga"
