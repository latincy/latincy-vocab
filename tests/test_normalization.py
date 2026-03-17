"""Tests for normalization utilities."""

from vocabbuilder.utils.normalization import (
    strip_enclitic,
    to_u_form,
    to_v_form,
    upos_to_gloss_pos,
)


class TestToUForm:
    def test_v_to_u(self):
        assert to_u_form("divisa") == "diuisa"

    def test_j_to_i(self):
        assert to_u_form("jam") == "iam"

    def test_already_u_form(self):
        assert to_u_form("diuisa") == "diuisa"

    def test_mixed(self):
        assert to_u_form("juvenis") == "iuuenis"


class TestToVForm:
    def test_u_to_v(self):
        assert to_v_form("diuisa") == "divisa"

    def test_already_v_form(self):
        assert to_v_form("divisa") == "divisa"


class TestStripEnclitic:
    def test_que(self):
        base, enc = strip_enclitic("populusque")
        assert base == "populus"
        assert enc == "que"

    def test_ne(self):
        base, enc = strip_enclitic("vidistine")
        assert base == "vidisti"
        assert enc == "ne"

    def test_ve(self):
        base, enc = strip_enclitic("nocturnaeve")
        assert base == "nocturnae"
        assert enc == "ve"

    def test_no_enclitic(self):
        base, enc = strip_enclitic("Roma")
        assert base == "Roma"
        assert enc is None

    def test_too_short(self):
        """Don't strip if the remaining base would be too short."""
        base, enc = strip_enclitic("que")
        assert base == "que"
        assert enc is None


class TestUposToGlossPos:
    def test_noun(self):
        assert upos_to_gloss_pos("NOUN") == "noun"

    def test_verb(self):
        assert upos_to_gloss_pos("VERB") == "verb"

    def test_propn(self):
        assert upos_to_gloss_pos("PROPN") == "name"

    def test_unknown(self):
        assert upos_to_gloss_pos("UNKNOWN") is None

    def test_punct_not_mapped(self):
        assert upos_to_gloss_pos("PUNCT") is None
