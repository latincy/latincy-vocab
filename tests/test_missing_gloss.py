"""Coverage-gap handling: gloss-less entries hidden by default, tracked separately.

An ADJ-tagged proper noun absent from Whitaker's Words (``Lavinia`` from
``Laviniaque``) is a real content word the list includes but cannot gloss. Rendered
views hide such gaps by default (``include_missing_gloss=True`` opts them back in),
and :attr:`VocabList.missing_gloss` surfaces them for a future supplementary source.
The lexicon-free path — where every entry is legitimately gloss-less — is protected
by the ``glosses_expected`` flag and always renders in full.
"""

from __future__ import annotations

import spacy
from spacy.tokens import Doc, Token

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabEntry, VocabList
from vocabbuilder.processors.vocab_core import build_vocab_list


def make_entry(lemma, pos="NOUN", *, glosses=None):
    return VocabEntry(
        lemma=lemma,
        display_lemma=lemma,
        pos=pos,
        glosses=list(glosses or []),
    )


def _ensure_gloss_ext() -> None:
    if not Token.has_extension("gloss"):
        Token.set_extension("gloss", default=None)


class TestHasGloss:
    def test_true_when_glossed(self):
        assert make_entry("toga", glosses=["garment"]).has_gloss is True

    def test_false_when_unglossed(self):
        assert make_entry("lavinia", pos="ADJ").has_gloss is False


class TestGlossesExpectedFlag:
    def test_default_is_false(self):
        assert VocabList(entries=[make_entry("toga")]).glosses_expected is False

    def test_propagates_through_views(self):
        vl = VocabList(entries=[make_entry("toga", glosses=["garment"])], glosses_expected=True)
        assert vl.by_frequency().glosses_expected is True
        assert vl.by_alpha().glosses_expected is True
        assert vl.by_first_occurrence().glosses_expected is True
        assert vl.filter_pos({"NOUN"}).glosses_expected is True
        assert vl.filter_min_frequency(1).glosses_expected is True
        assert vl.filter_lemmas(keep=["toga"]).glosses_expected is True


class TestRenderFiltering:
    def _mixed(self) -> VocabList:
        return VocabList(
            entries=[
                make_entry("musa", pos="PROPN", glosses=["muse"]),
                make_entry("lavinia", pos="ADJ"),  # coverage gap: no gloss
            ],
            glosses_expected=True,
        )

    def test_markdown_hides_gap_by_default(self):
        md = self._mixed().to_markdown()
        assert "musa" in md
        assert "lavinia" not in md

    def test_markdown_includes_gap_on_opt_in(self):
        md = self._mixed().to_markdown(include_missing_gloss=True)
        assert "musa" in md and "lavinia" in md

    def test_dicts_hide_gap_by_default(self):
        lemmas = [d["lemma"] for d in self._mixed().to_dicts()]
        assert lemmas == ["musa"]

    def test_dicts_include_gap_on_opt_in(self):
        lemmas = [d["lemma"] for d in self._mixed().to_dicts(include_missing_gloss=True)]
        assert lemmas == ["musa", "lavinia"]

    def test_json_honours_flag(self):
        import json

        vl = self._mixed()
        assert len(json.loads(vl.to_json())) == 1
        assert len(json.loads(vl.to_json(include_missing_gloss=True))) == 2

    def test_hiding_survives_chained_view(self):
        """glosses_expected must ride through by_frequency() so gaps stay hidden."""
        md = self._mixed().by_frequency().to_markdown()
        assert "lavinia" not in md


class TestLexiconFreeProtection:
    """glosses_expected=False → nothing is a gap; every entry always renders."""

    def _all_gloss_less(self) -> VocabList:
        return VocabList(
            entries=[make_entry("gallia", pos="NOUN"), make_entry("belgae", pos="NOUN")],
            glosses_expected=False,
        )

    def test_markdown_renders_all(self):
        md = self._all_gloss_less().to_markdown()
        assert "gallia" in md and "belgae" in md

    def test_dicts_render_all(self):
        assert len(self._all_gloss_less().to_dicts()) == 2

    def test_missing_gloss_is_empty(self):
        assert list(self._all_gloss_less().missing_gloss) == []


class TestMissingGlossView:
    def test_returns_only_unglossed_when_expected(self):
        vl = VocabList(
            entries=[
                make_entry("musa", pos="PROPN", glosses=["muse"]),
                make_entry("lavinia", pos="ADJ"),
            ],
            glosses_expected=True,
        )
        gaps = vl.missing_gloss
        assert isinstance(gaps, VocabList)
        assert [e.lemma for e in gaps] == ["lavinia"]
        assert gaps.glosses_expected is True

    def test_empty_when_not_expected(self):
        vl = VocabList(entries=[make_entry("lavinia", pos="ADJ")], glosses_expected=False)
        assert list(vl.missing_gloss) == []


class TestLaviniaRegression:
    """Model-free end-to-end through build_vocab_list with a real gloss extension."""

    def test_unglossed_adj_is_a_tracked_hidden_gap(self):
        nlp = spacy.blank("la")
        _ensure_gloss_ext()
        doc = Doc(nlp.vocab, words=["Musa", "Lavinia"])
        doc[0].lemma_, doc[0].pos_ = "musa", "PROPN"
        doc[0]._.gloss = "muse"           # WW covers Musa
        doc[1].lemma_, doc[1].pos_ = "lavinia", "ADJ"
        doc[1]._.gloss = None             # WW has no Lavinius/-a/-um

        vl = build_vocab_list(doc, PipelineConfig())

        # The gloss pipe was in effect, so gaps are real.
        assert vl.glosses_expected is True
        # Both tokens are kept (Musa is a glossed PROPN; Lavinia is a content ADJ).
        assert {e.lemma for e in vl} == {"musa", "lavinia"}
        # The gap is tracked...
        assert [e.lemma for e in vl.missing_gloss] == ["lavinia"]
        # ...hidden from the default glossary...
        md = vl.to_markdown()
        assert "musa" in md and "lavinia" not in md
        # ...and recoverable on demand.
        assert "lavinia" in vl.to_markdown(include_missing_gloss=True)
