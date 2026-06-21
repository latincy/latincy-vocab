"""Model-free tests for VocabList ordering + export views."""

from __future__ import annotations

import json

from vocabbuilder.core.models import VocabEntry, VocabList


def make_entry(lemma, pos="NOUN", *, freq=1, glosses=None, first_index=0, forms=None):
    return VocabEntry(
        lemma=lemma,
        display_lemma=lemma,
        pos=pos,
        glosses=list(glosses or []),
        forms_seen=set(forms or [lemma]),
        frequency=freq,
        first_index=first_index,
    )


def test_by_first_occurrence_restores_reading_order():
    vl = VocabList(entries=[
        make_entry("c", first_index=2),
        make_entry("a", first_index=0),
        make_entry("b", first_index=1),
    ])
    assert [e.lemma for e in vl.by_first_occurrence()] == ["a", "b", "c"]


def test_to_dicts_is_json_safe():
    vl = VocabList(entries=[make_entry("toga", glosses=["toga"], forms=["toga", "togam"])])
    d = vl.to_dicts()[0]
    assert d["lemma"] == "toga"
    assert d["pos"] == "NOUN"
    assert d["glosses"] == ["toga"]
    assert isinstance(d["forms_seen"], list)
    assert sorted(d["forms_seen"]) == ["toga", "togam"]
    json.dumps(vl.to_dicts())  # must not raise (no sets)


def test_to_dict_includes_formatted_line():
    """The serialized entry carries a ready-to-render `formatted` line so
    consumers (e.g. the viewer reader) need no formatting logic."""
    e = make_entry("toga", glosses=["toga"])
    e.citation_form = "toga, togae, f."
    d = e.to_dict()
    assert d["formatted"] == e.formatted() == "toga, togae, f., toga"


def test_to_dict_exposes_structured_headword_and_marker():
    """The serialized entry carries the `headword` and `pos_marker` parts so the
    viewer can render rich typography (bold headword, italic marker) without
    re-parsing `formatted`."""
    noun = make_entry("toga", glosses=["toga"])
    noun.citation_form = "toga, togae, f."
    d = noun.to_dict()
    assert d["headword"] == "toga, togae, f."  # gender lives inside citation
    assert d["pos_marker"] == ""               # suppressed: citation ends in f.

    verb = make_entry("narro", pos="VERB", glosses=["tell"])
    verb.citation_form = "narro, narrare, narravi, narratum"
    dv = verb.to_dict()
    assert dv["headword"] == "narro, narrare, narravi, narratum"
    assert dv["pos_marker"] == "v."


def test_nonnoun_with_noun_citation_keeps_marker_and_plain_headword():
    """A non-noun the lexicon mistypes as a noun (e.g. `modus` ADV from a
    mislemmatized `modo`) must NOT render a noun paradigm with the marker blanked.
    The gender-suppression guard and noun-shaped headword are NOUN-only."""
    adv = make_entry("modus", pos="ADV", glosses=["manner"])
    adv.citation_form = "modus, modi, m."
    assert adv.pos_marker == "adv."        # marker NOT suppressed for non-nouns
    assert adv.headword == "modus"         # falls back to display_lemma, not the noun citation
    d = adv.to_dict()
    assert d["pos_marker"] == "adv."
    assert d["headword"] == "modus"


def test_true_noun_still_suppresses_marker_and_keeps_citation_headword():
    """Regression: true nouns keep their full citation as headword, marker blank."""
    noun = make_entry("amicus", pos="NOUN", glosses=["friend"])
    noun.citation_form = "amicus, amici, m."
    assert noun.pos_marker == ""
    assert noun.headword == "amicus, amici, m."


def test_participle_tagged_adj_with_verb_citation_marks_v():
    """Regression: a participle token tags ADJ but its citation is a verb — still v."""
    e = make_entry("conjungo", pos="ADJ", glosses=["join"])
    e.citation_form = "conjungo, conjungere, conjunxi, conjunctum"
    assert e.pos_marker == "v."
    assert e.headword == "conjungo, conjungere, conjunxi, conjunctum"


def test_short_gloss_truncates_slash_pile():
    e = make_entry("facio", pos="VERB", glosses=["make/build/construct/create/cause/do"])
    assert e.short_gloss == "make, build, construct"
    assert e.full_gloss == "make/build/construct/create/cause/do"


def test_short_gloss_caps_long_comma_list():
    e = make_entry("narro", pos="VERB",
                   glosses=["tell, tell about, relate, narrate, recount, describe"])
    assert e.short_gloss == "tell, tell about, relate"


def test_short_gloss_strips_parenthetical_and_keeps_plain_gloss():
    e = make_entry("sapientia", pos="NOUN", glosses=["wisdom (goal of philosopher, Stoic virtue)"])
    assert e.short_gloss == "wisdom"
    plain = make_entry("socer", pos="NOUN", glosses=["father in law"])
    assert plain.short_gloss == "father in law"  # no slash → unchanged


def test_to_dict_exposes_short_and_full_gloss():
    e = make_entry("valeo", pos="VERB", glosses=["be strong/powerful/influential/healthy"])
    d = e.to_dict()
    assert d["short_gloss"] == "be strong, powerful, influential"
    assert d["full_gloss"] == "be strong/powerful/influential/healthy"


def test_to_json_round_trips():
    vl = VocabList(entries=[make_entry("toga")])
    parsed = json.loads(vl.to_json())
    assert parsed[0]["lemma"] == "toga"


def test_to_markdown_renders_lemma_and_gloss():
    vl = VocabList(entries=[make_entry("toga", glosses=["toga, garment"])])
    md = vl.to_markdown()
    assert "toga" in md and "toga, garment" in md
