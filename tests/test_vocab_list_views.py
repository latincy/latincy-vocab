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


def test_to_json_round_trips():
    vl = VocabList(entries=[make_entry("toga")])
    parsed = json.loads(vl.to_json())
    assert parsed[0]["lemma"] == "toga"


def test_to_markdown_renders_lemma_and_gloss():
    vl = VocabList(entries=[make_entry("toga", glosses=["toga, garment"])])
    md = vl.to_markdown()
    assert "toga" in md and "toga, garment" in md
