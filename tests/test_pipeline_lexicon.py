"""End-to-end: VocabPipeline sources glosses + citation forms from latincy-lexicon.

This is the beta acceptance gate. With latincy-words gone, ``VocabPipeline``
adds ``whitakers_words`` (zero-config bundled lexicon) to its own pipeline, so
``process()`` yields entries carrying Whitaker glosses and principal-part
citation forms — and never the pre-fix defective-verb placeholder ``zzzo``.
"""

import pytest

from vocabbuilder import VocabPipeline


@pytest.fixture(scope="module")
def pipeline() -> VocabPipeline:
    return VocabPipeline()


def test_no_zzz_placeholder_in_output(pipeline: VocabPipeline):
    vl = pipeline.process("Odi et amo. Quare id faciam fortasse requiris.")
    for e in vl:
        assert "zzz" not in (e.citation_form or ""), e
        assert "zzz" not in e.formatted(), e


def test_citation_forms_present(pipeline: VocabPipeline):
    vl = pipeline.process("Gallia est omnis divisa in partes tres.")
    cited = [e for e in vl if e.citation_form]
    assert cited, "expected citation forms from latincy-lexicon whitakers_words"


def test_glosses_present(pipeline: VocabPipeline):
    vl = pipeline.process("Gallia est omnis divisa in partes tres.")
    assert any(e.glosses for e in vl), "expected Whitaker glosses on some entries"


def test_odi_renders_fixed_citation_when_present(pipeline: VocabPipeline):
    """If the model lemmatizes the defective verb, it must read 'odi, odisse,
    osus sum' — never 'zzzo, osere'."""
    vl = pipeline.process("Odi et amo.")
    odi = [e for e in vl if e.lemma == "odi"]
    if odi:
        assert odi[0].headword == "odi, odisse, osus sum"
        assert "zzzo" not in odi[0].formatted()
