"""Pure-function tests for the curated lemma-override table."""

from __future__ import annotations

from vocabbuilder.data.lemma_overrides import resolve_lemma_override


def test_fires_on_exact_match():
    assert resolve_lemma_override("latus", "VERB", {"VerbForm": "Part"}) == "fero"


def test_does_not_fire_wrong_pos():
    assert resolve_lemma_override("latus", "NOUN", {"VerbForm": "Part"}) is None


def test_does_not_fire_missing_morph_feature():
    assert resolve_lemma_override("latus", "VERB", {}) is None


def test_unrelated_lemma_untouched():
    assert resolve_lemma_override("amo", "VERB", {}) is None


def test_contemplo_deponent_fires():
    assert resolve_lemma_override("contemplo", "VERB", {"Voice": "Pass"}) == "contemplor"


def test_contemplo_active_voice_untouched():
    assert resolve_lemma_override("contemplo", "VERB", {"Voice": "Act"}) is None


class TestCustomRuleIndexCache:
    def test_custom_rules_indexed_once(self, monkeypatch):
        from vocabbuilder.data import lemma_overrides as lo

        rules = (lo.LemmaOverrideRule("foo", "VERB", "bar"),)
        calls = []
        real = lo._build_index
        monkeypatch.setattr(lo, "_build_index", lambda r: calls.append(1) or real(r))
        for _ in range(5):
            assert lo.resolve_lemma_override("foo", "VERB", {}, rules) == "bar"
        assert len(calls) == 1
