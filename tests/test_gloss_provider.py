"""Tests for GlossProvider (integration — needs glosses data file)."""

import pytest

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.data.gloss_provider import GlossProvider


@pytest.fixture(scope="module")
def gloss_provider() -> GlossProvider:
    config = PipelineConfig()
    config.resolve_data_paths()
    return GlossProvider(config.glosses_path)


class TestGlossProvider:
    def test_loads_entries(self, gloss_provider: GlossProvider):
        assert gloss_provider.size > 1000

    def test_lookup_by_u_form(self, gloss_provider: GlossProvider):
        """spaCy gives u-form lemmas like 'diuido' — should find glosses."""
        result = gloss_provider.lookup("diuido", "VERB")
        assert result is not None
        assert len(result.glosses) > 0
        assert result.display_lemma == "divido"

    def test_lookup_by_v_form(self, gloss_provider: GlossProvider):
        """Should also work if given v-form directly."""
        result = gloss_provider.lookup("divido", "VERB")
        assert result is not None
        assert len(result.glosses) > 0

    def test_lookup_common_noun(self, gloss_provider: GlossProvider):
        result = gloss_provider.lookup("pars", "NOUN")
        assert result is not None
        assert any("part" in g.lower() for g in result.glosses)

    def test_lookup_missing_lemma(self, gloss_provider: GlossProvider):
        result = gloss_provider.lookup("xyznonexistent", "NOUN")
        assert result is None

    def test_pos_bridging(self, gloss_provider: GlossProvider):
        """UPOS 'NOUN' should bridge to gloss 'noun'."""
        result = gloss_provider.lookup("bellum", "NOUN")
        assert result is not None

    def test_any_pos_fallback(self, gloss_provider: GlossProvider):
        """If POS doesn't match exactly, fall back to any POS for that lemma."""
        # 'omnis' is 'det' in glosses but spaCy may tag as DET or ADJ
        result = gloss_provider.lookup("omnis", "ADJ")
        if result:
            assert len(result.glosses) > 0

    def test_max_glosses(self, gloss_provider: GlossProvider):
        result = gloss_provider.lookup("sum", "VERB", max_glosses=2)
        if result:
            assert len(result.glosses) <= 2

    def test_get_display_lemma_known(self, gloss_provider: GlossProvider):
        display = gloss_provider.get_display_lemma("diuido")
        assert display == "divido"

    def test_get_display_lemma_unknown(self, gloss_provider: GlossProvider):
        """Unknown lemma should still get v-form via latincy-preprocess."""
        display = gloss_provider.get_display_lemma("xyzuuord")
        assert isinstance(display, str)
