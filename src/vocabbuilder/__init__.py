"""Latin vocabulary list builder powered by LatinCy spaCy ecosystem."""

from vocabbuilder.component import LatinVocab  # noqa: F401 — registers @Language.factory
from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.core.models import VocabEntry, VocabList
from vocabbuilder.core.pipeline import VocabPipeline

__all__ = ["VocabPipeline", "VocabEntry", "VocabList", "PipelineConfig", "LatinVocab"]
