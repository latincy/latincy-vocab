"""Pipeline configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_sibling_dir(name: str) -> Path | None:
    """Look for a sibling directory relative to this project."""
    # Walk up from this file to find the project root's parent
    project_root = Path(__file__).resolve().parents[3]  # src/vocabbuilder/core -> project root
    parent = project_root.parent  # latincy-v3 dir
    candidate = parent / name
    if candidate.is_dir():
        return candidate
    return None


@dataclass
class PipelineConfig:
    """Configuration for the vocabulary pipeline."""

    spacy_model: str = "la_core_web_sm"
    spacy_disable: list[str] = field(default_factory=lambda: ["lookup_lemmatizer"])

    glosses_path: Path | None = None
    latincy_words_dir: Path | None = None

    use_glosses: bool = True
    max_glosses: int = 5
    min_frequency: int = 1
    # PROPN is excluded by design: proper names route to a separate NER/NEL
    # channel, not the vocabulary list.
    exclude_pos: set[str] = field(default_factory=lambda: {"PROPN", "PUNCT", "SPACE", "X"})

    # Standalone enclitics left behind by tokenization (e.g. 'populusque' →
    # 'populus' + 'que') are dropped; the host word is already lemmatized.
    drop_enclitics: bool = True
    enclitic_lemmas: set[str] = field(default_factory=lambda: {"que"})

    def resolve_data_paths(self) -> PipelineConfig:
        """Resolve gloss data paths, best-effort.

        Glosses are optional: when ``use_glosses`` is False, or the latincy-words
        dataset cannot be found, ``glosses_path`` is left ``None`` and the pipeline
        runs lexicon-free rather than raising. Returns self for chaining.
        """
        if not self.use_glosses:
            self.glosses_path = None
            return self

        if self.latincy_words_dir is None:
            env_dir = os.environ.get("LATINCY_WORDS_DIR")
            if env_dir:
                self.latincy_words_dir = Path(env_dir)
            else:
                self.latincy_words_dir = _find_sibling_dir("latincy-words")

        if self.glosses_path is None and self.latincy_words_dir is not None:
            self.glosses_path = self.latincy_words_dir / "outputs" / "latin-glosses.jsonl"

        # Gloss data is optional — degrade to lexicon-free rather than fail.
        if self.glosses_path is None or not self.glosses_path.exists():
            self.glosses_path = None

        return self
