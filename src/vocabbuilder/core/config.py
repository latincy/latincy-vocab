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

    max_glosses: int = 5
    min_frequency: int = 1
    exclude_pos: set[str] = field(default_factory=lambda: {"PUNCT", "SPACE", "X"})

    def resolve_data_paths(self) -> PipelineConfig:
        """Resolve data paths from env vars or auto-detect sibling dirs.

        Returns self for chaining.
        """
        if self.latincy_words_dir is None:
            env_dir = os.environ.get("LATINCY_WORDS_DIR")
            if env_dir:
                self.latincy_words_dir = Path(env_dir)
            else:
                self.latincy_words_dir = _find_sibling_dir("latincy-words")

        if self.latincy_words_dir is None:
            raise FileNotFoundError(
                "Cannot find latincy-words directory. "
                "Set LATINCY_WORDS_DIR env var or place it as a sibling directory."
            )

        if self.glosses_path is None:
            self.glosses_path = self.latincy_words_dir / "outputs" / "latin-glosses.jsonl"

        if not self.glosses_path.exists():
            raise FileNotFoundError(f"Glosses file not found: {self.glosses_path}")

        return self
