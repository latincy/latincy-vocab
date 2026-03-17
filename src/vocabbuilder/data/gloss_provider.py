"""Gloss lookup from latincy-words JSONL data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from vocabbuilder.utils.normalization import to_u_form, to_v_form, upos_to_gloss_pos


class GlossResult(NamedTuple):
    """Result of a gloss lookup."""

    glosses: list[str]
    display_lemma: str


class GlossProvider:
    """Provides English glosses for Latin lemmas.

    Loads latin-glosses.jsonl and indexes by (u-form lemma, pos) for fast lookup.
    Builds a u→v mapping at load time to bridge spaCy's u-only lemmas with the
    v-form lemmas in the glosses file.
    """

    def __init__(self, glosses_path: Path) -> None:
        # (u_form_lemma, gloss_pos) -> {"glosses": [...], "v_lemma": str}
        self._index: dict[tuple[str, str], dict] = {}
        # u_form -> v_form display lemma (from glosses file)
        self._u_to_v: dict[str, str] = {}
        self._load(glosses_path)

    def _load(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                v_lemma: str = entry["lemma"]
                pos: str = entry["pos"]
                glosses: list[str] = entry["glosses"]

                u_lemma = to_u_form(v_lemma)
                key = (u_lemma, pos)

                # First entry wins for a given (u_lemma, pos)
                if key not in self._index:
                    self._index[key] = {"glosses": glosses, "v_lemma": v_lemma}

                # Build u→v mapping (first v-form seen wins)
                if u_lemma not in self._u_to_v:
                    self._u_to_v[u_lemma] = v_lemma

    def lookup(
        self, lemma: str, upos: str, max_glosses: int = 5
    ) -> GlossResult | None:
        """Look up glosses for a lemma+POS.

        Tries: exact (u_lemma, gloss_pos) → any-POS fallback.
        """
        u_lemma = to_u_form(lemma)
        gloss_pos = upos_to_gloss_pos(upos)

        # Try exact POS match
        if gloss_pos:
            entry = self._index.get((u_lemma, gloss_pos))
            if entry:
                return GlossResult(
                    glosses=entry["glosses"][:max_glosses],
                    display_lemma=entry["v_lemma"],
                )

        # Fallback: any POS for this lemma
        for (idx_lemma, _pos), entry in self._index.items():
            if idx_lemma == u_lemma:
                return GlossResult(
                    glosses=entry["glosses"][:max_glosses],
                    display_lemma=entry["v_lemma"],
                )

        return None

    def get_display_lemma(self, u_lemma: str) -> str:
        """Convert a u-only lemma to v-form for display.

        Falls back to latincy-preprocess v-form conversion if not in glosses.
        """
        u_key = to_u_form(u_lemma)
        if u_key in self._u_to_v:
            return self._u_to_v[u_key]
        return to_v_form(u_lemma)

    @property
    def size(self) -> int:
        return len(self._index)
