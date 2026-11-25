"""``latincy_vocab`` spaCy pipeline component.

Registers a ``@Language.factory("latincy_vocab")`` that consumes an already-parsed
``Doc`` and sets the doc-level extension ``doc._.vocab_list`` (a
:class:`~vocabbuilder.core.models.VocabList`). It reads lemma/POS and, when an
upstream pipe set it, ``token._.gloss`` — it never runs a model or loads
gloss/lexicon files itself. Runs producer-side, like ``latincy_wsd`` / ``latincy_nel``.

Usage::

    import spacy
    import vocabbuilder  # registers the factory on import

    nlp = spacy.load("la_core_web_lg")
    nlp.add_pipe("whitakers_words")   # optional: sets token._.gloss upstream
    nlp.add_pipe("latincy_vocab")
    doc = nlp("Gallia est omnis divisa in partes tres.")
    for entry in doc._.vocab_list:
        print(entry.display_lemma, entry.pos, entry.glosses)
"""

import weakref
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from spacy.language import Language
from spacy.tokens import Doc

from vocabbuilder.core.config import PipelineConfig
from vocabbuilder.data.lemma_overrides import DEFAULT_LEMMA_OVERRIDES, LemmaOverrideRule
from vocabbuilder.processors.vocab_core import build_vocab_list


def _ensure_doc_extension() -> None:
    if not Doc.has_extension("vocab_list"):
        Doc.set_extension("vocab_list", default=None)


@Language.factory(
    "latincy_vocab",
    default_config={
        "exclude_pos": ["PROPN", "PUNCT", "X"],
        "drop_enclitics": True,
        "enclitic_lemmas": ["que"],
        "keep_glossed_propn": True,
        "exclude_ent_types": [],
        "lemma_overrides": None,
    },
    assigns=["doc._.vocab_list"],
)
def create_latincy_vocab(
    nlp: Language,
    name: str,
    exclude_pos: List[str],
    drop_enclitics: bool,
    enclitic_lemmas: List[str],
    keep_glossed_propn: bool,
    exclude_ent_types: List[str],
    lemma_overrides: Optional[List[Dict[str, Any]]],
) -> "LatinVocab":
    return LatinVocab(
        nlp,
        name,
        exclude_pos,
        drop_enclitics,
        enclitic_lemmas,
        keep_glossed_propn,
        exclude_ent_types,
        lemma_overrides,
    )


def _rules_from_dicts(
    rules: Optional[List[Dict[str, Any]]],
) -> tuple[LemmaOverrideRule, ...]:
    """``None`` -> the curated default table; ``[]`` -> overrides disabled."""
    if rules is None:
        return DEFAULT_LEMMA_OVERRIDES
    return tuple(LemmaOverrideRule(**r) for r in rules)


class LatinVocab:
    """Aggregate a parsed ``Doc`` into ``doc._.vocab_list``.

    Consumes upstream token annotations (lemma, POS, ``token._.gloss``); does not
    run nlp or load files. Config knobs are JSON-safe lists/bools (spaCy
    serialization) and rebuilt into sets for the shared core.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        exclude_pos: list[str],
        drop_enclitics: bool,
        enclitic_lemmas: list[str],
        keep_glossed_propn: bool,
        exclude_ent_types: list[str] | None = None,
        lemma_overrides: list[dict[str, Any]] | None = None,
    ) -> None:
        _ensure_doc_extension()
        self.name = name
        self._nlp = weakref.ref(nlp)  # weak: avoid an nlp <-> component cycle
        # use_glosses=False and no resolve_data_paths(): glosses come only from
        # upstream token._.gloss, never from a file loaded here.
        self._config = PipelineConfig(
            use_glosses=False,
            exclude_pos=set(exclude_pos),
            drop_enclitics=drop_enclitics,
            enclitic_lemmas=set(enclitic_lemmas),
            keep_glossed_propn=keep_glossed_propn,
            exclude_ent_types=set(exclude_ent_types or []),
            lemma_overrides=_rules_from_dicts(lemma_overrides),
        )

    def _has_gloss_pipe(self) -> bool:
        """Whether this ``nlp`` carries latincy-lexicon's ``whitakers_words`` pipe
        (by factory, so a renamed pipe counts) -- per-pipeline, not process-global."""
        nlp = self._nlp()
        return nlp is not None and any(
            nlp.get_pipe_meta(n).factory == "whitakers_words" for n in nlp.pipe_names
        )

    def __call__(self, doc: Doc) -> Doc:
        doc._.vocab_list = build_vocab_list(
            doc, self._config, glosses_expected=self._has_gloss_pipe()
        )
        return doc

    # -- serialization (config only) ------------------------------------------
    def to_disk(self, path, exclude=()) -> None:
        import srsly

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # The curated default table is saved as null, not written out, so a
        # reloaded pipeline follows the package's table rather than a copy frozen
        # at save time. from_disk reads null/absent as the default.
        overrides = self._config.lemma_overrides
        saved_overrides = (
            None
            if overrides is DEFAULT_LEMMA_OVERRIDES
            else [asdict(r) for r in overrides]
        )
        srsly.write_json(
            path / "cfg.json",
            {
                "exclude_pos": sorted(self._config.exclude_pos),
                "drop_enclitics": self._config.drop_enclitics,
                "enclitic_lemmas": sorted(self._config.enclitic_lemmas),
                "keep_glossed_propn": self._config.keep_glossed_propn,
                "exclude_ent_types": sorted(self._config.exclude_ent_types),
                "lemma_overrides": saved_overrides,
            },
        )

    def from_disk(self, path, exclude=()) -> "LatinVocab":
        import srsly

        cfg = srsly.read_json(Path(path) / "cfg.json")
        self._config.exclude_pos = set(cfg["exclude_pos"])
        self._config.drop_enclitics = cfg["drop_enclitics"]
        self._config.enclitic_lemmas = set(cfg["enclitic_lemmas"])
        # Tolerate configs written before keep_glossed_propn existed.
        self._config.keep_glossed_propn = cfg.get("keep_glossed_propn", True)
        self._config.exclude_ent_types = set(cfg.get("exclude_ent_types", []))
        # Absent (older configs) or null -> the curated default table.
        self._config.lemma_overrides = _rules_from_dicts(cfg.get("lemma_overrides"))
        return self
