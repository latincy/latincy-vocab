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

from pathlib import Path
from typing import List

from spacy.language import Language
from spacy.tokens import Doc

from vocabbuilder.core.config import PipelineConfig
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
) -> "LatinVocab":
    return LatinVocab(
        nlp, name, exclude_pos, drop_enclitics, enclitic_lemmas, keep_glossed_propn
    )


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
    ) -> None:
        _ensure_doc_extension()
        self.name = name
        # use_glosses=False and no resolve_data_paths(): glosses come only from
        # upstream token._.gloss, never from a file loaded here.
        self._config = PipelineConfig(
            use_glosses=False,
            exclude_pos=set(exclude_pos),
            drop_enclitics=drop_enclitics,
            enclitic_lemmas=set(enclitic_lemmas),
            keep_glossed_propn=keep_glossed_propn,
        )

    def __call__(self, doc: Doc) -> Doc:
        doc._.vocab_list = build_vocab_list(doc, self._config)
        return doc

    # -- serialization (config only) ------------------------------------------
    def to_disk(self, path, exclude=()) -> None:
        import srsly

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        srsly.write_json(
            path / "cfg.json",
            {
                "exclude_pos": sorted(self._config.exclude_pos),
                "drop_enclitics": self._config.drop_enclitics,
                "enclitic_lemmas": sorted(self._config.enclitic_lemmas),
                "keep_glossed_propn": self._config.keep_glossed_propn,
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
        return self
