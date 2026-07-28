"""Data models for vocabulary building."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from vocabbuilder.utils.normalization import to_u_form, upos_to_abbrev


def _norm_lemma(text: str) -> str:
    """Canonical key for matching lemmas against user-supplied lists.

    Folds to u-form (v→u, j→i) and lowercases, so a wordlist or keyness table
    written with any orthography (``vita``/``uita``, ``Ianus``/``ianus``) matches
    the internal spaCy lemma regardless of u/v/j or case."""
    return to_u_form(text).lower()


def _coerce_keyness_scores(
    scores: Any, feature_names: Iterable[str] | None, document: int | str
) -> dict[str, float]:
    """Normalize a keyness input into a plain ``{lemma: weight}`` dict.

    The digital-humanities workflow for keyness is usually scikit-learn's
    ``TfidfVectorizer`` → a scipy-sparse document-term matrix, sometimes wrapped in
    a pandas DataFrame. This accepts those shapes without importing pandas/scipy/
    sklearn (all duck-typed), so any of the following work:

    * a ``Mapping`` — ``{lemma: weight}`` (or a ``collections.Counter``);
    * a **pandas Series** — one DTM row, index = terms (``dtm_df.loc["ep6.16"]``);
    * a **pandas DataFrame** — a full DTM; ``document`` selects the row by index
      label (``.loc``), falling back to positional ``.iloc`` for an int that is
      not itself an index label;
    * a **scipy sparse matrix or numpy array** — the raw ``fit_transform`` output;
      pass ``feature_names=vectorizer.get_feature_names_out()`` and ``document``
      picks the row (a 1-D vector is taken as-is);
    * an iterable of ``(lemma, weight)`` pairs.

    Weights are cast to ``float``. Raises ``ValueError`` if an unlabeled matrix is
    given without ``feature_names``, or ``TypeError`` if the shape is unrecognized."""
    if isinstance(scores, Mapping):
        return {str(k): float(v) for k, v in scores.items()}

    ndim = getattr(scores, "ndim", None)

    # pandas Series — self-labeled 1-D (a single DTM row).
    if ndim == 1 and hasattr(scores, "to_dict") and hasattr(scores, "index"):
        return {str(k): float(v) for k, v in scores.to_dict().items()}

    # pandas DataFrame — self-labeled 2-D DTM; pick the document row by index
    # label, falling back to positional .iloc for an int that is not a label (so
    # an integer-labeled DTM, e.g. index=[1954, 1955], selects by label not
    # position).
    if ndim == 2 and hasattr(scores, "columns") and hasattr(scores, "loc"):
        index = getattr(scores, "index", None)
        if index is not None and document in index:
            row = scores.loc[document]
        elif isinstance(document, int):
            row = scores.iloc[document]
        else:
            raise KeyError(f"document {document!r} is not a label in the DataFrame index")
        return {str(k): float(v) for k, v in row.to_dict().items()}

    # Unlabeled numeric matrix/array/scipy-sparse — needs explicit feature_names.
    if ndim in (1, 2):
        if feature_names is None:
            raise ValueError(
                "filter_keyness needs feature_names when scores is an unlabeled "
                "matrix/array (e.g. a scipy sparse DTM from sklearn's "
                "TfidfVectorizer). Pass feature_names=vectorizer.get_feature_names_out()."
            )
        row = scores if ndim == 1 else scores[document]
        if hasattr(row, "toarray"):  # scipy sparse row → dense
            row = row.toarray()
        if hasattr(row, "ravel"):  # (1, n_terms) → (n_terms,)
            row = row.ravel()
        names = list(feature_names)
        values = list(row)
        if len(names) != len(values):
            raise ValueError(
                f"feature_names has {len(names)} entries but the selected row has "
                f"{len(values)} weights; they must align — pass the matching "
                "vectorizer.get_feature_names_out()."
            )
        return {str(name): float(v) for name, v in zip(names, values)}

    # Iterable of (lemma, weight) pairs.
    _pairs_error = TypeError(
        f"filter_keyness could not interpret scores of type {type(scores).__name__}; "
        "pass a {lemma: weight} mapping, a pandas Series/DataFrame, a scipy/numpy "
        "matrix with feature_names, or an iterable of (lemma, weight) pairs."
    )
    try:
        items = list(scores)
    except TypeError as exc:
        raise _pairs_error from exc
    result: dict[str, float] = {}
    for item in items:
        try:
            k, v = item
        except (TypeError, ValueError) as exc:
            raise _pairs_error from exc
        # A non-string key means this is almost certainly a raw matrix row
        # (numeric-vs-numeric), not (lemma, weight) pairs — guide the caller to
        # feature_names rather than silently building a garbage {"0.8": 0.1} map.
        if not isinstance(k, str):
            raise TypeError(
                "filter_keyness got a non-string key from an iterable of pairs, which "
                "looks like a raw matrix row. For a 2-D document-term matrix, pass it "
                "with feature_names=vectorizer.get_feature_names_out() instead."
            )
        result[str(k)] = float(v)
    return result

#: Trailing gender abbreviations that mark a noun-shaped citation form.
_GENDER_SUFFIXES = ("m.", "f.", "n.")
#: Senses kept when shortening a slash-packed raw Whitaker gloss.
_SHORT_GLOSS_SENSES = 3


def _shorten_gloss(text: str, max_senses: int = _SHORT_GLOSS_SENSES) -> str:
    """Trim a verbose raw-Whitaker gloss to a few atomic senses.

    Whitaker shortdefs pack many senses into one slash/comma list
    (``make/build/construct/create/cause/do``; ``tell, relate, narrate, …``) and
    tack on encyclopedic parentheticals (``wisdom (goal of philosopher...)``).
    Keep the first ``max_senses`` senses of the first ``;`` clause, splitting on
    ``/`` and ``,``, dropping parentheticals and the ``//`` empty-sense artifact.
    A single multi-word sense (``father in law``, ``be in the habit of``) has no
    separator and is left intact. Presentation-only — the full gloss is preserved
    separately."""
    text = re.sub(r"\s*\([^)]*\)", "", text)  # strip parenthetical tails
    text = text.split(";")[0]
    senses = [s.strip() for s in re.split(r"[/,]", text) if s.strip()]
    if not senses:
        return text.strip()
    return ", ".join(senses[:max_senses])


@dataclass
class TokenInfo:
    """Per-token annotation from spaCy."""

    text: str
    lemma: str
    pos: str
    morph: dict[str, str] = field(default_factory=dict)
    sent_idx: int = 0


@dataclass
class ProcessedPassage:
    """A passage after spaCy processing."""

    text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)


@dataclass
class VocabEntry:
    """A single vocabulary entry."""

    lemma: str
    display_lemma: str
    pos: str
    glosses: list[str] = field(default_factory=list)
    forms_seen: set[str] = field(default_factory=set)
    frequency: int = 1
    morphology: list[dict[str, str]] = field(default_factory=list)
    passage_indices: list[int] = field(default_factory=list)
    first_index: int = 0  # rank of first appearance in the passage (0-based)
    # Textbook citation form from latincy-lexicon (principal parts / gen+gender /
    # -a,-um), when an upstream lexicon pipe set ``token._.lexicon``. None → fall
    # back to ``display_lemma``.
    citation_form: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Unique key for deduplication: (lemma, pos)."""
        return (self.lemma, self.pos)

    @property
    def _noun_shaped_citation(self) -> bool:
        """The citation form ends in a gender abbrev (``amicus, amici, m.``)."""
        return (self.citation_form or "").rstrip().endswith(_GENDER_SUFFIXES)

    @property
    def headword(self) -> str:
        """Citation form if available, else the v-form display lemma.

        A noun-shaped citation is trusted for true nouns and for proper nouns
        (a glossed ``Musa`` legitimately declines ``musa, musae, f.``): only when
        some *other* non-noun token (e.g. an adverb the lexicon mistypes as a
        noun, ``modus`` for mislemmatized ``modo``) carries a ``x, xis, m.``
        citation do we fall back to the plain display lemma rather than render a
        bogus noun paradigm."""
        bogus_noun_paradigm = self.pos not in ("NOUN", "PROPN") and self._noun_shaped_citation
        if self.citation_form and not bogus_noun_paradigm:
            return self.citation_form
        return self.display_lemma

    @property
    def pos_marker(self) -> str:
        """POS tag for display. Empty for true nouns and glossed proper nouns
        (the gender lives inside the citation form, so we never emit a
        contradictory ``m., adj.``)."""
        if self.pos in ("NOUN", "PROPN"):
            return ""
        cit = (self.citation_form or "").rstrip()
        # The citation form carries the lemma's TRUE paradigm; trust it over the
        # inflected token's UPOS. A participle/gerundive token tags ADJ, but its
        # citation form ("conjungo, conjungere, conjunxi, conjunctum") is a verb —
        # detect the infinitive (2nd/3rd principal part in -re/-ri) and mark "v.".
        parts = [p.strip() for p in cit.split(",")]
        if len(parts) >= 3 and any(p.endswith(("re", "ri")) for p in parts[1:3]):
            return "v."
        return upos_to_abbrev(self.pos)

    @property
    def full_gloss(self) -> str:
        """All senses, joined — the unabridged gloss."""
        return "; ".join(self.glosses)

    @property
    def short_gloss(self) -> str:
        """A trimmed gloss for glossary display (see :func:`_shorten_gloss`)."""
        return _shorten_gloss(self.full_gloss)

    def formatted(self) -> str:
        """One-line glossary entry: ``headword, marker, gloss`` (empties omitted)."""
        gloss = "; ".join(self.glosses)
        return ", ".join(p for p in (self.headword, self.pos_marker, gloss) if p)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (``forms_seen`` set → sorted list)."""
        return {
            "lemma": self.lemma,
            "display_lemma": self.display_lemma,
            "citation_form": self.citation_form,
            "headword": self.headword,
            "pos_marker": self.pos_marker,
            "short_gloss": self.short_gloss,
            "full_gloss": self.full_gloss,
            "formatted": self.formatted(),
            "pos": self.pos,
            "glosses": self.glosses,
            "forms_seen": sorted(self.forms_seen),
            "frequency": self.frequency,
            "morphology": self.morphology,
            "passage_indices": self.passage_indices,
            "first_index": self.first_index,
        }

    def merge(self, other: VocabEntry) -> None:
        """Merge another entry (same lemma+pos) into this one."""
        self.forms_seen |= other.forms_seen
        self.frequency += other.frequency
        self.morphology.extend(other.morphology)
        self.passage_indices.extend(other.passage_indices)
        if not self.glosses and other.glosses:
            self.glosses = other.glosses
        if not self.display_lemma and other.display_lemma:
            self.display_lemma = other.display_lemma


@dataclass
class VocabList:
    """Collection of vocabulary entries with filtering and sorting."""

    entries: list[VocabEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VocabEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> VocabEntry:
        return self.entries[index]

    def by_frequency(self, descending: bool = True) -> VocabList:
        """Return a new VocabList sorted by frequency."""
        sorted_entries = sorted(self.entries, key=lambda e: e.frequency, reverse=descending)
        return VocabList(entries=sorted_entries)

    def by_alpha(self) -> VocabList:
        """Return a new VocabList sorted alphabetically by display_lemma."""
        sorted_entries = sorted(self.entries, key=lambda e: e.display_lemma.lower())
        return VocabList(entries=sorted_entries)

    def by_first_occurrence(self) -> VocabList:
        """Return a new VocabList in passage reading order (first appearance)."""
        sorted_entries = sorted(self.entries, key=lambda e: e.first_index)
        return VocabList(entries=sorted_entries)

    def to_dicts(self) -> list[dict[str, Any]]:
        """JSON-safe list of entry dicts."""
        return [e.to_dict() for e in self.entries]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the list to a JSON string."""
        return json.dumps(self.to_dicts(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Render a Markdown glossary: ``- **headword**, marker, gloss``."""
        lines = []
        for e in self.entries:
            tail = ", ".join(p for p in (e.pos_marker, "; ".join(e.glosses)) if p)
            suffix = f", {tail}" if tail else ""
            lines.append(f"- **{e.headword}**{suffix}")
        return "\n".join(lines)

    def filter_pos(self, pos_tags: set[str]) -> VocabList:
        """Return entries matching the given POS tags."""
        return VocabList(entries=[e for e in self.entries if e.pos in pos_tags])

    def filter_min_frequency(self, min_freq: int) -> VocabList:
        """Return entries with frequency >= min_freq."""
        return VocabList(entries=[e for e in self.entries if e.frequency >= min_freq])

    def filter_lemmas(
        self,
        *,
        exclude: Iterable[str] | None = None,
        keep: Iterable[str] | None = None,
    ) -> VocabList:
        """Filter entries against static lemma lists.

        ``exclude`` drops entries whose lemma is in the list — e.g. pass the DCC
        Latin Core Vocabulary to keep only the words a passage adds beyond it.
        ``keep`` retains only entries whose lemma is in the list. The two combine
        (keep ∩ not-exclude). Matching is u-form/j→i- and case-insensitive (see
        :func:`_norm_lemma`), so a list written with any orthography still matches.
        Returns a new list; raises :class:`ValueError` if neither argument is given."""
        if exclude is None and keep is None:
            raise ValueError("filter_lemmas requires exclude and/or keep")
        exclude_set = {_norm_lemma(w) for w in exclude} if exclude is not None else None
        keep_set = {_norm_lemma(w) for w in keep} if keep is not None else None
        result = []
        for e in self.entries:
            key = _norm_lemma(e.lemma)
            if keep_set is not None and key not in keep_set:
                continue
            if exclude_set is not None and key in exclude_set:
                continue
            result.append(e)
        return VocabList(entries=result)

    def filter_keyness(
        self,
        scores: Any,
        *,
        feature_names: Iterable[str] | None = None,
        document: int | str = 0,
        min_score: float | None = None,
        top_n: int | None = None,
    ) -> VocabList:
        """Filter entries by an external keyness measure such as TF-IDF.

        ``scores`` supplies a lemma → keyness weight for each term. Compute it
        however you like — the common digital-humanities path is scikit-learn's
        ``TfidfVectorizer`` over a corpus. Accepted forms (see
        :func:`_coerce_keyness_scores`), all duck-typed so pandas/scipy/sklearn are
        never imported:

        * a ``{lemma: weight}`` mapping;
        * a pandas Series (one DTM row) or DataFrame (``document`` picks the row);
        * a scipy-sparse / numpy DTM straight from ``fit_transform`` — pass
          ``feature_names=vectorizer.get_feature_names_out()`` and ``document`` to
          select the target text's row;
        * an iterable of ``(lemma, weight)`` pairs.

        A lemma absent from ``scores`` is treated as weight 0. Weights that collide
        under u/v/j-folding are summed. Then:

        * ``min_score`` — keep entries whose weight is >= this threshold.
        * ``top_n`` — keep the ``top_n`` highest-weighted entries.
        * neither — keep entries the measure actually scores (any weight, including
          zero or negative — so signed measures like log-likelihood survive);
          words absent from the measure are dropped.

        ``min_score`` and ``top_n`` combine: threshold first, then cap to ``top_n``.
        Lemma lookup is normalized like :meth:`filter_lemmas`. Surviving entries keep
        the list's current order, so chain a ``by_frequency()`` /
        ``by_first_occurrence()`` after to sort."""
        weights = _coerce_keyness_scores(scores, feature_names, document)
        # Fold to canonical lemma keys, SUMMING collisions: a vectorizer over
        # un-normalized Latin emits separate 'vita'/'uita' columns that both map to
        # one lemma, and their keyness should combine rather than clobber.
        norm_scores: dict[str, float] = {}
        for k, v in weights.items():
            key = _norm_lemma(k)
            norm_scores[key] = norm_scores.get(key, 0.0) + v
        # Track whether the measure covers each lemma at all, separately from its
        # weight, so the default filter drops only unscored words while KEEPING
        # signed measures (a negative log-likelihood weight is still "scored").
        scored = []
        for e in self.entries:
            key = _norm_lemma(e.lemma)
            scored.append((e, norm_scores.get(key, 0.0), key in norm_scores))
        if min_score is not None:
            kept = [(e, s) for e, s, _ in scored if s >= min_score]
        else:
            kept = [(e, s) for e, s, is_scored in scored if is_scored]
        if top_n is not None:
            top_ids = {id(e) for e, _ in sorted(kept, key=lambda p: p[1], reverse=True)[:top_n]}
            kept = [(e, s) for e, s in kept if id(e) in top_ids]
        return VocabList(entries=[e for e, _ in kept])
