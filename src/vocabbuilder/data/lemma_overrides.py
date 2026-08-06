"""Curated lemma-override table for known spaCy/LatinCy homograph mis-lemmas.

Deliberately small and hand-curated -- NOT a general WSD system. Each rule
corrects one documented case where the upstream lemmatizer commits to the
wrong headword string for a token it otherwise tags correctly (right POS,
right morphology), by keying on (spacy lemma, POS, required morph features)
rather than surface form alone, so genuine other-POS/other-sense uses of the
same wordform are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LemmaOverrideRule:
    spacy_lemma: str  # token.lemma_ as emitted upstream (the bug)
    pos: str  # required token.pos_ (UPOS) gate
    corrected_lemma: str  # must be a real headword key in latincy_lexicon.build_lexicon()
    require_morph: dict[str, str] = field(default_factory=dict)
    # ^ ALL of these (k: v) must be present in the token's parsed morph dict
    # for the rule to fire (subset match).
    note: str = ""  # rationale, for maintainers only


DEFAULT_LEMMA_OVERRIDES: tuple[LemmaOverrideRule, ...] = (
    LemmaOverrideRule(
        spacy_lemma="latus",
        pos="VERB",
        require_morph={"VerbForm": "Part"},
        corrected_lemma="fero",
        note=(
            "The lemmatizer correctly tags the perfect passive participle of "
            "fero as VERB/VerbForm=Part but leaves the lemma string as the "
            "surface-identical noun/adj 'latus' instead of resolving to "
            "'fero'. Gating on VerbForm=Part keeps genuine NOUN 'latus' "
            "('side, flank') and ADJ 'latus' ('wide, broad') untouched."
        ),
    ),
    LemmaOverrideRule(
        spacy_lemma="contemplo",
        pos="VERB",
        require_morph={"Voice": "Pass"},
        corrected_lemma="contemplor",
        note=(
            "DICTLINE carries both an active-transitive 'contemplo' and the "
            "classical deponent 'contemplor' under the same stem; the "
            "lemmatizer sometimes commits to the active headword for "
            "genuinely deponent (always-passive-morphology) forms. Gating on "
            "Voice=Pass fixes the deponent case; a true passive-voice use of "
            "transitive 'contemplo' would be mis-corrected too -- accepted "
            "v1 trade-off, no such use is expected in target corpora."
        ),
    ),
)


def _build_index(
    rules: tuple[LemmaOverrideRule, ...],
) -> dict[tuple[str, str], tuple[LemmaOverrideRule, ...]]:
    index: dict[tuple[str, str], list[LemmaOverrideRule]] = {}
    for rule in rules:
        index.setdefault((rule.spacy_lemma, rule.pos), []).append(rule)
    return {k: tuple(v) for k, v in index.items()}


_DEFAULT_INDEX = _build_index(DEFAULT_LEMMA_OVERRIDES)


def resolve_lemma_override(
    lemma: str,
    pos: str,
    morph: dict[str, str],
    rules: tuple[LemmaOverrideRule, ...] = DEFAULT_LEMMA_OVERRIDES,
) -> str | None:
    """The corrected lemma for (lemma, pos, morph), or None if no rule fires."""
    index = _DEFAULT_INDEX if rules is DEFAULT_LEMMA_OVERRIDES else _build_index(rules)
    for rule in index.get((lemma, pos), ()):
        if all(morph.get(k) == v for k, v in rule.require_morph.items()):
            return rule.corrected_lemma
    return None
