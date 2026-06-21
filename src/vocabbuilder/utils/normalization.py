"""Text normalization utilities for Latin text processing."""

from __future__ import annotations

from latincy_preprocess import normalize_uv, normalize_vu

# UPOS (spaCy) → gloss file POS mapping
UPOS_TO_GLOSS_POS: dict[str, str] = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ": "adj",
    "ADV": "adv",
    "PROPN": "name",
    "DET": "det",
    "PRON": "pron",
    "ADP": "prep",
    "CCONJ": "conj",
    "SCONJ": "conj",
    "NUM": "num",
    "INTJ": "intj",
}

# Reverse mapping for display
GLOSS_POS_TO_UPOS: dict[str, str] = {v: k for k, v in UPOS_TO_GLOSS_POS.items()}

# UPOS → textbook part-of-speech abbreviation (for vocab-list display).
UPOS_TO_ABBREV: dict[str, str] = {
    "NOUN": "n.",
    "VERB": "v.",
    "AUX": "v.",
    "ADJ": "adj.",
    "DET": "det.",
    "ADV": "adv.",
    "ADP": "prep.",
    "CCONJ": "conj.",
    "SCONJ": "conj.",
    "PRON": "pron.",
    "NUM": "adj.",  # numerals shown as adjectives (maintainer decision)
    "INTJ": "interj.",
    "PART": "part.",
}

# Common enclitics in Latin
ENCLITICS = ("-que", "-ne", "-ve")


def to_u_form(text: str) -> str:
    """Normalize text to u-form (v→u, j→i) to match spaCy conventions.

    Uses latincy-preprocess for v→u; handles j→i separately.
    """
    result = normalize_vu(text)
    result = result.replace("j", "i").replace("J", "I")
    return result


def to_v_form(text: str) -> str:
    """Normalize text to v-form (u→v where appropriate) for display.

    Uses latincy-preprocess which applies contextual Latin u/v rules.
    """
    return normalize_uv(text)


def strip_enclitic(form: str) -> tuple[str, str | None]:
    """Strip common Latin enclitics from a word form.

    Returns (base_form, enclitic) where enclitic is None if none found.
    """
    lower = form.lower()
    for enc in ENCLITICS:
        suffix = enc[1:]  # remove the dash
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            base = form[: -len(suffix)]
            return base, suffix
    return form, None


def upos_to_gloss_pos(upos: str) -> str | None:
    """Convert spaCy UPOS tag to gloss file POS string."""
    return UPOS_TO_GLOSS_POS.get(upos)


def upos_to_abbrev(upos: str) -> str:
    """Textbook POS abbreviation for a UPOS tag (``""`` if unknown)."""
    return UPOS_TO_ABBREV.get(upos, "")
