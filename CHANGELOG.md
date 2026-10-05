# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-02

### Added

- `VocabList.missing_gloss` — a `VocabList` view of *coverage gaps*: content
  words the list keeps but Whitaker's Words cannot gloss (e.g. the ADJ-tagged
  proper noun `Lavinia` from `Laviniaque`, where `Lavinius/-a/-um` is absent from
  the dictionary). The hook for a future supplementary proper-noun source.
- `PipelineConfig.lemma_overrides` and `vocabbuilder.data.lemma_overrides`
  (`LemmaOverrideRule`, `resolve_lemma_override`) — a small curated,
  POS/morph-gated lemma-correction table applied at aggregation time. Pass `()`
  in `PipelineConfig` to disable (not yet configurable on the spaCy component).
- `examples/plaintext2vocablist.py` — minimal plaintext-to-vocab-list script.
- `VocabEntry.has_gloss` — whether any dictionary gloss was attached.
- `VocabList.glosses_expected` — whether a gloss pipe was expected when the list
  was built: `VocabPipeline` uses `use_glosses`, the `latincy_vocab` component
  checks its own `nlp` for a `whitakers_words` pipe, and `build_vocab_list`
  falls back to `Token.has_extension("gloss")`. Propagated through all
  `by_*`/`filter_*` views.

### Fixed

- Homograph mis-lemmatization: participial `latus` now resolves to `fero`
  (not the noun `latus, lateris`), and deponent `contemplor` forms no longer
  cite active `contemplo`. Noun and adjective `latus` are untouched.

### Changed

- `to_markdown`, `to_dicts`, and `to_json` now **omit gloss-less entries by
  default** when glosses were expected, so a coverage gap no longer renders as a
  glossary line with a blank definition (`lavinia, adj.`). Pass the new
  `include_missing_gloss=True` to keep them. The lexicon-free path
  (`use_glosses=False`), where every entry is intentionally gloss-less, is
  unaffected — `glosses_expected` is `False` there, so all entries still render.

## [0.3.2] - 2026-07-29

### Changed

- Raised the `latincy-lexicon` floor to `>=0.11.0`, which bundles a
  morphological analyzer in `whitakers_words` by default. This recovers glosses
  on forms the upstream lemmatizer mis-lemmatizes (e.g. `contemplemur`, pres.
  pass. subj. of the deponent `contemplor`) by parsing the surface form
  directly, where previously the lemma-keyed lookup alone missed and the token
  was left with no gloss.
- Added Python 3.13 and 3.14 to the supported/classified versions. Verified
  clean (full test suite + a real `la_core_web_lg` run) on 3.11, 3.13, and 3.14.

## [0.3.1] - 2026-07-28

### Fixed

- Added `click` as an explicit direct dependency. `import vocabbuilder` pulls in
  spaCy, which pulls in `typer`/`typer-slim`; on recent `typer` releases the
  transitive chain does not reliably install `click`, causing
  `ModuleNotFoundError: No module named 'click'` at import time in some
  environments. `click>=8.1` is now pinned directly so the import path no
  longer depends on typer's packaging split.

## [0.3.0] - 2026-07-28

### Added

- `PipelineConfig.keep_glossed_propn` (default `True`): proper nouns (`PROPN`)
  are now retained in the vocabulary list when Whitaker's Words supplies a gloss
  for them. This recovers legitimate vocabulary items that the tagger labels as
  proper nouns (e.g. `Musa` → `musa, musae, f.` "muse"), which were previously
  dropped by the blanket `PROPN` exclusion. Set to `False` to restore the prior
  drop-all-`PROPN` behavior.

### Changed

- `VocabEntry.headword` trusts noun-paradigm citation forms for both `NOUN` and
  `PROPN` tokens, since gloss-rescued proper nouns decline normally. Bogus
  paradigms on other parts of speech are still rejected.
- The `latincy_vocab` component accepts and persists the `keep_glossed_propn`
  setting, defaulting to `True` for configs written before the option existed.
- Raised the `latincy-lexicon` floor from `>=0.5.0` to `>=0.9.0`.

### Notes

- Rescue is gated on the presence of a gloss, not on whether a token is a
  common word. Proper names that Whitaker's Words happens to gloss (e.g. `Roma`)
  are therefore also retained.
