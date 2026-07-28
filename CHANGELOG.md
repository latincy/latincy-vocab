# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
