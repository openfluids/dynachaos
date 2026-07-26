# Changelog

Versions before 0.4.0 were private and unpublished; their entries were
reconstructed from git history. 0.4.0 is the first public release.

## 0.4.0 — 2026-07-26

First public release, under the openfluids organization and on PyPI.

### Project

- Moved to `github.com/openfluids/dynachaos` and published to PyPI as
  `dynachaos`. Install with `pip install dynachaos`.
- Removed material specific to a separate, unpublished manuscript that used
  this package. Citations to Kunihiko Kaneko's published papers throughout
  `src/` are unchanged — they are scientific attribution.
- `CITATION.cff` now cites the software itself rather than an unpublished
  manuscript.
- Reframed the `figures/` tree as a reproduction gallery: section-indexed
  reproductions of Kaneko's published work that double as golden test data.
- Removed changelog commit links that pointed at a repository and commit
  hashes that no longer exist.

### Dependencies

- Rust edition 2021 → 2024; pyo3 0.28 → 0.29, numpy 0.28 → 0.29,
  rayon 1.10 → 1.12, ndarray → 0.17.2.
- Development tooling moved to current releases (maturin, ruff, pytest).
- Runtime floors for numpy and scipy are unchanged; they are minimums, not
  targets, and raising them would force needless upgrades on users.

### Fixed

- `TestVersion` asserted a hardcoded `"0.2.0"` while the package reported
  `0.3.0`, so the suite was red from the 0.3.0 release prep onward. It now
  compares `__version__` against the installed package metadata, which is the
  drift this test existed to catch and which no longer breaks on a bump.

### Packaging

- Renamed the release workflow to `release.yml`, matching the PyPI trusted
  publisher and the other openfluids packages.
- Excluded `figures/`, `tests/`, and benchmark results from the sdist; the
  tracked figure data would otherwise have pushed it past PyPI's size limit.

## 0.3.0 — 2026-06-11

### Licensing

- Relicensed the project from MIT to Apache-2.0 with a `NOTICE` file; earlier
  unpublished versions were MIT.

### User documentation spine

- Added `docs/real-analysis-guide.md`, a user guide for real analyses: input
  expectations, diagnostic choice, long-signal/RQA scaling guidance,
  reliability-metadata interpretation, and package positioning. README
  quickstart now runs the tested external-signal recipe; all shown commands
  are executed in checks or explicitly marked local/full-run.

### RQA consolidation

- Consolidated the matrix-free RQA scan into one shared core
  (`_trajectory_rqa_scan`); `rqa_streaming_from_trajectory` now delegates to
  it. Fixed `rqa_from_trajectory`'s `eps=None` percentile to match the dense
  `recurrence_matrix` squareform multiset, with a pinning regression test.

### Example recipes

- Added tested example recipes under `examples/recipes/`: external-signal
  analysis with diagnostic selection and reliability metadata, and a
  long-signal/downsampled streaming-RQA recipe that stays inside the dense
  recurrence memory envelope; smoke-tested by new `tests/test_examples.py`.

### Rust acceleration roadmap

- Added a measured Rust-kernel acceleration roadmap
  (`docs/rust-acceleration-roadmap.md`) and the subprocess-isolated hotspot
  profiler `benchmarks/rust_hotspot_profile.py` with checked-in artifacts;
  streaming RQA is ranked as the next port candidate with a recorded parity
  test plan.

### Scalable analysis workflow

- Added `dynachaos analyze <config.jsonc>`: config-driven workflow for external
  `.npy`/`.npz` or generated signals writing a stable output directory
  (`results.json`, `metadata.json` with scale/cost and reliability metadata,
  `summary.md`), with explicit failure modes and a dense-RQA scale-envelope
  guard.

### Long-signal RQA scaling

- Added `rqa_streaming_from_trajectory`: exact matrix-free RQA (RR, DET, LAM,
  L, TT, ENTR, Lmax and ENTR bins) with Theiler-window support and a recorded
  interface decision plus RSS evidence in `docs/rqa-scaling-design.md`.

### Private release posture and documentation

- Renamed the default branch to `main` and hardened project checks around that
  branch convention.
- Sharpened the internal maintenance workflow, including private-only pushes
  and per-change review/commit discipline.
- Clarified that the repository, package publication, benchmark numbers, and
  citation metadata remain private/provisional until a future public release
  phase.
- Refreshed README backend notes so they match the exported Rust surface and
  documented the verified manuscript build sequence.

### Diagnostics correctness and numerical edge cases

- Added validation for correlation norms, radius grids, Theiler windows, and
  undefined correlation-dimension cases.
- Hardened time-delay embedding parameter validation with deterministic
  fuzz-style tests.
- Fixed degenerate recurrence auto-thresholding for constant signals.
- Rejected non-finite entropy and recurrence diagnostic inputs, invalid
  recurrence thresholds, invalid 0-1 test parameters, and zero-MSD 0-1 test
  regressions found during a systematic bug hunt.
- Added entropy and recurrence metamorphic tests for translation and scaling
  invariants.
- Tightened RQA recurrence-matrix and line-threshold validation across Python
  and direct Rust entry points.
- Added metamorphic tests for correlation-integral radius monotonicity and
  Theiler-window valid-pair counts.

### Rust backend hardening

- Fixed CI and parity-test behavior for pure-Python runs without the Rust
  extension.
- Added direct validation for ordinal-pattern Rust inputs and matching Python
  wrapper validation.
- Recorded that the in-tree Rust extension contains no `unsafe` sites.
- Avoided debug-build overflow in direct Rust calls with huge Theiler windows.
- Validated Rust AMI inputs directly and recorded a Rust undefined-behavior audit.

### Pipeline architecture and performance evidence

- Added a reusable NPZ cache contract with required-key validation for figure
  pipeline caches.
- Adopted the cache contract in the circle-map figure pipeline.
- Added architecture, simplification, complexity, profiling, and optimization
  decision artifacts so future maintainers can distinguish measured work from
  deferred ideas.
- Captured further architecture, simplification, complexity, profiling, and
  optimization artifacts, including an RQA count-reuse optimization measured at
  roughly 1.3--1.5x faster in the profiled range.
- Added a `dynachaos --version` CLI path for installed-package introspection.

### Paper and README maintenance

- Relaxed optional dependencies in the vendored journal class so local builds can
  progress farther on smaller TeX installations.
- Removed optional `enumitem`/TikZ manuscript dependencies and regenerated the
  tracked paper PDF after a clean pdflatex/BibTeX build.
- Added Python, Rust unsafe, and LaTeX review artifacts for Loop 1 of the
  improvement campaign.

### Earlier project build-out

- Created the initial dynachaos codebase, maps, diagnostics, visual assets,
  paper materials, and CI/release workflow scaffolding.
- Added paper figure polish, manuscript synchronization, entropy-family
  diagnostics, JSONC example configs, Rust solver timing work, and benchmark
  corrections across the pre-goal history.

## Evidence Sources

- `git log --reverse --oneline --decorate=no --no-merges`
- `git tag --list --sort=-creatordate`
- `gh release list --limit 50`
- `README.md`, `pyproject.toml`, `.github/workflows/ci.yml`,
  `.github/workflows/publish.yml`
