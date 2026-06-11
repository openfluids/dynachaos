# Changelog

Versions before 0.3.0 were private and unpublished; their entries were
reconstructed from git history.

## 0.3.0 — 2026-06-11

### Licensing

- Relicensed the project from MIT to Apache-2.0 with a `NOTICE` file; earlier
  unpublished versions were MIT.

### User documentation spine

- Added `docs/real-analysis-guide.md`, a user guide for real analyses: input
  expectations, diagnostic choice, long-signal/RQA scaling guidance,
  reliability-metadata interpretation, and positioning. README
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
  test plan ([66ee0e8](https://github.com/ricardofrantz/dynachaos/commit/66ee0e8)).

### Scalable analysis workflow

- Added `dynachaos analyze <config.jsonc>`: config-driven workflow for external
  `.npy`/`.npz` or generated signals writing a stable output directory
  (`results.json`, `metadata.json` with scale/cost and reliability metadata,
  `summary.md`), with explicit failure modes and a dense-RQA scale-envelope
  guard ([2c9d6b6](https://github.com/ricardofrantz/dynachaos/commit/2c9d6b6)).

### Long-signal RQA scaling

- Added `rqa_streaming_from_trajectory`: exact matrix-free RQA (RR, DET, LAM,
  L, TT, ENTR, Lmax and ENTR bins) with Theiler-window support and a recorded
  interface decision plus RSS evidence in `docs/rqa-scaling-design.md` ([d625980](https://github.com/ricardofrantz/dynachaos/commit/d625980)).

### Private release posture and documentation

- Renamed the default branch to `main` and hardened project checks around that
  branch convention ([e71ed4e](https://github.com/ricardofrantz/dynachaos/commit/e71ed4e)).
- Sharpened the internal maintenance workflow, including private-only pushes
  and per-change review/commit discipline
  ([c8da3a0](https://github.com/ricardofrantz/dynachaos/commit/c8da3a0),
  [cdda594](https://github.com/ricardofrantz/dynachaos/commit/cdda594)).
- Clarified that the repository, package publication, benchmark numbers, and
  citation metadata remain private/provisional until a future public release
  phase ([5ec636a](https://github.com/ricardofrantz/dynachaos/commit/5ec636a)).
- Refreshed README backend notes so they match the exported Rust surface and
  documented the verified manuscript build sequence
  ([0c0ae1c](https://github.com/ricardofrantz/dynachaos/commit/0c0ae1c)).

### Diagnostics correctness and numerical edge cases

- Added validation for correlation norms, radius grids, Theiler windows, and
  undefined correlation-dimension cases ([15c76fd](https://github.com/ricardofrantz/dynachaos/commit/15c76fd)).
- Hardened time-delay embedding parameter validation with deterministic
  fuzz-style tests ([7ce6f50](https://github.com/ricardofrantz/dynachaos/commit/7ce6f50)).
- Fixed degenerate recurrence auto-thresholding for constant signals
  ([0ff1cb3](https://github.com/ricardofrantz/dynachaos/commit/0ff1cb3)).
- Rejected non-finite entropy and recurrence diagnostic inputs, invalid
  recurrence thresholds, invalid 0-1 test parameters, and zero-MSD 0-1 test
  regressions found during a systematic bug hunt
  ([4ff83ad](https://github.com/ricardofrantz/dynachaos/commit/4ff83ad),
  [b473141](https://github.com/ricardofrantz/dynachaos/commit/b473141),
  [8dc0541](https://github.com/ricardofrantz/dynachaos/commit/8dc0541),
  [93c02e2](https://github.com/ricardofrantz/dynachaos/commit/93c02e2)).
- Added entropy and recurrence metamorphic tests for translation and scaling
  invariants ([9c8f596](https://github.com/ricardofrantz/dynachaos/commit/9c8f596)).
- Tightened RQA recurrence-matrix and line-threshold validation across Python
  and direct Rust entry points
  ([66f4d6e](https://github.com/ricardofrantz/dynachaos/commit/66f4d6e),
  [d1340b1](https://github.com/ricardofrantz/dynachaos/commit/d1340b1)).
- Added metamorphic tests for correlation-integral radius monotonicity and
  Theiler-window valid-pair counts ([c1c316a](https://github.com/ricardofrantz/dynachaos/commit/c1c316a)).

### Rust backend hardening

- Fixed CI and parity-test behavior for pure-Python runs without the Rust
  extension ([0f58bba](https://github.com/ricardofrantz/dynachaos/commit/0f58bba)).
- Added direct validation for ordinal-pattern Rust inputs and matching Python
  wrapper validation ([d511c11](https://github.com/ricardofrantz/dynachaos/commit/d511c11)).
- Recorded that the in-tree Rust extension contains no `unsafe` sites
  ([e0d1fc4](https://github.com/ricardofrantz/dynachaos/commit/e0d1fc4),
  [0f58ef2](https://github.com/ricardofrantz/dynachaos/commit/0f58ef2)).
- Avoided debug-build overflow in direct Rust calls with huge Theiler windows
  ([4735647](https://github.com/ricardofrantz/dynachaos/commit/4735647)).
- Validated Rust AMI inputs directly and recorded a Rust undefined-behavior audit
  ([dd8ee6c](https://github.com/ricardofrantz/dynachaos/commit/dd8ee6c),
  [66f4d6e](https://github.com/ricardofrantz/dynachaos/commit/66f4d6e)).

### Pipeline architecture and performance evidence

- Added a reusable NPZ cache contract with required-key validation for figure
  pipeline caches ([d605795](https://github.com/ricardofrantz/dynachaos/commit/d605795)).
- Adopted the cache contract in the circle-map figure pipeline
  ([eff0b34](https://github.com/ricardofrantz/dynachaos/commit/eff0b34)).
- Added architecture, simplification, complexity, profiling, and optimization
  decision artifacts so future maintainers can distinguish measured work from
  deferred ideas ([acbaa38](https://github.com/ricardofrantz/dynachaos/commit/acbaa38),
  [d0b4d4e](https://github.com/ricardofrantz/dynachaos/commit/d0b4d4e),
  [e7ef0ec](https://github.com/ricardofrantz/dynachaos/commit/e7ef0ec),
  [9ab2bb4](https://github.com/ricardofrantz/dynachaos/commit/9ab2bb4),
  [bc997b5](https://github.com/ricardofrantz/dynachaos/commit/bc997b5)).
- Captured further architecture, simplification, complexity, profiling, and
  optimization artifacts, including an RQA count-reuse optimization measured at
  roughly 1.3--1.5x faster in the profiled range
  ([2b56ac5](https://github.com/ricardofrantz/dynachaos/commit/2b56ac5),
  [60dd71f](https://github.com/ricardofrantz/dynachaos/commit/60dd71f),
  [ee0f98c](https://github.com/ricardofrantz/dynachaos/commit/ee0f98c),
  [4c7a153](https://github.com/ricardofrantz/dynachaos/commit/4c7a153),
  [8295d26](https://github.com/ricardofrantz/dynachaos/commit/8295d26)).
- Added a `dynachaos --version` CLI path for installed-package introspection
  ([049782e](https://github.com/ricardofrantz/dynachaos/commit/049782e)).

### Paper and README maintenance

- Relaxed optional dependencies in the vendored journal class so local builds can
  progress farther on smaller TeX installations ([3072d1c](https://github.com/ricardofrantz/dynachaos/commit/3072d1c)).
- Removed optional `enumitem`/TikZ manuscript dependencies and regenerated the
  tracked paper PDF after a clean pdflatex/BibTeX build
  ([69ee577](https://github.com/ricardofrantz/dynachaos/commit/69ee577)).
- Added Python, Rust unsafe, and LaTeX review artifacts for Loop 1 of the
  improvement campaign ([e5164df](https://github.com/ricardofrantz/dynachaos/commit/e5164df),
  [e0d1fc4](https://github.com/ricardofrantz/dynachaos/commit/e0d1fc4),
  [3072d1c](https://github.com/ricardofrantz/dynachaos/commit/3072d1c)).

### Earlier project build-out

- Created the initial dynachaos codebase, maps, diagnostics, visual assets,
  paper materials, and CI/release workflow scaffolding
  ([f8c3acf](https://github.com/ricardofrantz/dynachaos/commit/f8c3acf),
  [58da059](https://github.com/ricardofrantz/dynachaos/commit/58da059)).
- Added paper figure polish, manuscript synchronization, entropy-family
  diagnostics, JSONC example configs, Rust solver timing work, and benchmark
  corrections across the pre-goal history
  ([7fbddbd](https://github.com/ricardofrantz/dynachaos/commit/7fbddbd),
  [6073de8](https://github.com/ricardofrantz/dynachaos/commit/6073de8),
  [7b87562](https://github.com/ricardofrantz/dynachaos/commit/7b87562),
  [3d64ab5](https://github.com/ricardofrantz/dynachaos/commit/3d64ab5),
  [8e99790](https://github.com/ricardofrantz/dynachaos/commit/8e99790),
  [1d138e3](https://github.com/ricardofrantz/dynachaos/commit/1d138e3)).

## Evidence Sources

- `git log --reverse --oneline --decorate=no --no-merges`
- `git tag --list --sort=-creatordate`
- `gh release list --limit 50`
- `README.md`, `pyproject.toml`, `.github/workflows/ci.yml`,
  `.github/workflows/publish.yml`
