# Changelog

This project currently has no git tags and no GitHub Releases. The entries
below describe unreleased private-repository work reconstructed from git
history.

## Unreleased

### Private release posture and documentation

- Renamed the default branch to `main` and hardened project checks around that
  branch convention ([e71ed4e](https://github.com/ricardofrantz/dynachaos/commit/e71ed4e)).
- Added the two-loop, skill-driven improvement goal that now governs agent
  maintenance work ([c8da3a0](https://github.com/ricardofrantz/dynachaos/commit/c8da3a0)).
- Clarified that the repository, package publication, benchmark numbers, and
  citation metadata remain private/provisional until a future public release
  phase ([5ec636a](https://github.com/ricardofrantz/dynachaos/commit/5ec636a)).

### Diagnostics correctness and numerical edge cases

- Added validation for correlation norms, radius grids, Theiler windows, and
  undefined correlation-dimension cases ([15c76fd](https://github.com/ricardofrantz/dynachaos/commit/15c76fd)).
- Hardened time-delay embedding parameter validation with deterministic
  fuzz-style tests ([7ce6f50](https://github.com/ricardofrantz/dynachaos/commit/7ce6f50)).
- Fixed degenerate recurrence auto-thresholding for constant signals
  ([0ff1cb3](https://github.com/ricardofrantz/dynachaos/commit/0ff1cb3)).
- Added metamorphic tests for correlation-integral radius monotonicity and
  Theiler-window valid-pair counts ([c1c316a](https://github.com/ricardofrantz/dynachaos/commit/c1c316a)).

### Rust backend hardening

- Fixed CI and parity-test behavior for pure-Python runs without the Rust
  extension ([0f58bba](https://github.com/ricardofrantz/dynachaos/commit/0f58bba)).
- Added direct validation for ordinal-pattern Rust inputs and matching Python
  wrapper validation ([d511c11](https://github.com/ricardofrantz/dynachaos/commit/d511c11)).
- Recorded that the in-tree Rust extension contains no `unsafe` sites
  ([e0d1fc4](https://github.com/ricardofrantz/dynachaos/commit/e0d1fc4)).
- Avoided debug-build overflow in direct Rust calls with huge Theiler windows
  ([4735647](https://github.com/ricardofrantz/dynachaos/commit/4735647)).

### Pipeline architecture and performance evidence

- Added a reusable NPZ cache contract with required-key validation for figure
  pipeline caches ([d605795](https://github.com/ricardofrantz/dynachaos/commit/d605795)).
- Added architecture, simplification, complexity, profiling, and optimization
  decision artifacts so future agents can distinguish measured work from
  deferred ideas ([acbaa38](https://github.com/ricardofrantz/dynachaos/commit/acbaa38),
  [d0b4d4e](https://github.com/ricardofrantz/dynachaos/commit/d0b4d4e),
  [e7ef0ec](https://github.com/ricardofrantz/dynachaos/commit/e7ef0ec),
  [9ab2bb4](https://github.com/ricardofrantz/dynachaos/commit/9ab2bb4),
  [bc997b5](https://github.com/ricardofrantz/dynachaos/commit/bc997b5)).
- Added a `dynachaos --version` CLI path for installed-package introspection
  ([049782e](https://github.com/ricardofrantz/dynachaos/commit/049782e)).

### Paper and README maintenance

- Relaxed optional dependencies in the vendored journal class so local builds can
  progress farther on smaller TeX installations ([3072d1c](https://github.com/ricardofrantz/dynachaos/commit/3072d1c)).
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
