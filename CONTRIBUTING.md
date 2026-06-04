# Contributing to dynachaos

Thanks for helping improve dynachaos. Please use GitHub Issues for bug reports, feature requests, and support questions: <https://github.com/ricardofrantz/dynachaos/issues>.

## Development setup

```bash
uv sync
```

## Running tests

Run the main test suite with the default Rust-backed package:

```bash
uv run --extra viz pytest tests/ -q
```

Verify the pure-Python fallback path as well:

```bash
DYNACHAOS_NO_RUST=1 uv run --extra viz pytest tests/ -q
```

To exercise the installed Rust extension locally, rebuild it first:

```bash
uv run maturin develop --release
uv run --extra viz pytest tests/ -q
```

## Linting and formatting

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
cargo fmt --manifest-path rust/Cargo.toml -- --check
cargo clippy --manifest-path rust/Cargo.toml -- -D warnings
```

## Code conventions

- Figure scripts should follow the compute/plot pattern with `.npz` caching.
- Cache loaders should use the standardized helper name `_safe_load()`.
- Use `np.random.default_rng(seed)` for random number generation; do not use legacy `np.random` module-level RNG calls.
- Resolve paths relative to `__file__`; do not hardcode local absolute paths.
