# Running the kernels in a browser: architecture contract

This note records the decisions for computing figures in the reader's browser
with the same Rust kernels the paper used. It is written before most of the
implementation, so that the kernels are built to a settled contract instead of
one reverse-engineered afterwards. Update it when a decision here changes.

## Why do this at all

The gallery ships pre-rendered PNGs plus decimated JSON (`scripts/export_figure_data.py`
writes `site/data/`). A reader can look at a parameter plane but cannot zoom into
it, and cannot move a parameter and watch the attractor change. Both are cheap
to compute and expensive to ship as data: the Arnold-tongue plane is a 2000x1000
grid today, and any zoom deeper than that grid has no data behind it.

Compiling the kernels to WebAssembly removes the shipping step. The reader's
machine recomputes the visible region at the resolution of their screen. The
same code that produced the published figure produces what they explore, so the
two cannot drift.

## Crate layout

The split landed in Phase 0 (2026-09).

| crate | contents | depends on |
|---|---|---|
| `rust/core` | every kernel; slices in, plain Rust out; no pyo3, no numpy | ndarray, rayon (optional) |
| `rust/py` | `#[pyfunction]` wrappers, numpy conversion, `py.detach` around core calls | `dynachaos-core`, pyo3, numpy |
| `rust/wasm` | `wasm-bindgen` exports over `rust/core`; one export so far, `rotation_number_tile` | `dynachaos-core` with `--no-default-features` |

`rust/core` carries 1491 lines of kernel code; `rust/py` carries 629 lines of
binding with no arithmetic in it. Keeping the arithmetic in exactly one place is
the point of the split: a browser and a Python caller cannot disagree about a
number if only one implementation exists.

`rust/wasm` gets its own `[workspace]`, so a browser build never depends on the
state of the native build. This follows the FrankenSim layout
(`crates/fs-wasm` in github.com/Dicklesworthstone/frankensim), which is the
closest published example of shipping real Rust numerics to a browser.

### rayon is optional

`rust/core` declares `rayon` as an optional dependency behind a `parallel`
feature, `default = ["parallel"]`. The Python extension keeps the default and
runs parallel; a wasm build compiles with `--no-default-features` and takes a
sequential path. `cargo build -p dynachaos-core --no-default-features` is
checked in CI so this path cannot rot.

Both paths call the same per-item function. Only the accumulation differs: the
parallel path folds per-thread partial sums and reduces them pairwise, the
sequential path adds left to right. **Floating-point addition is not
associative, so the two can differ in the last digit for kernels that sum
floats** (`fuzzy_entropy_sum`, the correlation sums). This is inherent to
parallel reduction, not a defect, and it is why the parity check below states a
tolerance for those kernels instead of demanding equality.

## The kernel contract

Every function exported to the browser obeys these rules. They come from the
FrankenSim `fs-wasm` contract and exist because a web page is a hostile caller:
its arguments come from a URL, a slider, or a stranger.

1. **Clamp every input.** No argument is trusted. A grid size, an iteration
   count and a parameter value each get a documented valid range and are
   clamped into it, not rejected.
2. **Cap every loop.** An iteration count has a hard ceiling. A browser tab that
   locks up is worse than a coarse picture.
3. **Return one flat `Vec<f64>`** with the layout written in the doc comment,
   including where each block starts and what the first elements mean. The
   JavaScript side reads that layout; there is no shared struct.
4. **No allocation surprises.** Size the output from the clamped inputs so the
   caller can predict the buffer.

## Threads: worker tiling, not SharedArrayBuffer

WebAssembly threads put the linear memory in a `SharedArrayBuffer`, which
browsers only expose to a cross-origin isolated page. Isolation requires the
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` response
headers, and **GitHub Pages cannot set response headers**. The gallery is
served from GitHub Pages.

Decision: parallelism comes from a pool of independent web workers, each
holding its own wasm instance and computing a tile of the domain. Tiles need no
shared memory because they share no state. This also keeps the main thread free,
so the page never freezes while a figure computes.

| option | works on GitHub Pages | cost | decision |
|---|---|---|---|
| `SharedArrayBuffer` + `wasm-bindgen-rayon` | no, headers cannot be set | — | reject |
| `coi-serviceworker` (service worker forges the headers) | yes | a reload on the visitor's first arrival | keep as a later upgrade, not now |
| pool of independent workers, one tile each | yes | tiles must be independent | **choose** |

The service-worker route is real and documented, and it is the way in if a
figure ever needs genuinely shared memory. It costs the visitor a page reload
the first time they arrive, which is a poor trade for figures that tile cleanly.

### Live JS runtime

The worker pool is hand-written ESM in `site-src/live/`, copied verbatim by
`scripts/build_paper.py` into `site/live/`. No bundler, no npm. Three files:

| file | job |
|---|---|
| `scheduler.js` | Pure function: state in, commands out. No DOM, no Worker, no timer. Node unit-tests this. |
| `pool.js` | `N = min(navigator.hardwareConcurrency, 8)` workers, each with its own wasm instance. A pan or zoom bumps the generation; workers stay alive so the in-flight tile can finish, and a late result is dropped. `terminate()` runs only in `destroy()`. |
| `tile-worker.js` | Loads `site/wasm/dynachaos_wasm.js`, calls `rotation_number_tile`, transfers the `Float64Array` back. Reads the 4-element header rather than trusting the request. |

Tiles cover the viewport at a pyramid of levels. Level 0 is one tile over the
whole view; each finer level splits 2×2. The scheduler issues every coarser
tile before any finer one, and issues each visible tile exactly once per
generation. A viewport change increments the generation; workers are not
terminated, and results that carry an older generation are dropped, not painted.
In-flight bookkeeping is `{ id, generation }`, so a late result cannot evict
the current generation's entry for the same tile. Degenerate viewports
(`omegaMax <= omegaMin` or `kMax <= kMin`) are rejected. An error reply, a
worker-level failure, or a malformed worker message frees its tile and is
counted as a drop, not left in flight. A failed worker is not handed more work.

Debug telemetry (`?debug=1` or `localStorage.dynachaosDebug`) logs per-tile
compute milliseconds, worker count, queue depth, dropped generations (viewport
changes that abandoned work), and dropped tiles (stale or error results).
There is no `SharedArrayBuffer` anywhere in this path.

## WebGPU is a fast path, never the baseline

As of 2026-09: Chromium ships WebGPU (113+, and Android 121+), Safari turned it
on by default in Safari 26 (2025-09), and **Firefox does not ship it in the
release channel**. A figure that only works in Chromium is not acceptable for a
paper.

Decision: CPU kernels in WebAssembly drawing to a canvas are the baseline and
must be sufficient for every live figure. WebGPU may be added later for a
figure that genuinely needs it, always behind a capability check with the CPU
path as the fallback.

## The PNG stays canonical

Every live figure keeps its published PNG, and the PNG is what the document
means.

- The PNG paints first, so the reader sees the figure immediately.
- Live mode is opt-in per figure and replaces the canvas contents afterwards.
- Live mode is off under `prefers-reduced-data` (the page already reads it) and
  absent without JavaScript.
- A figure gets exactly **one** interactive mode. Where a figure already has a
  decimated-JSON chart, live mode replaces that chart; the two never stack.

The archived reproduction pipeline and the committed `.npz` caches stay the
record of what the paper computed. Live views are companions to it.

## Proving the browser agrees with the paper

A live figure is only worth showing if its numbers are the paper's numbers. CI
checks this directly: a small fixed tile is evaluated by the native `rust/core`
test binary and by the wasm module, and the two are compared element by element.

Build the native reference **without** `target-cpu=native`. The repository's
`.cargo/config.toml` enables it for local builds, and it lets the compiler
contract multiplications and additions into fused instructions that `wasm32`
does not have. Comparing against a natively-tuned build measures the compiler,
not the port.

### The two builds cannot agree everywhere, and that is physics

Measured on 2026-09-08 over a 64 x 64 tile of the Arnold-tongue plane
(`scripts/check_wasm_parity.py`):

| comparison | cells differing at all | worst difference |
|---|---|---|
| NumPy vs native Rust | 0 of 4096 | 0 |
| native Rust vs wasm, `K <= 0.159` | 2 of 2176 | 7.8e-16 |
| native Rust vs wasm, `K > 0.159` | 15 of 1920 | 2.9e-2 |

The check itself prints a smaller number above the critical line, **13 of
1920**, because it counts a cell only past a `1e-12` floor. The other two
differ by less than that, which is the same orbit through a slightly different
sine rather than a diverged one.

NumPy and native Rust agree exactly because they call the same platform sine.
The wasm module carries its own sine, compiled into it, and that one differs by
about one unit in the last place. Below `K = 1 / (2 * pi)` the map is
invertible and the rotation number is well conditioned, so a one-bit difference
stays a one-bit difference. Above it the map is not invertible, the orbit is
chaotic, and the same one bit grows by fourteen orders of magnitude over 700
iterations.

So the honest claim is narrower than "the browser returns the paper's numbers":

- **Where the dynamics are locked, it does** — and those regions are the
  subject of the figure. The tongues are exactly where the reader is looking.
- **In the chaotic sea, a single pixel is not reproducible** across sine
  implementations. It is not reproducible across compilers or CPUs either. That
  is a property of the system, not of this port, and a figure that showed
  otherwise would be hiding it.

The check asserts each of those separately: a tight bound below the critical
line, and only that divergence stays rare above it. Never loosen the
subcritical bound to make a red run pass. A difference there is well
conditioned, which means it is a defect, not sensitivity. A deliberate `1e-7`
perturbation of the starting angle trips both halves, so the check is known to
work rather than assumed to.

This is also why a live figure must not be presented as a re-derivation of the
published pixel values in the chaotic region. It shows the same system computed
by the same code; it does not promise the same last digits where the system
itself refuses to have any.

## Measuring performance honestly

Kernel timings in this project are small (a Grassberger-Procaccia pass at
N=1000 takes under a millisecond) and easy to mismeasure. Two effects dominate:

- **Cold start.** A single un-warmed call measures thread-pool spin-up, first
  page touches and a cold cache. Warm up, then take a median of repeats.
- **Oversubscription.** rayon defaults to one thread per core. On the 24-core
  development box the same kernel runs 0.67 ms on 8 threads and 0.88 ms on 24,
  because the threads contend. More threads is not more speed for a workload
  this short.

Measured on the development box (median of 15, after 3 warm-up calls,
Grassberger-Procaccia, logistic, N=1000):

| threads | 1 | 2 | 4 | 8 | 24 |
|---|---|---|---|---|---|
| time (ms) | 2.68 | 1.38 | 0.81 | **0.67** | 0.88 |

A browser has far fewer workers than this box has cores, so the useful figure
for planning a live figure is the 2-to-8-thread range, not the single-thread
number and not the 24-thread number.
