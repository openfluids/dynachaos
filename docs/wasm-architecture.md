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
| `rust/wasm` | `wasm-bindgen` exports over `rust/core`; `rotation_number_tile` for the picture, `rotation_number_point` for the quoted readout, `zero_one_k` for the 0-1 test for chaos, `correlation_counts`, `apen_counts`, `fuzzy_entropy_sum`, `ordinal_distribution`, `diagonal_lines`, `vertical_lines`, `multifractal_moments`, `ami_histogram`, `select_dimension_cao`, `delayed_logistic_attractor_tile`, `torus_doubling_attractor_tile`, `modulated_circle_rotation_tile` and `cml_spacetime_tile` for the diagnostics, map-attractor and space-time panels | `dynachaos-core` with `--no-default-features` |

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

`zero_one_k` is the exception to "cap every loop": its cost is `n_c` FFTs of
a power-of-two length below `4N`, so the `N <= 20000` and `n_c <= 100`
truncations bound the work on their own and no pair budget is needed. Its
clamps are `phi` to 20000 samples, `c_values` to 100 frequencies, and
`n_cut` to `[2, N]` — a request below 2 or above `N` comes back clamped,
never refused, and the header reports the value used. A series shorter than
three samples cannot produce a statistic and returns an empty array.

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
`scripts/build_paper.py` into `site/live/`. No bundler, no npm. Seven files:

| file | job |
|---|---|
| `scheduler.js` | Pure function: state in, commands out. No DOM, no Worker, no timer. Node unit-tests this. `lockOmega` switches it to a 1-D mode for the staircase: tiles carry `nOmega = 1` at the viewport's fixed Omega and the pyramid refines K only. |
| `pool.js` | `N = min(navigator.hardwareConcurrency, 8)` workers, each with its own wasm instance. A pan or zoom bumps the generation; workers stay alive so the in-flight tile can finish, and a late result is dropped. `terminate()` runs only in `destroy()`. |
| `tile-worker.js` | Loads `site/wasm/dynachaos_wasm.js`, calls `rotation_number_tile`, transfers the `Float64Array` back. Reads the 4-element header rather than trusting the request. |
| `raster.js` | Pure tile→pixel mapping, `liveTileColorKey`, and the tile lookup used as the readout fallback. Row 0 of a tile is `kMin`, the bottom; the canvas y axis points down. Node unit-tests this. |
| `point.js` | Main-thread lazy load of the wasm glue; synchronous `sample()` of one (Omega, K) point with lock detection on and the display stop off. Hover and keyboard readout share this path. |
| `live-figure.js` | The glue mountLive calls: the onPaint store update with its paint-sequence stamp, the readout fallback (point kernel, else the raster lookup), the onView generation reset, and the staircase's 1-D pieces (view mapping, tile→trace fold, y-fit, parameter wiring). No DOM at import time; node unit-tests this. |
| `params.js` | Pure parameter plumbing: spec clamping, debounce, and URL-hash serialise/parse (`#section&fig:<id>.<name>=<value>`). Node unit-tests this. |

The paper page loads `pool.js` from `app.js` with a dynamic `import()` when the
reader presses interact on `figure#fig:arnold_tongues`. `Plot()` keeps axes,
colourbar, the K_c line, readout, drag zoom, keyboard and reset; only the heatmap
raster comes from the pool. Colour scale is fixed at ρ ∈ [0, 1]. Live mode
replaces the JSON chart; it never fetches it. Under `prefers-reduced-data` the
page does not create the pool or download the wasm module, and falls back to
the JSON chart if present.

Tiles cover the viewport at a pyramid of levels. Level 0 is one tile over the
whole view; each finer level splits 2×2. Coarser levels use fewer cells: the
finest level uses `tileCells`, and each coarser level halves that count, so
the first paint is one cheap tile on one worker. The live Arnold-tongues
figure uses five levels. The scheduler issues every coarser tile before any
finer one, and issues each visible tile exactly once per generation. A
viewport change increments the generation; workers are not terminated, and
results that carry an older generation are dropped, not painted. In-flight
bookkeeping is `{ id, generation }`, so a late result cannot evict the
current generation's entry for the same tile. Degenerate viewports
(`omegaMax <= omegaMin` or `kMax <= kMin`) are rejected. An error reply, a
worker-level failure, or a malformed worker message frees its tile and is
counted as a drop, not left in flight. A failed worker is not handed more work.
Live tiles are coloured once; redraw blits the cached ImageData. The cache
key is `liveTileColorKey` in `raster.js` and does not include the page theme,
because live colour is a fixed viridis ramp.

Debug telemetry (`?debug=1` or `localStorage.dynachaosDebug`) logs per-tile
compute milliseconds, worker count, queue depth, dropped generations (viewport
changes that abandoned work), and dropped tiles (stale or error results).
There is no `SharedArrayBuffer` anywhere in this path.

The devil's staircase (`figure#fig:devils_staircase`) is the second live
figure and the first 1-D one: ρ(A) at a reader-chosen drive frequency D,
drawn as a line with `Plot()`'s axes and readout rather than a heatmap. A
slider (range + number field, keyboard accessible) sets D ∈ [0, 0.5],
default 0.25; a debounced apply pushes `{omegaMin: D, omegaMax: D, kMin,
kMax}` into the pool — the scheduler's `lockOmega` mode emits `nOmega = 1`
tiles, so Omega = D and K = A exactly — and the generation drop clears the
stale curve. Painted tiles fold into one sorted polyline; the first paint
of a generation refits the y domain. The readout quotes `point.sample(D, A)`
at the cursor's A. The Lyapunov panel has no wasm kernel yet, so the
published PNG stays below the live curve with a note saying it is not live.
Parameter state serialises into the URL hash (`#section&fig:<id>.D=<v>`) so
a shared link restores the exact view; the scroll-spy preserves the
`&key=value` tokens when it rewrites the section id. Under
`prefers-reduced-data` no slider is created and the PNG stays.

The delayed-logistic attractors (`figure#fig:delayed_logistic_attractors`)
are the third live figure and the first point cloud: the (x, y) orbit of
the delayed logistic map at a reader-chosen D ∈ [1.4, 3.5] (the kernel's
own clamp), A = 0.3 fixed, with an iteration-count control bounded by the
kernel's 4096 plotted states. It does not use the tile pool: one
`delayed_logistic_attractor_tile` call per parameter set is at most
20000 + 4096 map steps, microseconds of work, so the worker round trip
would buy nothing. The debounce collapses a slider burst into one call and
a sequence number drops a stale result. The start state is the published
convention (fp + 0.01, fp − 0.01) at the analytic fixed point, which the
canvas marks in vermilion over the slate cloud. The map uses only +, −, *,
so the wasm orbit is bit-identical to the npz trajectories;
`scripts/check_wasm_delayed_logistic.py` asserts exactly that at every
committed D in the wasm-parity CI job. Parameter state rides the URL hash
like the staircase's; under `prefers-reduced-data` no slider is created
and the PNG stays.

The torus-doubling attractors (`figure#fig:map_I_attractors`) are the
fourth live figure: the (X, Y) projection of map (I) or map (IV) at a
reader-chosen D, drawn by `torus_doubling_attractor_tile` on the main
thread like the delayed-logistic cloud. A map selector — the page's first
non-slider control, a `<select>` — switches between map (I) (A = 0.4,
x0 = (0.5, 0.5, 0.5), D ∈ [1.9, 2.25]) and map (IV) (A = 0.3,
x0 = (0.5, 0.45, 0.52, 0.48), D ∈ [1.48, 1.53]); each window is the
published sweep inside the kernel's [1.48, 2.25] clamp, and switching maps
resets D to the new map's published default rather than clamping the old
value into the new window. An iteration-count control is bounded by the
kernel's 4096 plotted states. Both maps project the state onto (X, Y) —
components 0 and 1 — on one fixed domain, the union of both committed npz
extents, so the selector compares the maps on the paper's axes. The maps
use only +, −, *, so the wasm orbit is bit-identical to the npz
trajectories; `scripts/check_wasm_torus_doubling.py` asserts exactly that
at every committed D in the wasm-parity CI job. Parameter state — the map
included — rides the URL hash like the staircase's; under
`prefers-reduced-data` no controls are created and the PNG stays.

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

So the honest claim is narrower than "the browser returns the paper's
numbers":

- **Where the orbit locks, the rotation number is exact** — the kernel returns
  the rational `p / q` as soon as the unwrapped orbit closes. Those regions
  are the subject of the figure; the tongues are exactly where the reader is
  looking.
- **The picture is accurate to the displayed colour resolution** (one step in
  256). The tile kernel stops once its running estimate is provably within
  half that tolerance. The colour a reader sees never promises more.
- **The quoted readout does not carry that display tolerance.** Hover and
  keyboard sample one (Omega, K) point with lock detection on and the display
  stop off: the exact rational when the orbit locks, the full `n_iter`
  average otherwise. Until that main-thread module is ready, the readout
  falls back to the tile sample.
- **In the chaotic sea, a single pixel is still not reproducible** across sine
  implementations. It is not reproducible across compilers or CPUs either.
  That is a property of the system, not of this port, and a figure that showed
  otherwise would be hiding it.

`scripts/check_wasm_parity.py` asserts each population separately: locked cells
are held to the existing subcritical bound at any `K`; display-stop cells are
held to `1 / 256`; exhausted cells below the critical line stay on the
subcritical bound; exhausted cells above it keep the chaotic share limit.
Never loosen the subcritical bound to make a red run pass — it still binds
every locked cell. A deliberate `1e-7` perturbation of a tile value trips the
assertions; the script prints a self-check line so that trip is known to work
rather than assumed to.

This is also why a live figure must not be presented as a re-derivation of the
published pixel values in the chaotic region. It shows the same system computed
by the same code; it does not promise the same last digits where the system
itself refuses to have any.

## Checking the live figure in CI

Parity of one tile is not the live path. The live path is: the paper page,
the interact button, a pool of workers, the coarse-to-fine pyramid, a view
change, the rotation numbers at the tongues, and `prefers-reduced-data`.
That check lives at `tests/e2e/live_figure.mjs` and CI runs it.

**Driver: node plus the Chrome DevTools protocol.** No npm, no `package.json`,
no bundler, no playwright, no new dependency of any kind. The reasons, checked
rather than assumed:

- `ubuntu-latest` already carries node and Chrome. The `wasm-parity` job
  already depends on node being on the image and says so in a comment.
- A working driver of exactly this shape has caught real defects: it serves
  the built site with `python3 -m http.server` and drives
  `google-chrome --headless=new` over the DevTools protocol using only node
  built-ins (`node:child_process`, `node:net`, `node:fs`) plus the global
  `WebSocket` and `fetch` that node 22 provides.
- playwright-python would add a dev dependency and a browser download to get
  capabilities this repository does not need.

figure's chart JSON, then runs `node --test tests/js/index.js tests/js/raster.js tests/js/point.js tests/js/live-figure.js tests/js/params.js`
and the end-to-end script. A missing Chrome fails the job; the script prints
the command it tried and exits non-zero. The job does not skip the browser
check.

`scripts/check_wasm_staircase.py` is the staircase's parity sibling, run in
the wasm-parity CI job: the wasm tile at D = 0.25 against the committed
`devils_staircase.npz` rho at exactly the npz A values (tile K ranges chosen
so their linspace reproduces npz points bit for bit, asserted before the
comparison). Below K_c every point stays within the display tolerance; above
it divergence is allowed but bounded by the same share limit.

`scripts/check_wasm_delayed_logistic.py` is the attractor figure's parity
sibling in the same job: the wasm tile at each committed D in
`attractors.npz` and `locking_sequence.npz` against the first 4096 plotted
states, same x0 convention and n_transient 20000. The delayed logistic map
has no transcendental call, so the rule is bit-exact equality — a
difference means the kernel's operation order drifted, not a tolerance to
negotiate.

`scripts/check_wasm_torus_doubling.py` is the torus figure's parity
sibling in the same job: the wasm tile at each committed D in
`map_I_attractors.npz` and `map_IV_attractors.npz` against the first 4096
plotted states, every state component, same x0 and n_transient 20000. The
torus-doubling maps have no transcendental call either, so the rule is
again bit-exact equality.

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

## Figure inventory for live mode

The class names the cheapest browser path that reproduces the figure: A has a
WASM export for the plotted quantity, B needs one bounded map kernel, and C
remains a canonical precomputed figure. `MAX_STEPS` is the per-call budget
defined in `rust/wasm/src/lib.rs:32`.

| section | png | module | class | interaction | kernel | caps |
|---|---|---|---|---|---|---|
| sec02_circle_map | devils_staircase.png | maps.circle_map | A | slider D and zoom A (rotation panel) | rotation_number_tile | A for the rotation panel; n_omega=1 and n_k<=512; Lyapunov panel stays PNG pending circle_map_lyapunov_sum |
| sec02_circle_map | arnold_tongues.png | maps.arnold_tongues | A | zoom (Omega,K) | rotation_number_tile | n_omega,n_k<=512; n_cells*(transient+iter)<=MAX_STEPS |
| sec02_circle_map | staircase_zoom.png | maps.circle_map | A | zoom K window | rotation_number_tile | n_omega,n_k<=512; n_cells*(transient+iter)<=MAX_STEPS |
| sec03_transition | phase_diagram.png | maps.coupled_logistic | B | zoom (A,D) | coupled_logistic_phase_tile | viewport cells<=512; cells*(transient+sample)<=MAX_STEPS |
| sec03_transition | attractors.png | maps.coupled_logistic | B | A slider with iteration-count control | coupled_logistic_attractor_tile | n_A<=64,n_plot<=4096; n_A*(transient+plot)<=MAX_STEPS |
| sec03_transition | basins.png | maps.coupled_logistic | B | zoom initial-state plane | coupled_logistic_basin_grid | n_x,n_y<=512; cells*transient+reference_transient<=MAX_STEPS |
| sec04_doubling | map_I_attractors.png | maps.torus_doubling | B | D slider with iteration-count control | torus_doubling_attractor_tile | map I dim=3; n_plot<=4096; transient+plot<=MAX_STEPS |
| sec04_doubling | map_IV_attractors.png | maps.torus_doubling | B | D slider with iteration-count control | torus_doubling_attractor_tile | map IV dim=4; n_plot<=4096; transient+plot<=MAX_STEPS |
| sec04_doubling | map_IV_lyapunov.png | maps.torus_doubling | B | zoom D in the doubling window | torus_doubling_lyapunov_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec05_oscillation | attractors.png | maps.delayed_logistic | B | D slider with iteration-count control | delayed_logistic_attractor_tile | n_D<=64,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec05_oscillation | lyapunov_vs_D.png | maps.delayed_logistic | B | zoom D and iteration-count control | delayed_logistic_lyapunov_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec05_oscillation | locking_sequence.png | maps.delayed_logistic | B | D slider through the locking window | delayed_logistic_attractor_tile | n_D<=64,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec06_three_torus | lyapunov_vs_DB.png | maps.coupled_delayed | B | DB and epsilon sliders | coupled_delayed_lyapunov_tile | n_DB<=128; n_DB*(transient+iter)<=MAX_STEPS |
| sec06_three_torus | xz_projections.png | maps.coupled_delayed | B | DB slider with projection zoom | coupled_delayed_projection_tile | n_DB<=32,n_plot<=4096; n_DB*(transient+plot)<=MAX_STEPS |
| sec06_three_torus | double_staircase.png | maps.modulated_circle | B | D slider and zoom | modulated_circle_rotation_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec06_three_torus | double_staircase_zoom.png | maps.modulated_circle | B | zoom either locking window | modulated_circle_rotation_tile | n_D<=512; n_D*(transient+iter)<=MAX_STEPS |
| sec07_fractalization | fractal_attractors.png | maps.fractalization | B | D slider with iteration-count control | fractalization_attractor_tile | n_D<=32,n_plot<=4096; n_D*(transient+plot)<=MAX_STEPS |
| sec07_fractalization | correlation_dimension.png | maps.fractalization | C | none | precomputed | C: 200 parameters, 100,000 samples, and max_pairs=1,000,000 are paper-scale |
| sec08_sti | spacetime_diagrams.png | cml.spatiotemporal | B | play/pause and epsilon slider | cml_spacetime_tile | n_sites<=512,n_record<=2048; n_sites*(transient+record)<=MAX_STEPS |
| sec08_sti | comoving_lyapunov.png | cml.comoving_figure | C | none | precomputed | C: three 301-velocity scans at 100,000 iterations on N=500 exceed a tile budget |
| sec08_sti | correlation_decay.png | cml.correlation_figure | C | none | precomputed | C: four regimes plus 20,000-step Lyapunov-density diagnostics are a published cache |
| sec09_pattern | phase_diagram.png | cml.pattern_dynamics | C | none | precomputed | C: 160x200 phase sweep with 5,000 transient and 2,000 sample steps on N=100 |
| sec09_pattern | space_amplitude.png | cml.pattern_dynamics | C | none | precomputed | C: five long phase exemplars are static snapshots, not a bounded exploration |
| sec10_gcm | gcm_msd.png | cml.globally_coupled | C | none | precomputed | C: N reaches 20,000 and each series has 100,000 samples |
| sec10_gcm | gcm_distribution.png | cml.globally_coupled | C | none | precomputed | C: N reaches 20,000 and each series has 100,000 samples |
| sec10_gcm | gcm_clusters.png | cml.gcm_clusters | C | none | precomputed | C: cluster labels are a fixed phase exemplar with 20,000 transient and 500 record steps |
| sec10_gcm | collective_lyapunov.png | cml.gcm_clusters | C | none | precomputed | C: 100 parameter values at N=500 and 50,000 measurement steps are paper-scale |
| sec11_diagnostics | test01_sweep.png | diagnostics.compare_all | C | none | precomputed | C: 500-parameter sweep, 5,000-series samples, and 50 frequencies per point |
| sec11_diagnostics | sali_comparison.png | diagnostics.compare_all | C | none | precomputed | C: four fixed 10,000-step SALI diagnostic traces are published comparisons |
| sec11_diagnostics | permutation_entropy.png | diagnostics.compare_all | C | none | precomputed | C: 500 logistic plus 300 delayed-logistic parameter sweeps over 5,000 samples |
| sec11_diagnostics | complexity_entropy_plane.png | diagnostics.compare_all | C | none | precomputed | C: two 200-parameter sweeps over 5,000 samples feed a diagnostic plane |
| sec11_diagnostics | rqa_measures.png | diagnostics.compare_all | C | none | precomputed | C: 80 parameters each require a 2,000-point recurrence matrix |
| sec12_intermittency | type_i_intermittency.png | diagnostics.intermittency_figure | C | none | precomputed | C: 200,000-point tail, bootstrap fits, normal-form scaling, and a Lorenz return map |
| sec12_intermittency | on_off_intermittency.png | diagnostics.on_off_intermittency_figure | C | none | precomputed | C: benchmark orbit, 100,000-point scaling, and bootstrap fits are proof diagnostics |
| sec12_intermittency | type_ii_intermittency.png | diagnostics.type_ii_intermittency_figure | C | none | precomputed | C: stochastic reinjection proof data and bootstrap fits are published values |
| sec12_intermittency | type_iii_intermittency.png | diagnostics.type_iii_intermittency_figure | C | none | precomputed | C: stochastic reinjection, escape episodes, and bootstrap fits are proof data |
| sec12_intermittency | sti_spine.png | cml.sti_spine_figure | C | none | precomputed | C: 512-site spacetime plus a nine-point coupling sweep and cluster fit are a diagnostic cache |
