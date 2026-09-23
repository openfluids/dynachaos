/**
 * Glue between the live figure's DOM wiring and its kernels.
 *
 * The page template (scripts/paper_shell.py, mountLive) imports this module
 * and keeps only DOM work: elements, the Plot() call, the pool options. The
 * decisions live here so node tests can drive them — the handlers used to be
 * inside the Python template string, which node cannot import.
 *
 * No DOM and no globals at import time: every collaborator (store, pool,
 * raster, point, plot) arrives as an argument. Handlers that must exist
 * before their collaborator does take a getter — onView is built before the
 * pool is created, and onPaint can only fire after the pool exists, so both
 * read the current binding when they run.
 */

import { HEADER } from "./raster.js";
import { clampValue, debounce } from "./params.js";

/**
 * The published figure's iteration parameters. The readout's point.sample
 * call, the tile scheduler options and the raster's tile-size budget all
 * read this one object, so the quoted number and the picture cannot drift
 * apart. Frozen: a mutation would silently change only the readers that
 * already captured a value.
 */
export const LIVE_ARNOLD = Object.freeze({nTransient: 200, nIter: 2000, theta0: 0.1});

/**
 * The devil's staircase figure: rho(A) at a reader-chosen drive frequency D.
 * The paper computes D = 0.25, A in [0, 0.25], n_transient 5000, n_iter
 * 50000, theta0 0.1 (maps/circle_map.py). The tile kernel reads Omega = D
 * and K = A, so one tile column (n_omega = 1) at the current D is the whole
 * curve. D is capped at 0.5: rho stays under D + K <= 0.75, inside the
 * plot's [0, 1] y base.
 */
export const LIVE_STAIRCASE = Object.freeze({
  nTransient: 5000,
  nIter: 50000,
  theta0: 0.1,
  paramSpecs: Object.freeze([
    Object.freeze({ name: "D", label: "drive frequency D", min: 0, max: 0.5, step: 0.005, default: 0.25 }),
  ]),
});

/**
 * The delayed-logistic attractors figure: the (x, y) point cloud of
 * (x, y)' = (A x + (1 - A)(1 - D y^2), x) at a reader-chosen D, A = 0.3
 * fixed. The paper computes A = 0.3, n_transient 20000, n_plot 100000
 * (maps/delayed_logistic.py compute_attractors); the live figure plots at
 * most 4096 states, the wasm kernel's cap. D's range is the kernel's own
 * clamp [1.4, 3.5]; the default 1.90 is one of the published panels. `n`
 * is the plotted-state count: the control that trades cloud density for
 * response time. `domain` is the published figure's shared axis limits —
 * the union of both committed npz files padded by 5% — so the slider
 * compares attractors on the paper's axes instead of a refit frame.
 */
export const LIVE_ATTRACTORS = Object.freeze({
  A: 0.3,
  nTransient: 20000,
  nPlot: 2048,
  domain: Object.freeze({ x0: -0.5642697119632384, x1: 0.9817445276044394, y0: -0.5642697119632384, y1: 0.9817445276044394 }),
  paramSpecs: Object.freeze([
    Object.freeze({ name: "D", label: "map parameter D", min: 1.4, max: 3.5, step: 0.005, default: 1.9 }),
    Object.freeze({ name: "n", label: "plotted states n", min: 256, max: 4096, step: 256, default: 2048 }),
  ]),
});

/**
 * The torus-doubling attractors figure: the (X, Y) projection of map (I) or
 * map (IV) at a reader-chosen D. The paper computes map I at A = 0.4 from
 * x0 = (0.5, 0.5, 0.5) and map IV at A = 0.3 from x0 = (0.5, 0.45, 0.52,
 * 0.48), both with n_transient 20000 (maps/torus_doubling.py
 * compute_map_I/compute_map_IV); the live figure plots at most 4096 states,
 * the wasm kernel's cap. Each map's D window is its published sweep —
 * [1.9, 2.25] for map I, [1.48, 1.53] for map IV — inside the kernel's
 * [1.48, 2.25] clamp; the defaults 2.16 and 1.5206 are the published
 * doubled-torus panels. The D spec's range is the union of both windows so
 * a hash token for either map parses; the figure itself clamps D into the
 * active map's window. `domain` is the union of both committed npz files'
 * (X, Y) extents padded by 5%, so the selector compares the two maps on
 * one fixed frame. `project` is the paper's projection: both figures
 * scatter columns 0 and 1 (X, Y) of the state.
 */
export const LIVE_TORUS = Object.freeze({
  nTransient: 20000,
  domain: Object.freeze({ x0: -0.5148468759369567, x1: 1.0121499654527584, y0: -0.5253812228060238, y1: 0.971379815168809 }),
  maps: Object.freeze({
    1: Object.freeze({
      kind: 1,
      a: 0.4,
      x0: Object.freeze([0.5, 0.5, 0.5]),
      dMin: 1.9,
      dMax: 2.25,
      dStep: 0.005,
      dDefault: 2.16,
      title: "Map (I), \u03b1 = 0.4",
      project: Object.freeze([0, 1]),
    }),
    4: Object.freeze({
      kind: 4,
      a: 0.3,
      x0: Object.freeze([0.5, 0.45, 0.52, 0.48]),
      dMin: 1.48,
      dMax: 1.53,
      dStep: 0.0002,
      dDefault: 1.5206,
      title: "Map (IV), \u03b1 = 0.3",
      project: Object.freeze([0, 1]),
    }),
  }),
  paramSpecs: Object.freeze([
    Object.freeze({
      name: "map",
      label: "map",
      min: 1,
      max: 4,
      step: 3,
      default: 1,
      options: Object.freeze([
        Object.freeze({ value: 1, label: "map (I), \u03b1 = 0.4" }),
        Object.freeze({ value: 4, label: "map (IV), \u03b1 = 0.3" }),
      ]),
    }),
    Object.freeze({ name: "D", label: "map parameter D", min: 1.48, max: 2.25, step: 0.005, default: 2.16 }),
    Object.freeze({ name: "n", label: "plotted states n", min: 256, max: 4096, step: 256, default: 2048 }),
  ]),
});

/**
 * The double devil's staircase figure: rho_theta and rho_phi against D for
 * the modulated circle map (Kaneko Eq. 3.1). The paper computes A = 0.1,
 * C = (sqrt(5) - 1) / 2, eps = 0.05, theta0 = phi0 = 0.1, n_transient 3000,
 * n_iter 20000 over D on np.linspace(0, 1, 10000)
 * (maps/modulated_circle.py compute); the live figure recomputes the same
 * pair over a reader-chosen D window with modulated_circle_rotation_tile.
 * eps is the one free parameter (0 to 0.2, the paper's moderate-forcing
 * neighbourhood); A and C stay at the paper values, so rho_phi is C to
 * numerical precision and the curve is a flat line the reader can check.
 * `dMin`/`dMax` are the compute window and the plot's x domain at once —
 * the number fields, the two presets and a drag zoom all move the same
 * pair. `presets` reproduce the D windows of the zoom panels in
 * modulated_circle.py plot_zoom: longest_plateau_window(D, rho_theta,
 * target, 5e-4) on the committed npz padded by 0.008 (target 1/4) and
 * 0.006 (target C) — computed once from
 * figures/sec06_three_torus/double_staircase.npz and hard-coded here.
 * `n` is the D count, bounded by the kernel's 512 cap.
 */
export const LIVE_MODULATED = Object.freeze({
  A: 0.1,
  C: 0.6180339887498949,
  theta0: 0.1,
  phi0: 0.1,
  nTransient: 3000,
  nIter: 20000,
  presets: Object.freeze([
    Object.freeze({ name: "zoom (a): plateau near 1/4", dMin: 0.25502630263026305, dMax: 0.27372657265726574 }),
    Object.freeze({ name: "zoom (b): plateau near C", dMin: 0.5987604760476047, dMax: 0.6270621062106211 }),
  ]),
  paramSpecs: Object.freeze([
    Object.freeze({ name: "eps", label: "forcing \u03b5", min: 0, max: 0.2, step: 0.005, default: 0.05 }),
    Object.freeze({ name: "dMin", label: "D range min", min: 0, max: 1, step: 0.001, default: 0, numberOnly: true }),
    Object.freeze({ name: "dMax", label: "D range max", min: 0, max: 1, step: 0.001, default: 1, numberOnly: true }),
    Object.freeze({ name: "n", label: "points n", min: 16, max: 512, step: 16, default: 256 }),
  ]),
});

/**
 * data-live kind -> live figure config. The hash-restore path in the page
 * reads paramSpecs from here so a shared link can name its figure.
 */
export const LIVE_FIGURES = Object.freeze({
  arnold_tongues: LIVE_ARNOLD,
  devils_staircase: LIVE_STAIRCASE,
  delayed_logistic_attractors: LIVE_ATTRACTORS,
  torus_doubling_attractors: LIVE_TORUS,
  double_staircase: LIVE_MODULATED,
});

/**
 * The figure's mutable tile store. `tiles` is the same array the Plot()
 * live adapter reads, so a generation reset empties it in place. `trace`
 * is the 1-D figure's derived polyline ({x, y}); the heatmap never uses it.
 *
 * @param {{ nIter: number }} [params]
 * @returns {{ tiles: object[], trace: {x: number[], y: number[]}, generation: number, painted: number, nIter: number }}
 */
export function createStore(params = LIVE_ARNOLD) {
  return { tiles: [], trace: { x: [], y: [] }, generation: 0, painted: 0, nIter: params.nIter };
}

/**
 * The pool's onPaint: fold one finished tile into the store and repaint.
 * A stale-generation command and a tile id outside the viewport are dropped.
 * Every stored record carries a paint sequence so a re-painted tile keys
 * differently in the colour cache — without it the stale bitmap shows.
 * nIter rides in on the tile header so the readout tolerance tracks the
 * kernel's actual iteration count. `onStore`, when given, runs after the
 * fold and before the redraw: the 1-D figure rebuilds its trace there.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ tileWorld: (id: string, viewport: object) => object | null }} deps.raster
 * @param {() => object} deps.getPool
 * @param {() => object} deps.getPlot
 * @param {(store: object) => void} [deps.onStore]
 * @returns {(cmd: object) => void}
 */
export function createPaintHandler({ store, raster, getPool, getPlot, onStore }) {
  let paintSeq = 0;
  return function onPaint(cmd) {
    const state = getPool().getState();
    if (cmd.generation !== state.generation) return;
    const world = raster.tileWorld(cmd.id, state.viewport);
    if (!world) return;
    const rec = {
      ...world,
      generation: cmd.generation,
      header: cmd.header,
      data: cmd.data,
      paintSeq: ++paintSeq,
    };
    const idx = store.tiles.findIndex((t) => t.id === rec.id);
    if (idx >= 0) store.tiles[idx] = rec;
    else store.tiles.push(rec);
    store.generation = cmd.generation;
    store.painted = store.tiles.filter(
      (t) => t.generation === store.generation,
    ).length;
    if (cmd.header && Number.isFinite(cmd.header[3])) store.nIter = cmd.header[3];
    if (onStore) onStore(store);
    getPlot().redraw();
  };
}

/**
 * The hovered/keyboard readout: the point kernel when it is ready and
 * returns a finite number, else the raster lookup on the current
 * generation. The point path runs the full n_iter without the display
 * stop, so it is the value the paper quotes; the raster is the fallback
 * while the wasm glue is still loading or cannot answer.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ isReady: () => boolean, sample: (omega: number, K: number, nTransient: number, nIter: number, theta0: number) => number | null }} deps.point
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @returns {(omega: number, K: number) => number | null}
 */
export function createReadout({ store, point, raster, params = LIVE_ARNOLD }) {
  return function sampleReadout(omega, K) {
    if (point.isReady()) {
      try {
        const rho = point.sample(omega, K, params.nTransient, params.nIter, params.theta0);
        if (typeof rho === "number" && Number.isFinite(rho)) return rho;
      } catch (_) {}
    }
    return raster.sampleAt(store.tiles, store.generation, omega, K);
  };
}

/**
 * The staircase readout: rho at (D, A) where A is the cursor's x value and
 * D is the slider's current value. Same point-kernel-first rule as the
 * heatmap readout; the raster fallback samples (D, A) on the stored tiles.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ isReady: () => boolean, sample: (omega: number, K: number, nTransient: number, nIter: number, theta0: number) => number | null }} deps.point
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @param {() => number} deps.getD
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @returns {(a: number) => number | null}
 */
export function createStaircaseReadout({ store, point, raster, getD, params = LIVE_STAIRCASE }) {
  return function sampleStaircase(a) {
    if (point.isReady()) {
      try {
        const rho = point.sample(getD(), a, params.nTransient, params.nIter, params.theta0);
        if (typeof rho === "number" && Number.isFinite(rho)) return rho;
      } catch (_) {}
    }
    return raster.sampleAt(store.tiles, store.generation, getD(), a);
  };
}

/**
 * The raster half of the readout, exposed to the page as
 * fig._live.rasterSample so the e2e test can compare the two paths at the
 * same (Omega, K) — not at a neighbouring tile cell.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ sampleAt: (tiles: object[], generation: number, omega: number, K: number) => number | null }} deps.raster
 * @returns {(omega: number, K: number) => number | null}
 */
export function createRasterSample({ store, raster }) {
  return (omega, K) => raster.sampleAt(store.tiles, store.generation, omega, K);
}

/**
 * The scheduler options for the pool: the pyramid shape (levels, tileCells)
 * plus the iteration parameters every tile command carries. The same
 * `params` object feeds the readout, so the picture and the quoted number
 * are computed with identical settings. `lockOmega` switches the scheduler
 * to its 1-D mode (nOmega = 1 at the viewport's fixed Omega).
 *
 * @param {object} deps
 * @param {number} deps.levels
 * @param {number} deps.tileCells
 * @param {{ nTransient: number, nIter: number, theta0: number }} [deps.params]
 * @param {boolean} [deps.lockOmega]
 * @returns {{ levels: number, tileCells: number, nTransient: number, nIter: number, theta0: number, lockOmega?: boolean }}
 */
export function createSchedulerOptions({ levels, tileCells, params = LIVE_ARNOLD, lockOmega = false }) {
  const out = {
    levels,
    tileCells,
    nTransient: params.nTransient,
    nIter: params.nIter,
    theta0: params.theta0,
  };
  if (lockOmega) out.lockOmega = true;
  return out;
}

/**
 * The Plot() live adapter's view hook: push the new domain into the pool
 * and, when the pool answers with a new generation, reset the store. A new
 * generation also resets the scheduler's per-generation retry list, so a
 * tile that spent its one retry before the pan may be retried again after
 * it. Emptying `tiles` in place keeps the array identity the adapter
 * captured.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPool
 * @param {() => object | null} deps.getPlot
 * @returns {(d: { x0: number, x1: number, y0: number, y1: number }) => void}
 */
export function createViewHandler({ store, getPool, getPlot }) {
  return function onView(d) {
    const pool = getPool();
    if (!pool) return;
    pool.setViewport({ omegaMin: d.x0, omegaMax: d.x1, kMin: d.y0, kMax: d.y1 });
    const gen = pool.getState().generation;
    if (gen !== store.generation) {
      store.generation = gen;
      store.painted = 0;
      store.tiles.length = 0;
      const plot = getPlot();
      if (plot) plot.redraw();
    }
  };
}

/**
 * The staircase's view hook and parameter push, sharing one push helper.
 * The plot's x axis is A (the kernel's K); Omega is the slider's D. A
 * y-only domain change (the auto-fit, a y drag) must not recompute, so
 * onView pushes only when the x domain actually moved. applyD pushes the
 * current D at the current x domain — the slider's recompute path. Both
 * funnel through the same generation reset as the heatmap's onView.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPool
 * @param {() => object | null} deps.getPlot
 * @param {() => number} deps.getD
 * @returns {{ onView: (d: { x0: number, x1: number }) => void, applyD: () => void }}
 */
export function createStaircaseView({ store, getPool, getPlot, getD }) {
  let lastK = null;
  function push(kMin, kMax) {
    const pool = getPool();
    if (!pool) return;
    const D = getD();
    pool.setViewport({ omegaMin: D, omegaMax: D, kMin, kMax });
    lastK = { kMin, kMax };
    const gen = pool.getState().generation;
    if (gen !== store.generation) {
      store.generation = gen;
      store.painted = 0;
      store.tiles.length = 0;
      const plot = getPlot();
      if (plot) plot.redraw();
    }
  }
  return {
    onView(d) {
      if (lastK && d.x0 === lastK.kMin && d.x1 === lastK.kMax) return;
      push(d.x0, d.x1);
    },
    applyD() {
      const plot = getPlot();
      if (!plot) return;
      const dom = plot.getDomain();
      push(dom.x0, dom.x1);
    },
  };
}

/**
 * Fold the stored tiles into the staircase's polyline. Every current-
 * generation tile contributes its cells at their K coordinates; where two
 * tiles share a coordinate the finer level wins. The result is sorted by A
 * so the Plot() line path can stroke it directly. A 1-D tile carries
 * nOmega = 1, so cell iy sits at kMin + iy * (kMax - kMin) / (nK - 1) —
 * the same linspace the kernel iterates.
 *
 * @param {object[]} tiles
 * @param {number} generation
 * @returns {{ x: number[], y: number[] }}
 */
export function tilesToTrace(tiles, generation) {
  const byA = new Map();
  for (const t of tiles) {
    if (!t || t.generation !== generation || !t.header || !t.data) continue;
    const nK = Math.max(1, Math.floor(Number(t.header[1])));
    const nOmega = Math.max(1, Math.floor(Number(t.header[0])));
    if (t.data.length < HEADER + nK * nOmega) continue;
    const step = nK > 1 ? (t.kMax - t.kMin) / (nK - 1) : 0;
    for (let iy = 0; iy < nK; iy++) {
      const a = nK > 1 ? t.kMin + iy * step : t.kMin;
      const v = t.data[HEADER + iy * nOmega];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      const prev = byA.get(a);
      if (!prev || t.level >= prev.level) byA.set(a, { level: t.level, v });
    }
  }
  const xs = [...byA.keys()].sort((p, q) => p - q);
  return { x: xs, y: xs.map((a) => byA.get(a).v) };
}

/**
 * Refit the plot's y domain to the trace once per generation. The base
 * domain is [0, 1] so a reset always shows the full rotation-number range;
 * the first paint of a generation tightens it to what the curve actually
 * spans. Later paints of the same generation leave the user's view alone.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object | null} deps.getPlot
 * @returns {() => void}
 */
export function createLineYFit({ store, getPlot }) {
  let fittedGen = -1;
  return function fitLineY() {
    if (store.generation === fittedGen) return;
    const xs = store.trace.y;
    if (!xs.length) return;
    fittedGen = store.generation;
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    const pad = (hi - lo) * 0.08 || 0.05;
    const plot = getPlot();
    if (!plot) return;
    const dom = plot.getDomain();
    plot.setDomain({ x0: dom.x0, x1: dom.x1, y0: lo - pad, y1: hi + pad });
  };
}

/**
 * Slider wiring for one live figure's parameter set. `onInput(name, raw)`
 * clamps the value into its spec, updates the state, echoes the clamped
 * value back through `echo` (the page syncs its two inputs there), and
 * schedules `apply` through the debounce — a drag burst collapses into one
 * recompute. `setParams` applies a parsed hash state immediately, with no
 * debounce: a shared link must not wait. `writeHash` runs inside apply so
 * the URL and the recompute cannot drift apart. `onSet`, when given, runs
 * after each spec's state update — in `setParams` between specs, so a
 * selector that changes another parameter's range (the torus figure's map
 * switch moves D's window) sees the new value before the next spec applies
 * — and the echo reads the post-hook state, so a snapped or derived value
 * never disagrees with its control.
 *
 * @param {object} deps
 * @param {{ name: string, min: number, max: number, default: number }[]} deps.specs
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {() => void} deps.apply
 * @param {() => void} deps.writeHash
 * @param {(name: string, state: object) => void} [deps.onSet]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void }}
 */
export function createParamWiring({ specs, debounceMs, echo, apply, writeHash, onSet, timers }) {
  const state = {};
  for (const spec of specs) state[spec.name] = spec.default;
  const fire = debounce(() => {
    apply();
    writeHash();
  }, debounceMs, timers);
  function onInput(name, raw) {
    const spec = specs.find((s) => s.name === name);
    if (!spec) return;
    state[name] = clampValue(raw, spec);
    if (onSet) onSet(name, state);
    echo(name, state[name]);
    fire();
  }
  function setParams(values) {
    for (const spec of specs) {
      if (values[spec.name] === undefined) continue;
      state[spec.name] = clampValue(values[spec.name], spec);
      if (onSet) onSet(spec.name, state);
      echo(spec.name, state[spec.name]);
    }
    apply();
    writeHash();
  }
  return { state, onInput, setParams };
}

const ATTRACTOR_GLUE_HREF = new URL("../wasm/dynachaos_wasm.js", import.meta.url).href;

/**
 * The analytic fixed point of the delayed logistic map at parameter D:
 * solving x = A x + (1 - A)(1 - D x^2) gives x = (sqrt(1 + 4D) - 1) / 2D.
 * The published figure starts its orbits at (fp + 0.01, fp - 0.01), so the
 * live figure does too, and the canvas marks the point itself.
 *
 * @param {number} D
 * @returns {number}
 */
export function attractorFixedPoint(D) {
  return (Math.sqrt(1 + 4 * D) - 1) / (2 * D);
}

/**
 * The kernel request for one attractor state: A first, then the D range —
 * the order delayed_logistic_attractor_tile takes them. n_D is 1, so the
 * tile holds one block of n_plot (x, y) pairs. state0 is the published
 * convention (fp + 0.01, fp - 0.01); the same expression in the same order
 * as maps/delayed_logistic.py, so the orbit is bit-identical.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ A: number, nTransient: number, nPlot: number }} [params]
 * @returns {{ a: number, dMin: number, dMax: number, nD: number, nTransient: number, nPlot: number, state0: Float64Array }}
 */
export function attractorRequest(state, params = LIVE_ATTRACTORS) {
  const D = state.D;
  const fp = attractorFixedPoint(D);
  return {
    a: params.A,
    dMin: D,
    dMax: D,
    nD: 1,
    nTransient: params.nTransient,
    nPlot: Math.round(state.n),
    state0: new Float64Array([fp + 0.01, fp - 0.01]),
  };
}

/**
 * Fold an attractor tile into the point cloud. The layout is the kernel's:
 * a 4-element header [n_D, n_plot, n_transient, 2], then n_D blocks of
 * n_plot (x, y) pairs — x first, then y. A diverged D is a block of NaN
 * pairs; the plot skips non-finite points, so they fold in harmlessly.
 * Anything that does not match the header yields an empty cloud rather
 * than a misread buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @returns {{ x: number[], y: number[] }}
 */
export function attractorTrace(tile) {
  const x = [];
  const y = [];
  if (!tile || tile.length < HEADER) return { x, y };
  const nD = Math.floor(Number(tile[0]));
  const nPlot = Math.floor(Number(tile[1]));
  if (!(nD >= 1) || !(nPlot >= 1) || tile.length < HEADER + nD * nPlot * 2) {
    return { x, y };
  }
  for (let i = 0; i < nD * nPlot; i++) {
    x.push(tile[HEADER + 2 * i]);
    y.push(tile[HEADER + 2 * i + 1]);
  }
  return { x, y };
}

const kernelPromises = new Map();
const kernelCache = new Map();


/**
 * Lazily load the wasm glue and return one named export. The module is the
 * same instance point.js initialises (the import cache keys on the resolved
 * URL), and the glue's init is idempotent. A failed load is not cached
 * forever: the next call retries once.
 *
 * @param {string} name the wasm-bindgen export to return
 * @returns {Promise<Function>}
 */
function loadWasmExport(name) {
  let promise = kernelPromises.get(name);
  if (promise == null) {
    promise = (async () => {
      const glue = await import(ATTRACTOR_GLUE_HREF);
      if (typeof glue[name] !== "function") {
        throw new Error(`wasm glue is missing ${name}`);
      }
      const isNode =
        typeof process !== "undefined" && process.versions && process.versions.node;
      if (!isNode) {
        await glue.default();
      } else {
        const { readFile } = await import("node:fs/promises");
        const bytes = await readFile(new URL("dynachaos_wasm_bg.wasm", ATTRACTOR_GLUE_HREF));
        glue.initSync({ module: bytes });
      }
      kernelCache.set(name, glue[name]);
      return glue[name];
    })();
    promise.catch(() => {
      kernelPromises.delete(name);
    });
    kernelPromises.set(name, promise);
  }
  return promise;
}

/**
 * The already-resolved export, or null while the glue is still loading.
 * The double staircase's readout samples through this synchronously — a
 * per-pointermove await would lag the cursor.
 *
 * @param {string} name the wasm-bindgen export
 * @returns {Function | null}
 */
export function loadedKernel(name) {
  return kernelCache.get(name) || null;
}

/**
 * delayed_logistic_attractor_tile, loaded through the shared loader.
 *
 * @returns {Promise<(a: number, dMin: number, dMax: number, nD: number, nTransient: number, nPlot: number, state0: Float64Array) => Float64Array>}
 */
export function loadAttractorKernel() {
  return loadWasmExport("delayed_logistic_attractor_tile");
}

/**
 * torus_doubling_attractor_tile, loaded through the shared loader.
 *
 * @returns {Promise<(mapKind: number, a: number, d: number, nTransient: number, nPlot: number, state0: Float64Array) => Float64Array>}
 */
export function loadTorusKernel() {
  return loadWasmExport("torus_doubling_attractor_tile");
}

/**
 * modulated_circle_rotation_tile, loaded through the shared loader.
 *
 * @returns {Promise<(a: number, c: number, dMin: number, dMax: number, nD: number, eps: number, nTransient: number, nIter: number, theta0: number, phi0: number) => Float64Array>}
 */
export function loadModulatedKernel() {
  return loadWasmExport("modulated_circle_rotation_tile");
}


/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — A, then the D range, the
 * counts, and the start state.
 *
 * @param {ReturnType<typeof attractorRequest>} req
 * @returns {Promise<Float64Array>}
 */
export async function attractorKernelCall(req) {
  const kernel = await loadAttractorKernel();
  return kernel(req.a, req.dMin, req.dMax, req.nD, req.nTransient, req.nPlot, req.state0);
}

/**
 * The recompute loop behind the direct-call attractor figures. `request`
 * turns the parameter state into a kernel request, `call` runs it, `fold`
 * turns the returned tile into the (x, y) cloud, and a sequence number
 * drops a stale result: a request that resolves after a newer one was
 * issued is discarded, never painted. `onTrace` runs after the store's
 * arrays are spliced so the page can update its derived marks and redraw.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => object} deps.getState
 * @param {(state: object) => object} deps.request
 * @param {(tile: ArrayLike<number>, req: object) => {x: number[], y: number[]}} deps.fold
 * @param {(req: object) => Promise<ArrayLike<number>>} deps.call
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} [deps.onTrace]
 * @param {(err: unknown) => void} [deps.onError]
 * @returns {{ run: () => void, ready: () => boolean }}
 */
function createKernelRunner({ store, getState, request, fold, call, onTrace, onError }) {
  let seq = 0;
  let ready = false;
  async function run() {
    const req = request(getState());
    const mySeq = ++seq;
    try {
      const tile = await call(req);
      ready = true;
      if (mySeq !== seq) return; // a newer request superseded this one
      const trace = fold(tile, req);
      // Splice into the store's arrays instead of replacing them: the
      // page's panel.traces captured these arrays at mount, and a fresh
      // object would orphan the plot's view of the cloud. Every channel
      // the fold returns is spliced — the double staircase carries a
      // second curve in trace.y2 beside x and y.
      for (const key of Object.keys(trace)) {
        const target = store.trace[key];
        const source = trace[key];
        if (Array.isArray(target)) target.splice(0, target.length, ...source);
        else store.trace[key] = source;
      }
      store.generation = mySeq;
      store.painted = trace.x.length;
      if (onTrace) onTrace(store, trace, req);
    } catch (err) {
      if (onError) onError(err);
    }
  }
  return { run, ready: () => ready };
}

/**
 * The attractor figure's parameter wiring and recompute path. The page
 * (mountAttractors in scripts/paper_shell.py) builds the inputs and hands
 * in the DOM callbacks; the decisions live here so node tests can drive
 * them. `call` is injectable: the page passes attractorKernelCall, a test
 * passes a spy.
 *
 * The kernel is called directly on the main thread — one call is at most
 * 20000 + 4096 map steps, microseconds of work, so the tile pool's worker
 * round trip would buy nothing. The debounce collapses a slider burst into
 * one apply, and a sequence number drops a stale result: a request that
 * resolves after a newer one was issued is discarded, never painted.
 *
 * @param {object} deps
 * @param {{ A: number, nTransient: number, nPlot: number, paramSpecs: object[] }} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createAttractorFigure({
  params = LIVE_ATTRACTORS,
  debounceMs,
  echo,
  call = attractorKernelCall,
  onTrace,
  writeHash,
  onError,
  timers,
}) {
  const store = { trace: { x: [], y: [] }, generation: 0, painted: 0 };
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => attractorRequest(state, params),
    fold: (tile) => attractorTrace(tile),
    call,
    onTrace,
    onError,
  });
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    timers,
  });
  return {
    state: wiring.state,
    store,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
  };
}

/**
 * Snap a map selector value to a kernel map kind: 2 and below is map I, 3
 * and above is map IV — the same snap the wasm export applies, so the
 * figure and the kernel can never disagree about which map runs.
 *
 * @param {number} value
 * @returns {1 | 4}
 */
export function torusMapKind(value) {
  return Number(value) >= 3 ? 4 : 1;
}

/**
 * The kernel request for one torus-doubling state: map kind first, then A,
 * D, the counts, and the start state — the order
 * torus_doubling_attractor_tile takes them. A and x0 come from the chosen
 * map's config; D is clamped into that map's doubling window so a stale or
 * hand-edited hash value can never reach the kernel outside it.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ nTransient: number, maps: object }} [params]
 * @returns {{ mapKind: number, a: number, d: number, nTransient: number, nPlot: number, state0: Float64Array }}
 */
export function torusRequest(state, params = LIVE_TORUS) {
  const map = params.maps[torusMapKind(state.map)];
  const d = Math.min(map.dMax, Math.max(map.dMin, state.D));
  return {
    mapKind: map.kind,
    a: map.a,
    d,
    nTransient: params.nTransient,
    nPlot: Math.round(state.n),
    state0: new Float64Array(map.x0),
  };
}

/**
 * Fold a torus-doubling tile into the (X, Y) point cloud. The layout is
 * the kernel's: a 4-element header [map_kind, dim, n_transient,
 * n_produced], then n_produced states of dim components each. `project`
 * names the two state components the paper scatters — [0, 1] (X, Y) for
 * both maps. A tile that does not match its header, or a diverged orbit
 * with zero produced states, yields an empty cloud rather than a misread
 * buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @param {{ project: ArrayLike<number> }} map the chosen map's config
 * @returns {{ x: number[], y: number[] }}
 */
export function torusTrace(tile, map) {
  const x = [];
  const y = [];
  if (!tile || tile.length < HEADER) return { x, y };
  const dim = Math.floor(Number(tile[1]));
  const nProduced = Math.floor(Number(tile[3]));
  const px = Math.floor(Number(map.project[0]));
  const py = Math.floor(Number(map.project[1]));
  if (!(dim >= 1) || !(nProduced >= 0) || px >= dim || py >= dim) return { x, y };
  if (tile.length < HEADER + nProduced * dim) return { x, y };
  for (let i = 0; i < nProduced; i++) {
    x.push(tile[HEADER + i * dim + px]);
    y.push(tile[HEADER + i * dim + py]);
  }
  return { x, y };
}

/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — map kind, A, D, the counts,
 * and the start state.
 *
 * @param {ReturnType<typeof torusRequest>} req
 * @returns {Promise<Float64Array>}
 */
export async function torusKernelCall(req) {
  const kernel = await loadTorusKernel();
  return kernel(req.mapKind, req.a, req.d, req.nTransient, req.nPlot, req.state0);
}

/**
 * The torus-doubling figure's parameter wiring and recompute path, sharing
 * the delayed-logistic figure's runner. The page (mountTorus in
 * scripts/paper_shell.py) builds the inputs and hands in the DOM
 * callbacks; the decisions live here so node tests can drive them.
 *
 * The map selector is the figure's one non-slider control. Switching maps
 * snaps the value to a kernel kind, resets D to the new map's published
 * default — the window's own D range would clamp a map-I D to map IV's
 * ceiling, which reads as a broken slider — and lets the wiring's echo
 * carry the snapped value back to the control. A D input outside the
 * active window (the number field accepts the union range) clamps into it.
 *
 * @param {object} deps
 * @param {{ nTransient: number, maps: object, paramSpecs: object[] }} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: {x: number[], y: number[]}, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createTorusFigure({
  params = LIVE_TORUS,
  debounceMs,
  echo,
  call = torusKernelCall,
  onTrace,
  writeHash,
  onError,
  timers,
}) {
  const store = { trace: { x: [], y: [] }, generation: 0, painted: 0 };
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => torusRequest(state, params),
    fold: (tile, req) => torusTrace(tile, params.maps[req.mapKind]),
    call,
    onTrace,
    onError,
  });
  let activeKind = null;
  function syncState(name) {
    const kind = torusMapKind(wiring.state.map);
    if (wiring.state.map !== kind) {
      wiring.state.map = kind;
      echo("map", kind);
    }
    const map = params.maps[kind];
    if (kind !== activeKind) {
      if (name === "map") {
        // A map switch moves D's window; reset to the new map's published
        // default rather than clamping the old map's D into it.
        wiring.state.D = map.dDefault;
        echo("D", map.dDefault);
      }
      activeKind = kind;
    } else {
      const d = Math.min(map.dMax, Math.max(map.dMin, wiring.state.D));
      if (wiring.state.D !== d) {
        wiring.state.D = d;
        echo("D", d);
      }
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    onSet: (name) => syncState(name),
    timers,
  });
  return {
    state: wiring.state,
    store,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
  };
}

/**
 * The kernel request for one double-staircase sweep: A and C first, then
 * the D window, the point count, eps, the iteration counts and the start
 * phases — the order modulated_circle_rotation_tile takes them. A, C and
 * the phases come from the figure's config; eps, the window and n are the
 * reader's. dMin <= dMax is an invariant of the wiring's syncRange, so the
 * request never sees an inverted window.
 *
 * @param {object} state name -> value, from the parameter wiring
 * @param {{ A: number, C: number, theta0: number, phi0: number, nTransient: number, nIter: number }} [params]
 * @returns {{ a: number, c: number, dMin: number, dMax: number, nD: number, eps: number, nTransient: number, nIter: number, theta0: number, phi0: number }}
 */
export function modulatedRequest(state, params = LIVE_MODULATED) {
  return {
    a: params.A,
    c: params.C,
    dMin: state.dMin,
    dMax: state.dMax,
    nD: Math.round(state.n),
    eps: state.eps,
    nTransient: params.nTransient,
    nIter: params.nIter,
    theta0: params.theta0,
    phi0: params.phi0,
  };
}

/**
 * Fold a modulated-circle tile into the two rotation-number curves. The
 * layout is the kernel's: a 4-element header [n_d, n_transient, n_iter,
 * 2], then n_d (rho_theta, rho_phi) pairs, D-major. The D axis is the
 * request's window on the kernel's own grid — dMin + (dMax - dMin) k /
 * (n_d - 1) — so a plotted point sits at exactly the D the kernel
 * iterated. A tile that does not match its header yields empty curves
 * rather than a misread buffer.
 *
 * @param {ArrayLike<number> | null | undefined} tile
 * @param {ReturnType<typeof modulatedRequest>} req
 * @returns {{ x: number[], y: number[], y2: number[] }}
 */
export function modulatedTrace(tile, req) {
  const x = [];
  const y = [];
  const y2 = [];
  if (!tile || tile.length < HEADER) return { x, y, y2 };
  const nD = Math.floor(Number(tile[0]));
  if (!(nD >= 1) || tile.length < HEADER + nD * 2) return { x, y, y2 };
  const step = nD > 1 ? (req.dMax - req.dMin) / (nD - 1) : 0;
  for (let k = 0; k < nD; k++) {
    const rt = tile[HEADER + 2 * k];
    const rp = tile[HEADER + 2 * k + 1];
    if (!Number.isFinite(rt) || !Number.isFinite(rp)) continue;
    x.push(req.dMin + k * step);
    y.push(rt);
    y2.push(rp);
  }
  return { x, y, y2 };
}

/**
 * The default kernel call: load the glue, then invoke the export with the
 * request's fields in the kernel's own order — A, C, the D window, n_d,
 * eps, the counts, and the start phases. `kernel` is injectable so a node
 * test can check the argument order without the wasm build.
 *
 * @param {ReturnType<typeof modulatedRequest>} req
 * @param {Function} [kernel]
 * @returns {Promise<Float64Array>}
 */
export async function modulatedKernelCall(req, kernel) {
  const fn = kernel || (await loadModulatedKernel());
  return fn(req.a, req.c, req.dMin, req.dMax, req.nD, req.eps, req.nTransient, req.nIter, req.theta0, req.phi0);
}

/**
 * The double-staircase readout: rho_theta at the cursor's D, computed by
 * the loaded kernel at the figure's current eps — the same value the e2e
 * check recomputes. Until the glue resolves, the nearest plotted point of
 * the rho_theta trace answers instead, so the tip never shows a gap while
 * the module loads.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {() => number} deps.getEps
 * @param {{ nTransient: number, nIter: number, A: number, C: number, theta0: number, phi0: number }} [deps.params]
 * @returns {(d: number) => number | null}
 */
export function createModulatedReadout({ store, getEps, params = LIVE_MODULATED }) {
  return function sampleStaircase(d) {
    const kernel = loadedKernel("modulated_circle_rotation_tile");
    if (kernel) {
      try {
        const tile = kernel(
          params.A, params.C, d, d, 1, getEps(),
          params.nTransient, params.nIter, params.theta0, params.phi0,
        );
        if (tile && tile.length >= HEADER + 2 && Number.isFinite(tile[HEADER])) {
          return tile[HEADER];
        }
      } catch (_) {}
    }
    const xs = store.trace.x;
    const ys = store.trace.y;
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const dist = Math.abs(xs[i] - d);
      if (dist < bestDist) {
        bestDist = dist;
        best = ys[i];
      }
    }
    return best;
  };
}

/**
 * The double staircase's view hook and range sync. The plot's x domain IS
 * the compute window: a drag zoom or a reset pushes the new [x0, x1] into
 * the dMin/dMax inputs through `onRange` (the debounced input path), and
 * `syncView` — run once per painted trace — moves the domain onto the
 * current range when the change came from the inputs, a preset or a hash
 * restore instead. The two directions cannot loop: onView only fires when
 * the domain's x range differs from the state, and syncView only calls
 * setDomain when it does.
 *
 * @param {object} deps
 * @param {() => { dMin: number, dMax: number }} deps.getRange
 * @param {(dMin: number, dMax: number) => void} deps.onRange
 * @param {() => object | null} deps.getPlot
 * @returns {{ onView: (d: { x0: number, x1: number }) => void, syncView: () => void }}
 */
export function createRangeView({ getRange, onRange, getPlot }) {
  return {
    onView(d) {
      const r = getRange();
      if (d.x0 === r.dMin && d.x1 === r.dMax) return;
      onRange(d.x0, d.x1);
    },
    syncView() {
      const plot = getPlot();
      if (!plot) return;
      const r = getRange();
      const dom = plot.getDomain();
      if (dom.x0 === r.dMin && dom.x1 === r.dMax) return;
      plot.setDomain({ x0: r.dMin, x1: r.dMax, y0: dom.y0, y1: dom.y1 });
    },
  };
}

/**
 * The double-staircase figure's parameter wiring and recompute path,
 * sharing the attractor figures' runner. The page (mountDoubleStaircase
 * in scripts/paper_shell.py) builds the inputs and hands in the DOM
 * callbacks; the decisions live here so node tests can drive them.
 *
 * The dMin/dMax pair is one logical window: editing one end past the
 * other moves the far end to meet it (the edited value always wins), so
 * the kernel never sees an inverted window and a two-step edit lands
 * where the reader meant it. The view sync runs inside the runner's
 * onTrace — after the new curves are spliced, before the page redraws —
 * so the domain and the data move together.
 *
 * @param {object} deps
 * @param {object} [deps.params]
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {(req: object) => Promise<ArrayLike<number>>} [deps.call]
 * @param {(store: object, trace: object, req: object) => void} deps.onTrace
 * @param {() => void} deps.writeHash
 * @param {() => object | null} deps.getPlot
 * @param {(err: unknown) => void} [deps.onError]
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, store: object, view: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void, refresh: () => void, ready: () => boolean }}
 */
export function createModulatedFigure({
  params = LIVE_MODULATED,
  debounceMs,
  echo,
  call = modulatedKernelCall,
  onTrace,
  writeHash,
  getPlot,
  onError,
  timers,
}) {
  const store = { trace: { x: [], y: [], y2: [] }, generation: 0, painted: 0 };
  const fitY = createLineYFit({ store, getPlot });
  const view = createRangeView({
    getRange: () => ({ dMin: wiring.state.dMin, dMax: wiring.state.dMax }),
    onRange: (dMin, dMax) => {
      wiring.onInput("dMin", dMin);
      wiring.onInput("dMax", dMax);
    },
    getPlot,
  });
  const runner = createKernelRunner({
    store,
    getState: () => wiring.state,
    request: (state) => modulatedRequest(state, params),
    fold: (tile, req) => modulatedTrace(tile, req),
    call,
    onTrace: (s, trace, req) => {
      view.syncView();
      fitY();
      if (onTrace) onTrace(s, trace, req);
    },
    onError,
  });
  function syncRange(name) {
    // The edited end wins: pushing dMin past dMax moves dMax up to meet
    // it, and vice versa, so the window never inverts.
    if (name === "dMin" && wiring.state.dMin > wiring.state.dMax) {
      wiring.state.dMax = wiring.state.dMin;
      echo("dMax", wiring.state.dMax);
    } else if (name === "dMax" && wiring.state.dMax < wiring.state.dMin) {
      wiring.state.dMin = wiring.state.dMax;
      echo("dMin", wiring.state.dMin);
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      runner.run();
    },
    writeHash,
    onSet: (name) => syncRange(name),
    timers,
  });
  return {
    state: wiring.state,
    store,
    view,
    onInput: wiring.onInput,
    setParams: wiring.setParams,
    refresh: () => {
      runner.run();
    },
    ready: runner.ready,
  };
}
