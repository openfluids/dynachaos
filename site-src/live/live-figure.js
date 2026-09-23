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
 * data-live kind -> live figure config. The hash-restore path in the page
 * reads paramSpecs from here so a shared link can name its figure.
 */
export const LIVE_FIGURES = Object.freeze({
  arnold_tongues: LIVE_ARNOLD,
  devils_staircase: LIVE_STAIRCASE,
  delayed_logistic_attractors: LIVE_ATTRACTORS,
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
 * the URL and the recompute cannot drift apart.
 *
 * @param {object} deps
 * @param {{ name: string, min: number, max: number, default: number }[]} deps.specs
 * @param {number} deps.debounceMs
 * @param {(name: string, value: number) => void} deps.echo
 * @param {() => void} deps.apply
 * @param {() => void} deps.writeHash
 * @param {{ set?: (fn: () => void, ms: number) => unknown, clear?: (id: unknown) => void }} [deps.timers]
 * @returns {{ state: object, onInput: (name: string, raw: number) => void, setParams: (values: object) => void }}
 */
export function createParamWiring({ specs, debounceMs, echo, apply, writeHash, timers }) {
  const state = {};
  for (const spec of specs) state[spec.name] = spec.default;
  const fire = debounce(() => {
    apply();
    writeHash();
  }, debounceMs, timers);
  function onInput(name, raw) {
    const spec = specs.find((s) => s.name === name);
    if (!spec) return;
    const v = clampValue(raw, spec);
    state[name] = v;
    echo(name, v);
    fire();
  }
  function setParams(values) {
    for (const spec of specs) {
      if (values[spec.name] === undefined) continue;
      const v = clampValue(values[spec.name], spec);
      state[spec.name] = v;
      echo(spec.name, v);
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

let attractorKernelPromise = null;

/**
 * Lazily load the wasm glue and return delayed_logistic_attractor_tile.
 * The module is the same instance point.js initialises (the import cache
 * keys on the resolved URL), and the glue's init is idempotent. A failed
 * load is not cached forever: the next call retries once.
 *
 * @returns {Promise<(a: number, dMin: number, dMax: number, nD: number, nTransient: number, nPlot: number, state0: Float64Array) => Float64Array>}
 */
export function loadAttractorKernel() {
  if (attractorKernelPromise == null) {
    attractorKernelPromise = (async () => {
      const glue = await import(ATTRACTOR_GLUE_HREF);
      if (typeof glue.delayed_logistic_attractor_tile !== "function") {
        throw new Error("wasm glue is missing delayed_logistic_attractor_tile");
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
      return glue.delayed_logistic_attractor_tile;
    })();
    attractorKernelPromise.catch(() => {
      attractorKernelPromise = null;
    });
  }
  return attractorKernelPromise;
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
  let seq = 0;
  let ready = false;
  async function run() {
    const req = attractorRequest(wiring.state, params);
    const mySeq = ++seq;
    try {
      const tile = await call(req);
      ready = true;
      if (mySeq !== seq) return; // a newer request superseded this one
      const trace = attractorTrace(tile);
      // Splice into the store's arrays instead of replacing them: the
      // page's panel.traces captured these two arrays at mount, and a
      // fresh object would orphan the plot's view of the cloud.
      store.trace.x.splice(0, store.trace.x.length, ...trace.x);
      store.trace.y.splice(0, store.trace.y.length, ...trace.y);
      store.generation = mySeq;
      store.painted = trace.x.length;
    } catch (err) {
      if (onError) onError(err);
    }
  }
  const wiring = createParamWiring({
    specs: params.paramSpecs,
    debounceMs,
    echo,
    apply: () => {
      run();
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
      run();
    },
    ready: () => ready,
  };
}
