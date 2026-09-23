import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LIVE_ARNOLD,
  LIVE_ATTRACTORS,
  LIVE_FIGURES,
  LIVE_STAIRCASE,
  LIVE_TORUS,
  attractorFixedPoint,
  attractorRequest,
  attractorTrace,
  createAttractorFigure,
  createPaintHandler,
  createRasterSample,
  createReadout,
  createSchedulerOptions,
  createStaircaseReadout,
  createStaircaseView,
  createStore,
  createTorusFigure,
  createViewHandler,
  tilesToTrace,
  torusMapKind,
  torusRequest,
  torusTrace,
} from "../../site-src/live/live-figure.js";
import {
  HEADER,
  liveTileColorKey,
  sampleAt,
  tileWorld,
} from "../../site-src/live/raster.js";
import { visibleTiles } from "../../site-src/live/scheduler.js";
import { createPool } from "../../site-src/live/pool.js";

const VIEWPORT = { omegaMin: 0, omegaMax: 1, kMin: 0, kMax: 0.3 };

function makeFakeWorkerClass() {
  const instances = [];
  class FakeWorker {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.terminated = false;
      this.onmessage = null;
      this.onerror = null;
      this.posted = [];
      instances.push(this);
    }
    postMessage(msg) {
      this.posted.push(msg);
    }
    terminate() {
      this.terminated = true;
    }
  }
  FakeWorker.instances = instances;
  return FakeWorker;
}

function failWorker(worker) {
  if (typeof worker.onerror === "function") worker.onerror();
}

// A 2x2 tile over the whole viewport whose four cells hold distinct values,
// so a readout that samples a neighbouring cell returns a different number.
function makeTile(values = [0.1, 0.2, 0.3, 0.4]) {
  const data = new Float64Array(HEADER + 4);
  values.forEach((v, i) => {
    data[HEADER + i] = v;
  });
  return {
    ...tileWorld("0:0:0", VIEWPORT),
    generation: 1,
    header: [2, 2, 200, 2000],
    data,
  };
}

test("the hovered readout is the point kernel when it answers a finite number", () => {
  const store = createStore();
  let rasterCalls = 0;
  const point = { isReady: () => true, sample: () => 0.0743 };
  const raster = {
    sampleAt: () => {
      rasterCalls += 1;
      return 0.5;
    },
  };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.0743);
  assert.equal(rasterCalls, 0);
});

test("the readout falls back to the raster while the point kernel is not ready", () => {
  const store = createStore();
  let pointCalls = 0;
  const point = {
    isReady: () => false,
    sample: () => {
      pointCalls += 1;
      return 0.9;
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.5);
  assert.equal(pointCalls, 0);
});

test("the readout falls back to the raster when the point kernel throws", () => {
  const store = createStore();
  const point = {
    isReady: () => true,
    sample: () => {
      throw new Error("wasm glue exploded");
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  assert.equal(readout(0.08, 0.03), 0.5);
});

test("the readout falls back to the raster when the point kernel returns a non-number", () => {
  const store = createStore();
  for (const bad of [null, undefined, "0.5", { value: 0.5 }]) {
    const point = { isReady: () => true, sample: () => bad };
    const raster = { sampleAt: () => 0.5 };
    const readout = createReadout({ store, point, raster });
    assert.equal(readout(0.08, 0.03), 0.5, `fallback for ${String(bad)}`);
  }
});

test("the readout falls back to the raster when the point kernel returns NaN", () => {
  const store = createStore();
  const point = { isReady: () => true, sample: () => NaN };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster });
  // Without the Number.isFinite check this returns NaN, not the raster value.
  assert.equal(readout(0.08, 0.03), 0.5);
});

test("the readout and the raster comparison read the same (Omega, K)", () => {
  const store = createStore();
  store.generation = 1;
  store.tiles.push(makeTile());
  const calls = [];
  const raster = {
    sampleAt: (tiles, generation, omega, K) => {
      calls.push({ tiles, generation, omega, K });
      return sampleAt(tiles, generation, omega, K);
    },
  };
  const point = { isReady: () => false, sample: () => 0.9 };
  const readout = createReadout({ store, point, raster });
  const rasterSample = createRasterSample({ store, raster });
  // (0.08, 0.03) sits in cell (0, 0) of the 2x2 tile. A readout that samples
  // a shifted point lands in a neighbouring cell and returns 0.2, not 0.1.
  const readoutValue = readout(0.08, 0.03);
  const rasterValue = rasterSample(0.08, 0.03);
  assert.equal(readoutValue, 0.1);
  assert.equal(rasterValue, 0.1);
  assert.equal(readoutValue, rasterValue);
  assert.equal(calls.length, 2);
  for (const call of calls) {
    assert.equal(call.tiles, store.tiles);
    assert.equal(call.generation, 1);
    assert.equal(call.omega, 0.08);
    assert.equal(call.K, 0.03);
  }
});

test("the figure's onPaint stamps each stored record with a paint sequence", () => {
  // Without the stamp every record keys with an empty paint sequence and the
  // stale-bitmap collision liveTileColorKey was built for comes back.
  const store = createStore();
  const pool = { getState: () => ({ generation: 1, viewport: VIEWPORT }) };
  const raster = { tileWorld };
  let redraws = 0;
  const plot = { redraw: () => { redraws += 1; } };
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => plot,
  });
  const data = new Float64Array(HEADER + 4);
  data[0] = 2;
  data[1] = 2;
  data[HEADER] = 0.1;
  data[HEADER + 3] = 0.9;
  const cmd = { id: "0:0:0", generation: 1, header: [2, 2, 200, 1500], data };
  // The same tile painted twice in one generation replaces its record; the
  // replacement must key differently or the cached bitmap goes stale.
  onPaint(cmd);
  const firstKey = liveTileColorKey(store.tiles[0]);
  onPaint(cmd);
  assert.equal(store.tiles.length, 1);
  assert.equal(store.tiles[0].id, "0:0:0");
  assert.equal(store.tiles[0].paintSeq, 2);
  assert.equal(store.generation, 1);
  assert.equal(store.painted, 1);
  assert.equal(redraws, 2);
  // nIter rides in on header[3]; the store started at 2000.
  assert.equal(store.nIter, 1500);
  const stamped = liveTileColorKey(store.tiles[0]);
  assert.notEqual(stamped, firstKey);
  const { paintSeq, ...unstamped } = store.tiles[0];
  assert.notEqual(stamped, liveTileColorKey(unstamped));
});

test("onPaint ignores a stale generation and a tile outside the viewport", () => {
  const store = createStore();
  const pool = { getState: () => ({ generation: 2, viewport: VIEWPORT }) };
  const raster = { tileWorld };
  let redraws = 0;
  const onPaint = createPaintHandler({
    store,
    raster,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  onPaint({ id: "0:0:0", generation: 1, header: [2, 2, 200, 2000], data: [] });
  onPaint({ id: "not-a-tile", generation: 2, header: [2, 2, 200, 2000], data: [] });
  assert.equal(store.tiles.length, 0);
  assert.equal(store.painted, 0);
  assert.equal(redraws, 0);
});

test("onView pushes the domain into the pool and resets the store on a new generation", () => {
  const store = createStore();
  store.generation = 1;
  store.painted = 3;
  store.tiles.push(makeTile());
  const seen = [];
  const pool = {
    setViewport: (v) => seen.push(v),
    getState: () => ({ generation: 2 }),
  };
  let redraws = 0;
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.deepEqual(seen, [{ omegaMin: 0, omegaMax: 0.5, kMin: 0, kMax: 0.15 }]);
  assert.equal(store.generation, 2);
  assert.equal(store.painted, 0);
  assert.equal(store.tiles.length, 0);
  assert.equal(redraws, 1);
});

test("onView keeps the store when the generation did not change, and tolerates no pool", () => {
  const store = createStore();
  store.generation = 2;
  store.painted = 1;
  store.tiles.push(makeTile());
  let redraws = 0;
  const plot = { redraw: () => { redraws += 1; } };
  const pool = {
    setViewport: () => {},
    getState: () => ({ generation: 2 }),
  };
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => plot,
  });
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.equal(store.tiles.length, 1);
  assert.equal(store.painted, 1);
  assert.equal(redraws, 0);
  const noPool = createViewHandler({
    store,
    getPool: () => null,
    getPlot: () => plot,
  });
  noPool({ x0: 0, x1: 0.5, y0: 0, y1: 0.15 });
  assert.equal(store.tiles.length, 1);
});

test("a tile that spent its retry is retried again after onView pans", () => {
  // The retry list lives in the scheduler state and resets on a new
  // generation; onView is the glue that turns a pan into that generation.
  // Drive a real pool so the reset is observed through the same path the
  // page uses, not through a second copy of the wiring.
  const FakeWorker = makeFakeWorkerClass();
  const store = createStore();
  let redraws = 0;
  let pool = null;
  const onView = createViewHandler({
    store,
    getPool: () => pool,
    getPlot: () => ({ redraw: () => { redraws += 1; } }),
  });
  pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 4,
    workerUrl: "fake",
    scheduler: { levels: 1, tileCells: 8 },
  });
  onView({ x0: 0, x1: 1, y0: 0, y1: 0.3 });
  assert.equal(store.generation, 1);
  const [w0, w1, w2, w3] = FakeWorker.instances;
  const tile = w0.posted[0];
  // Lose the tile once in generation 1: it is requeued to a live worker and
  // spends its one retry.
  failWorker(w0);
  assert.equal(w1.posted.length, 1);
  assert.equal(w1.posted[0].id, tile.id);
  // Lose it again: the per-generation cap drops it for good.
  failWorker(w1);
  assert.equal(pool.getState().droppedTiles, 1);
  assert.equal(w2.posted.length, 0);
  // Pan through the glue: generation 2 resets the store and the retry list.
  onView({ x0: 0, x1: 0.5, y0: 0, y1: 0.3 });
  assert.equal(store.generation, 2);
  assert.equal(store.tiles.length, 0);
  assert.equal(store.painted, 0);
  assert.equal(redraws, 2);
  assert.equal(w2.posted.length, 1);
  assert.equal(w2.posted[0].id, tile.id);
  assert.equal(w2.posted[0].generation, 2);
  // Without the per-generation reset of the retry list this third loss drops
  // the tile instead of requeueing it to the last live worker.
  failWorker(w2);
  assert.equal(w3.posted.length, 1);
  assert.equal(w3.posted[0].id, tile.id);
  assert.equal(w3.posted[0].generation, 2);
  assert.equal(pool.getState().droppedTiles, 1);
  pool.destroy();
});

test("LIVE_ARNOLD is frozen and holds the published figure's parameters", () => {
  assert.ok(Object.isFrozen(LIVE_ARNOLD));
  assert.deepEqual({ ...LIVE_ARNOLD }, { nTransient: 200, nIter: 2000, theta0: 0.1 });
  // The store starts at the same nIter the tiles will be computed with.
  assert.equal(createStore().nIter, LIVE_ARNOLD.nIter);
});

test("one params object reaches both the readout and the scheduler options", () => {
  const params = { nTransient: 11, nIter: 222, theta0: 0.33 };
  const store = createStore();
  const calls = [];
  const point = {
    isReady: () => true,
    sample: (...args) => {
      calls.push(args);
      return 0.0743;
    },
  };
  const raster = { sampleAt: () => 0.5 };
  const readout = createReadout({ store, point, raster, params });
  readout(0.08, 0.03);
  assert.deepEqual(calls, [[0.08, 0.03, 11, 222, 0.33]]);
  const scheduler = createSchedulerOptions({ levels: 5, tileCells: 32, params });
  assert.deepEqual(scheduler, {
    levels: 5,
    tileCells: 32,
    nTransient: 11,
    nIter: 222,
    theta0: 0.33,
  });
  // Without params both fall back to LIVE_ARNOLD, so a hard-coded value in
  // either path is caught here too.
  const defaultCalls = [];
  const defaultReadout = createReadout({
    store,
    point: {
      isReady: () => true,
      sample: (...args) => {
        defaultCalls.push(args);
        return 0.1;
      },
    },
    raster,
  });
  defaultReadout(0.08, 0.03);
  assert.deepEqual(defaultCalls, [[0.08, 0.03, 200, 2000, 0.1]]);
  assert.deepEqual(createSchedulerOptions({ levels: 5, tileCells: 32 }), {
    levels: 5,
    tileCells: 32,
    nTransient: 200,
    nIter: 2000,
    theta0: 0.1,
  });
});

test("lockOmega tiles fix Omega at the viewport value and sweep only K", () => {
  // The staircase's tile request: Omega = D with nOmega = 1, K = A over the
  // visible range. Swapping Omega and K in the request is caught here: the
  // tile would carry the A range on the Omega axis instead.
  const viewport = { omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 };
  const tiles = visibleTiles(viewport, { levels: 3, tileCells: 8, lockOmega: true });
  // 1 + 2 + 4 tiles: the pyramid refines K only.
  assert.equal(tiles.length, 7);
  for (const t of tiles) {
    assert.equal(t.omegaMin, 0.25);
    assert.equal(t.omegaMax, 0.25);
    assert.equal(t.nOmega, 1);
    assert.ok(t.kMin >= 0 && t.kMax <= 0.25 && t.kMax > t.kMin);
  }
  const finest = tiles.filter((t) => t.level === 2);
  assert.equal(finest.length, 4);
  assert.equal(finest[0].kMin, 0);
  assert.equal(finest[3].kMax, 0.25);
  // A degenerate Omega range is still rejected without lockOmega.
  assert.equal(visibleTiles(viewport, { levels: 3, tileCells: 8 }).length, 0);
});

test("the staircase view pushes (D, A range) and resets the store once", () => {
  const store = createStore(LIVE_STAIRCASE);
  store.generation = 1;
  store.tiles.push({ id: "0:0:0" });
  const seen = [];
  let gen = 1;
  const pool = {
    setViewport: (v) => {
      seen.push(v);
      gen += 1;
    },
    getState: () => ({ generation: gen }),
  };
  let D = 0.25;
  const dom = { x0: 0, x1: 0.25, y0: 0, y1: 1 };
  const view = createStaircaseView({
    store,
    getPool: () => pool,
    getPlot: () => ({ getDomain: () => dom, redraw: () => {} }),
    getD: () => D,
  });
  view.onView({ x0: 0, x1: 0.25 });
  assert.deepEqual(seen, [{ omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 }]);
  assert.equal(store.generation, 2);
  assert.equal(store.tiles.length, 0);
  // A y-only domain change must not recompute.
  view.onView({ x0: 0, x1: 0.25 });
  assert.equal(seen.length, 1);
  // Moving D pushes the same A range at the new Omega.
  D = 0.4;
  view.applyD();
  assert.deepEqual(seen[1], { omegaMin: 0.4, omegaMax: 0.4, kMin: 0, kMax: 0.25 });
  assert.equal(store.generation, 3);
});

test("tilesToTrace folds 1-D tiles into a sorted polyline, finest level winning", () => {
  const viewport = { omegaMin: 0.25, omegaMax: 0.25, kMin: 0, kMax: 0.25 };
  const issued = visibleTiles(viewport, { levels: 2, tileCells: 4, lockOmega: true });
  const tiles = issued.map((t) => {
    const nK = t.nK;
    const data = new Float64Array(HEADER + nK);
    for (let iy = 0; iy < nK; iy++) {
      const a = nK > 1 ? t.kMin + (iy * (t.kMax - t.kMin)) / (nK - 1) : t.kMin;
      data[HEADER + iy] = a * 10 + t.level; // level-tagged value
    }
    return { ...t, generation: 1, header: [1, nK, 5000, 50000], data };
  });
  const trace = tilesToTrace(tiles, 1);
  assert.equal(trace.x.length, trace.y.length);
  assert.ok(trace.x.length > 4);
  for (let i = 1; i < trace.x.length; i++) assert.ok(trace.x[i] > trace.x[i - 1]);
  // A shared boundary coordinate keeps the finer tile's value (level 1).
  const mid = trace.x.indexOf(0.125);
  assert.ok(mid >= 0);
  assert.equal(trace.y[mid], 0.125 * 10 + 1);
  // Stale-generation tiles are ignored.
  assert.equal(tilesToTrace(tiles, 2).x.length, 0);
});

test("the staircase readout samples (D, A) through the point kernel", () => {
  const store = createStore(LIVE_STAIRCASE);
  const calls = [];
  const point = {
    isReady: () => true,
    sample: (...args) => {
      calls.push(args);
      return 0.2;
    },
  };
  const raster = { sampleAt: () => 0.9 };
  const readout = createStaircaseReadout({ store, point, raster, getD: () => 0.4 });
  assert.equal(readout(0.12), 0.2);
  assert.deepEqual(calls, [[0.4, 0.12, 5000, 50000, 0.1]]);
});

test("the staircase scheduler options carry lockOmega and the paper's counts", () => {
  const opts = createSchedulerOptions({ levels: 5, tileCells: 64, params: LIVE_STAIRCASE, lockOmega: true });
  assert.deepEqual(opts, {
    levels: 5,
    tileCells: 64,
    nTransient: 5000,
    nIter: 50000,
    theta0: 0.1,
    lockOmega: true,
  });
});

// A manual timer pair, as in tests/js/params.js: scheduled callbacks queue
// up and run only when the test flushes them, so the debounce is checked
// without real time.
function manualTimers() {
  const queue = [];
  return {
    queue,
    set: (fn) => {
      queue.push(fn);
      return queue.length - 1;
    },
    clear: (id) => {
      queue[id] = null;
    },
    flush() {
      const fns = queue.splice(0);
      for (const fn of fns) if (fn) fn();
    },
  };
}

const tick = () => new Promise((r) => setImmediate(r));

test("LIVE_ATTRACTORS is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.delayed_logistic_attractors, LIVE_ATTRACTORS);
  assert.equal(LIVE_ATTRACTORS.A, 0.3);
  assert.equal(LIVE_ATTRACTORS.nTransient, 20000);
  const [d, n] = LIVE_ATTRACTORS.paramSpecs;
  // The D slider's range is the wasm kernel's own clamp, so the control can
  // never ask for a value the kernel would silently move.
  assert.deepEqual(
    { name: d.name, min: d.min, max: d.max, default: d.default },
    { name: "D", min: 1.4, max: 3.5, default: 1.9 },
  );
  // The iteration-count control is bounded by the kernel's 4096 cap.
  assert.equal(n.name, "n");
  assert.equal(n.max, 4096);
  assert.ok(Object.isFrozen(LIVE_ATTRACTORS));
});

test("the attractor request passes A then D in the kernel's order", () => {
  // delayed_logistic_attractor_tile(a, d_min, d_max, n_d, n_transient,
  // n_plot, state0): swapping A and D here sends the slider's D into the
  // map's A slot, which this test catches.
  const req = attractorRequest({ D: 1.9, n: 2048 });
  assert.equal(req.a, 0.3);
  assert.equal(req.dMin, 1.9);
  assert.equal(req.dMax, 1.9);
  assert.equal(req.nD, 1);
  assert.equal(req.nTransient, 20000);
  assert.equal(req.nPlot, 2048);
  // The published x0 convention: (fp + 0.01, fp - 0.01) at the analytic
  // fixed point fp = (sqrt(1 + 4D) - 1) / 2D = 0.508572542032378 at D=1.9.
  const fp = attractorFixedPoint(1.9);
  assert.equal(fp, 0.508572542032378);
  assert.equal(req.state0[0], fp + 0.01);
  assert.equal(req.state0[1], fp - 0.01);
});

test("attractorTrace folds the tile into x then y pairs", () => {
  // Header [n_D, n_plot, n_transient, 2], then n_plot (x, y) pairs. An x/y
  // swap in the fold lands y on the x axis, which this test catches.
  const tile = new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]);
  const t = attractorTrace(tile);
  assert.deepEqual(t.x, [0.1, 0.3]);
  assert.deepEqual(t.y, [0.2, 0.4]);
});

test("attractorTrace folds every D block and rejects a short tile", () => {
  const two = new Float64Array([2, 1, 20000, 2, 0.5, 0.6, 0.7, 0.8]);
  const t = attractorTrace(two);
  assert.deepEqual(t.x, [0.5, 0.7]);
  assert.deepEqual(t.y, [0.6, 0.8]);
  assert.deepEqual(attractorTrace(new Float64Array([1, 9, 0, 2, 0.1])).x, []);
  assert.deepEqual(attractorTrace(null).x, []);
});

test("a slider burst recomputes once and a stale result is dropped", async () => {
  const timers = manualTimers();
  const calls = [];
  const deferreds = [];
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return new Promise((r) => deferreds.push(r));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  for (const v of [1.9, 1.95, 2.0]) fig.onInput("D", v);
  timers.flush();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].dMin, 2.0);

  // Two more requests in flight; the older one resolving last must not
  // overwrite the newer result.
  fig.setParams({ D: 1.86 });
  fig.setParams({ D: 2.16 });
  assert.equal(calls.length, 3);
  const tileFor = (d) => new Float64Array([1, 1, 20000, 2, d, -d]);
  deferreds[2](tileFor(2.16));
  await tick();
  deferreds[1](tileFor(1.86));
  await tick();
  assert.deepEqual(fig.store.trace.x, [2.16]);
  assert.deepEqual(fig.store.trace.y, [-2.16]);
  assert.equal(fig.store.generation, 3);
});

test("onTrace runs once per painted result and never for a stale one", async () => {
  // The page repaints and draws the fixed-point marker in onTrace; in
  // fe8a64d it was accepted but never called, and no test noticed.
  const deferreds = [];
  const traced = [];
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: () => new Promise((r) => deferreds.push(r)),
    onTrace: (store, trace, req) => traced.push([trace.x[0], req.dMin]),
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.setParams({ D: 1.86 });
  fig.setParams({ D: 2.16 });
  const tileFor = (d) => new Float64Array([1, 1, 20000, 2, d, -d]);
  deferreds[1](tileFor(2.16));
  await tick();
  deferreds[0](tileFor(1.86));
  await tick();
  assert.deepEqual(traced, [[2.16, 2.16]]);
});

test("the attractor store's trace arrays keep their identity across paints", async () => {
  // The page's panel.traces captures these arrays at mount; replacing them
  // would leave the plot drawing the empty arrays forever.
  const fig = createAttractorFigure({
    debounceMs: 150,
    echo: () => {},
    call: async () => new Float64Array([1, 2, 20000, 2, 0.1, 0.2, 0.3, 0.4]),
    onTrace: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  const xs = fig.store.trace.x;
  const ys = fig.store.trace.y;
  fig.refresh();
  await tick();
  fig.refresh();
  await tick();
  assert.equal(fig.store.trace.x, xs);
  assert.equal(fig.store.trace.y, ys);
  assert.deepEqual([...xs], [0.1, 0.3]);
});

test("LIVE_TORUS is frozen, registered, and holds the paper's parameters", () => {
  assert.equal(LIVE_FIGURES.torus_doubling_attractors, LIVE_TORUS);
  assert.equal(LIVE_TORUS.nTransient, 20000);
  // The paper's per-map constants (maps/torus_doubling.py): map I at
  // A = 0.4 from (0.5, 0.5, 0.5), map IV at A = 0.3 from (0.5, 0.45, 0.52,
  // 0.48); each D window is the published sweep inside the kernel's
  // [1.48, 2.25] clamp, and each default is a published panel.
  const m1 = LIVE_TORUS.maps[1];
  const m4 = LIVE_TORUS.maps[4];
  assert.equal(m1.a, 0.4);
  assert.deepEqual([...m1.x0], [0.5, 0.5, 0.5]);
  assert.deepEqual({ dMin: m1.dMin, dMax: m1.dMax, dDefault: m1.dDefault }, { dMin: 1.9, dMax: 2.25, dDefault: 2.16 });
  assert.equal(m4.a, 0.3);
  assert.deepEqual([...m4.x0], [0.5, 0.45, 0.52, 0.48]);
  assert.deepEqual({ dMin: m4.dMin, dMax: m4.dMax, dDefault: m4.dDefault }, { dMin: 1.48, dMax: 1.53, dDefault: 1.5206 });
  const [map, d, n] = LIVE_TORUS.paramSpecs;
  assert.equal(map.name, "map");
  assert.deepEqual(map.options.map((o) => o.value), [1, 4]);
  assert.equal(d.name, "D");
  assert.equal(n.name, "n");
  assert.equal(n.max, 4096);
  assert.ok(Object.isFrozen(LIVE_TORUS));
});

test("the torus request passes map kind, A, D in the kernel's order", () => {
  // torus_doubling_attractor_tile(map_kind, a, d, n_transient, n_plot,
  // state0): swapping A and D here sends the slider's D into the map's A
  // slot, and dropping the map kind runs the wrong map — this test catches
  // both.
  const req1 = torusRequest({ map: 1, D: 2.16, n: 2048 });
  assert.equal(req1.mapKind, 1);
  assert.equal(req1.a, 0.4);
  assert.equal(req1.d, 2.16);
  assert.equal(req1.nTransient, 20000);
  assert.equal(req1.nPlot, 2048);
  assert.deepEqual([...req1.state0], [0.5, 0.5, 0.5]);

  const req4 = torusRequest({ map: 4, D: 1.5206, n: 512 });
  assert.equal(req4.mapKind, 4);
  assert.equal(req4.a, 0.3);
  assert.equal(req4.d, 1.5206);
  assert.equal(req4.nPlot, 512);
  assert.deepEqual([...req4.state0], [0.5, 0.45, 0.52, 0.48]);

  // A D outside the chosen map's window clamps into it before the kernel
  // ever sees it.
  assert.equal(torusRequest({ map: 4, D: 2.16, n: 512 }).d, 1.53);
  assert.equal(torusRequest({ map: 1, D: 1.5, n: 512 }).d, 1.9);
});

test("torusMapKind snaps the selector the way the kernel does", () => {
  assert.equal(torusMapKind(1), 1);
  assert.equal(torusMapKind(2), 1);
  assert.equal(torusMapKind(3), 4);
  assert.equal(torusMapKind(4), 4);
  assert.equal(torusMapKind(0), 1);
});

test("torusTrace folds the tile by the map's projection", () => {
  // Header [map_kind, dim, n_transient, n_produced], then n_produced states
  // of dim components. The paper scatters components 0 and 1 (X, Y); a
  // swapped projection lands Y on the x axis, which this test catches.
  const tile3 = new Float64Array([1, 3, 20000, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  const t3 = torusTrace(tile3, LIVE_TORUS.maps[1]);
  assert.deepEqual(t3.x, [0.1, 0.4]);
  assert.deepEqual(t3.y, [0.2, 0.5]);

  const tile4 = new Float64Array([4, 4, 20000, 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
  const t4 = torusTrace(tile4, LIVE_TORUS.maps[4]);
  assert.deepEqual(t4.x, [0.1, 0.5]);
  assert.deepEqual(t4.y, [0.2, 0.6]);

  // A short tile, a null tile and a diverged orbit all yield an empty
  // cloud rather than a misread buffer.
  assert.deepEqual(torusTrace(new Float64Array([1, 3, 0, 9, 0.1]), LIVE_TORUS.maps[1]).x, []);
  assert.deepEqual(torusTrace(null, LIVE_TORUS.maps[1]).x, []);
  assert.deepEqual(torusTrace(new Float64Array([4, 4, 20000, 0]), LIVE_TORUS.maps[4]).x, []);
});

test("switching the torus map resets D to the new map's window and repaints", async () => {
  const timers = manualTimers();
  const calls = [];
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(new Float64Array([req.mapKind, req.mapKind === 1 ? 3 : 4, 20000, 1, 0.1, 0.2, 0.3, 0.4]));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  fig.onInput("map", 4);
  timers.flush();
  await tick();
  // The selector snapped to map IV and D moved to that map's published
  // default — a clamp would have pinned it to the window's edge instead.
  assert.equal(fig.state.map, 4);
  assert.equal(fig.state.D, 1.5206);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].mapKind, 4);
  assert.equal(calls[0].a, 0.3);
  assert.equal(calls[0].d, 1.5206);
  assert.deepEqual([...calls[0].state0], [0.5, 0.45, 0.52, 0.48]);
  assert.equal(fig.store.painted, 1);
});

test("a hash state with map and D restores both, in spec order", async () => {
  // parseHash returns name -> value; setParams applies the specs in order,
  // so map=4 resets D to map IV's default and then D=1.515 lands inside
  // the new window. A wiring that applied D before the map switch would
  // clamp it into map I's window first.
  const timers = manualTimers();
  const calls = [];
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: (req) => {
      calls.push(req);
      return Promise.resolve(new Float64Array([4, 4, 20000, 0]));
    },
    onTrace: () => {},
    writeHash: () => {},
    timers,
  });
  fig.setParams({ map: 4, D: 1.515 });
  await tick();
  assert.equal(fig.state.map, 4);
  assert.equal(fig.state.D, 1.515);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].d, 1.515);
});

test("a D outside the active map's window clamps into it", () => {
  const fig = createTorusFigure({
    debounceMs: 150,
    echo: () => {},
    call: () => Promise.resolve(new Float64Array([4, 4, 20000, 0])),
    onTrace: () => {},
    writeHash: () => {},
    timers: manualTimers(),
  });
  fig.setParams({ map: 4 });
  // The number field accepts the union range; the figure clamps into the
  // active window rather than asking the kernel for a D it would move.
  fig.onInput("D", 2.0);
  assert.equal(fig.state.D, 1.53);
});
