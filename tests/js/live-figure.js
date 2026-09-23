import { test } from "node:test";
import assert from "node:assert/strict";
import {
  createPaintHandler,
  createRasterSample,
  createReadout,
  createStore,
  createViewHandler,
} from "../../site-src/live/live-figure.js";
import {
  HEADER,
  liveTileColorKey,
  sampleAt,
  tileWorld,
} from "../../site-src/live/raster.js";
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
