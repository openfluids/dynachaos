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

/**
 * The figure's mutable tile store. `tiles` is the same array the Plot()
 * live adapter reads, so a generation reset empties it in place.
 *
 * @returns {{ tiles: object[], generation: number, painted: number, nIter: number }}
 */
export function createStore() {
  return { tiles: [], generation: 0, painted: 0, nIter: 2000 };
}

/**
 * The pool's onPaint: fold one finished tile into the store and repaint.
 * A stale-generation command and a tile id outside the viewport are dropped.
 * Every stored record carries a paint sequence so a re-painted tile keys
 * differently in the colour cache — without it the stale bitmap shows.
 * nIter rides in on the tile header so the readout tolerance tracks the
 * kernel's actual iteration count.
 *
 * @param {object} deps
 * @param {object} deps.store
 * @param {{ tileWorld: (id: string, viewport: object) => object | null }} deps.raster
 * @param {() => object} deps.getPool
 * @param {() => object} deps.getPlot
 * @returns {(cmd: object) => void}
 */
export function createPaintHandler({ store, raster, getPool, getPlot }) {
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
 * @returns {(omega: number, K: number) => number | null}
 */
export function createReadout({ store, point, raster }) {
  return function sampleReadout(omega, K) {
    if (point.isReady()) {
      try {
        const rho = point.sample(omega, K, 200, 2000, 0.1);
        if (typeof rho === "number" && Number.isFinite(rho)) return rho;
      } catch (_) {}
    }
    return raster.sampleAt(store.tiles, store.generation, omega, K);
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
