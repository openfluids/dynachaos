import { test } from "node:test";
import assert from "node:assert/strict";
import {
  initialState,
  reduce,
  tileCellsAtLevel,
  visibleTiles,
  tileId,
} from "../../site-src/live/scheduler.js";
import { createPool, debugEnabled, workerCount } from "../../site-src/live/pool.js";

const VIEWPORT = { omegaMin: 0, omegaMax: 1, kMin: 0, kMax: 0.3 };
const OPTIONS = { levels: 3, tileCells: 8 };

function viewportEvent(viewport = VIEWPORT) {
  return { type: "viewport", viewport };
}

function drain(state, limit = 1000) {
  const issued = [];
  for (let i = 0; i < limit; i++) {
    const out = reduce(state, { type: "idle" });
    state = out.state;
    const issue = out.commands.find((cmd) => cmd.type === "issue");
    if (!issue) break;
    issued.push(issue.tile);
  }
  return { state, issued };
}

test("scheduler is a pure function importable by node", () => {
  const state = initialState(OPTIONS);
  Object.freeze(state);
  Object.freeze(state.pending);
  Object.freeze(state.issued);
  Object.freeze(state.inFlight);
  Object.freeze(state.options);
  const out = reduce(state, viewportEvent());
  assert.equal(state.generation, 0);
  assert.equal(state.pending.length, 0);
  assert.equal(out.state.generation, 1);
  assert.ok(out.state.pending.length > 0);
  assert.equal(typeof document, "undefined");
  assert.equal(typeof window, "undefined");
});

test("tiles are issued coarse-first", () => {
  let { state } = reduce(initialState(OPTIONS), viewportEvent());
  const { issued } = drain(state);
  assert.ok(issued.length > 0);
  const levels = issued.map((tile) => tile.level);
  for (let i = 1; i < levels.length; i++) {
    assert.ok(levels[i] >= levels[i - 1], `level ${levels[i]} after ${levels[i - 1]}`);
  }
  const lastOf = (level) => {
    const idx = levels.lastIndexOf(level);
    assert.ok(idx >= 0, `missing level ${level}`);
    return idx;
  };
  const firstOf = (level) => {
    const idx = levels.indexOf(level);
    assert.ok(idx >= 0, `missing level ${level}`);
    return idx;
  };
  assert.ok(lastOf(0) < firstOf(1));
  assert.ok(lastOf(1) < firstOf(2));
  const expected = visibleTiles(VIEWPORT, OPTIONS);
  assert.deepEqual(
    issued.map((tile) => tile.id),
    expected.map((tile) => tile.id),
  );
});

test("coarser pyramid levels use fewer cells than the finest", () => {
  const options = { levels: 4, tileCells: 16 };
  assert.equal(tileCellsAtLevel(3, options), 16);
  assert.equal(tileCellsAtLevel(2, options), 8);
  assert.equal(tileCellsAtLevel(1, options), 4);
  assert.equal(tileCellsAtLevel(0, options), 2);
  const tiles = visibleTiles(VIEWPORT, options);
  const byLevel = (level) => tiles.filter((tile) => tile.level === level);
  assert.equal(byLevel(0)[0].nOmega, 2);
  assert.equal(byLevel(0)[0].nK, 2);
  assert.equal(byLevel(3)[0].nOmega, 16);
  assert.equal(byLevel(3)[0].nK, 16);
  assert.ok(byLevel(0)[0].nOmega * byLevel(0)[0].nOmega < byLevel(3)[0].nOmega * byLevel(3)[0].nOmega);
  assert.equal(tileCellsAtLevel(-1, options), 2);
  assert.equal(tileCellsAtLevel(99, options), 16);
});

test("a generation counter drops stale results instead of painting them", () => {
  let { state, commands } = reduce(initialState(OPTIONS), viewportEvent());
  ({ state, commands } = reduce(state, { type: "idle" }));
  const tile = commands.find((cmd) => cmd.type === "issue").tile;
  const firstGeneration = tile.generation;
  ({ state, commands } = reduce(state, {
    type: "viewport",
    viewport: { ...VIEWPORT, omegaMax: 0.5 },
  }));
  assert.ok(commands.some((cmd) => cmd.type === "cancel"));
  assert.equal(state.generation, firstGeneration + 1);
  assert.equal(state.droppedGenerations, 1);
  assert.equal(state.droppedTiles, 0);

  ({ state, commands } = reduce(state, {
    type: "result",
    id: tile.id,
    generation: firstGeneration,
    data: [1, 2, 3],
    computeMs: 4,
  }));
  assert.ok(commands.every((cmd) => cmd.type !== "paint"));
  assert.equal(commands.length, 1);
  assert.equal(commands[0].type, "drop");
  assert.equal(commands[0].reason, "stale");
  assert.equal(commands[0].generation, firstGeneration);
  assert.equal(state.droppedGenerations, 1);
  assert.equal(state.droppedTiles, 1);

  ({ state, commands } = reduce(state, { type: "idle" }));
  const fresh = commands.find((cmd) => cmd.type === "issue").tile;
  ({ state, commands } = reduce(state, {
    type: "result",
    id: fresh.id,
    generation: fresh.generation,
    data: [9],
    computeMs: 1.5,
  }));
  assert.equal(commands.length, 1);
  assert.equal(commands[0].type, "paint");
  assert.equal(commands[0].generation, fresh.generation);
  assert.deepEqual(commands[0].data, [9]);
});

test("every visible tile is issued exactly once per generation", () => {
  const expected = visibleTiles(VIEWPORT, OPTIONS);
  let { state } = reduce(initialState(OPTIONS), viewportEvent());
  let { state: drained, issued } = drain(state);
  assert.equal(issued.length, expected.length);
  assert.equal(new Set(issued.map((tile) => tile.id)).size, expected.length);
  assert.deepEqual(
    issued.map((tile) => tile.id).sort(),
    expected.map((tile) => tile.id).sort(),
  );

  const extra = reduce(drained, { type: "idle" });
  assert.equal(extra.commands.filter((cmd) => cmd.type === "issue").length, 0);
  assert.equal(extra.state.issued.length, expected.length);

  ({ state } = reduce(drained, viewportEvent()));
  assert.equal(state.generation, 2);
  ({ issued } = drain(state));
  assert.equal(issued.length, expected.length);
  assert.equal(new Set(issued.map((tile) => tile.id)).size, expected.length);
  assert.ok(issued.every((tile) => tile.generation === 2));
  assert.equal(tileId(0, 0, 0), "0:0:0");
});

test("worker count is min(hardwareConcurrency, 8)", () => {
  assert.equal(workerCount(32), 8);
  assert.equal(workerCount(4), 4);
  assert.equal(workerCount(1), 1);
  assert.equal(workerCount(undefined), 1);
  assert.equal(workerCount(0), 1);
});

test("debug telemetry is gated by query or localStorage", () => {
  assert.equal(debugEnabled({}), false);
  assert.equal(debugEnabled({ location: { search: "?debug=1" } }), true);
  assert.equal(debugEnabled({ location: { search: "?foo=1" } }), false);
  assert.equal(debugEnabled({ localStorage: { dynachaosDebug: "1" } }), true);
});

test("a degenerate viewport is rejected", () => {
  const start = initialState(OPTIONS);
  const cases = [
    { omegaMin: 0, omegaMax: 0, kMin: 0, kMax: 1 },
    { omegaMin: 1, omegaMax: 0, kMin: 0, kMax: 1 },
    { omegaMin: 0, omegaMax: 1, kMin: 0.3, kMax: 0.3 },
    { omegaMin: 0, omegaMax: 1, kMin: 0.3, kMax: 0.1 },
  ];
  for (const viewport of cases) {
    assert.deepEqual(visibleTiles(viewport, OPTIONS), []);
    const out = reduce(start, { type: "viewport", viewport });
    assert.equal(out.state, start);
    assert.equal(out.commands.length, 0);
    assert.equal(out.state.generation, 0);
  }
});

test("a stale result does not evict the current generation's in-flight entry", () => {
  let { state, commands } = reduce(initialState({ levels: 1 }), viewportEvent());
  ({ state, commands } = reduce(state, { type: "idle" }));
  const stale = commands.find((cmd) => cmd.type === "issue").tile;
  ({ state } = reduce(state, {
    type: "viewport",
    viewport: { ...VIEWPORT, omegaMax: 0.5 },
  }));
  ({ state, commands } = reduce(state, { type: "idle" }));
  const live = commands.find((cmd) => cmd.type === "issue").tile;
  assert.equal(live.id, stale.id);
  assert.notEqual(live.generation, stale.generation);
  assert.deepEqual(state.inFlight, [{ id: live.id, generation: live.generation }]);

  ({ state, commands } = reduce(state, {
    type: "result",
    id: stale.id,
    generation: stale.generation,
    data: [1],
  }));
  assert.equal(commands[0].type, "drop");
  assert.equal(commands[0].reason, "stale");
  assert.deepEqual(state.inFlight, [{ id: live.id, generation: live.generation }]);
  assert.equal(state.droppedTiles, 1);
  assert.equal(state.droppedGenerations, 1);
});

test("an error reply frees its tile and does not retry", () => {
  let { state, commands } = reduce(initialState({ levels: 1 }), viewportEvent());
  ({ state, commands } = reduce(state, { type: "idle" }));
  const tile = commands.find((cmd) => cmd.type === "issue").tile;
  assert.deepEqual(state.inFlight, [{ id: tile.id, generation: tile.generation }]);

  ({ state, commands } = reduce(state, {
    type: "error",
    id: tile.id,
    generation: tile.generation,
    message: "empty tile",
    computeMs: 0.5,
  }));
  assert.equal(state.inFlight.length, 0);
  assert.equal(commands.length, 1);
  assert.equal(commands[0].type, "drop");
  assert.equal(commands[0].reason, "error");
  assert.equal(commands[0].id, tile.id);
  assert.equal(state.droppedTiles, 1);
  assert.equal(state.droppedGenerations, 0);
  const extra = reduce(state, { type: "idle" });
  assert.equal(extra.commands.filter((cmd) => cmd.type === "issue").length, 0);
});

function makeFakeWorkerClass() {
  const instances = [];
  class FakeWorker {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.terminated = false;
      this.terminateCount = 0;
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
      this.terminateCount += 1;
    }
    deliver(data) {
      if (typeof this.onmessage === "function") {
        this.onmessage({ data });
      }
    }
    deliverRaw(data) {
      if (typeof this.onmessage === "function") {
        this.onmessage({ data });
      }
    }
  }
  FakeWorker.instances = instances;
  return FakeWorker;
}

function failWorker(worker) {
  if (typeof worker.onerror === "function") {
    worker.onerror();
  }
}

test("a pan does not terminate any worker", () => {
  const FakeWorker = makeFakeWorkerClass();
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 4,
    workerUrl: "fake",
    scheduler: { levels: 2, tileCells: 4 },
  });
  assert.equal(FakeWorker.instances.length, 4);
  pool.setViewport(VIEWPORT);
  assert.ok(FakeWorker.instances.some((worker) => worker.posted.length > 0));
  pool.setViewport({ ...VIEWPORT, omegaMax: 0.5 });
  assert.equal(FakeWorker.instances.length, 4);
  assert.ok(FakeWorker.instances.every((worker) => worker.terminateCount === 0));
  pool.destroy();
  assert.ok(FakeWorker.instances.every((worker) => worker.terminated));
  assert.ok(FakeWorker.instances.every((worker) => worker.terminateCount === 1));
});

test("a stale pool result leaves the current generation's in-flight entry intact", () => {
  const FakeWorker = makeFakeWorkerClass();
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 2,
    workerUrl: "fake",
    scheduler: { levels: 1, tileCells: 8 },
  });
  pool.setViewport(VIEWPORT);
  const [w0, w1] = FakeWorker.instances;
  assert.equal(w0.posted.length, 1);
  assert.equal(w1.posted.length, 0);
  const stale = w0.posted[0];
  pool.setViewport({ ...VIEWPORT, omegaMax: 0.5 });
  assert.equal(w0.terminateCount, 0);
  assert.equal(w1.posted.length, 1);
  const live = w1.posted[0];
  assert.equal(live.id, stale.id);
  assert.equal(live.generation, stale.generation + 1);
  assert.deepEqual(pool.getState().inFlight, [
    { id: live.id, generation: live.generation },
  ]);
  w0.deliver({
    type: "result",
    id: stale.id,
    generation: stale.generation,
    data: [1, 2, 3],
    computeMs: 1,
  });
  assert.deepEqual(pool.getState().inFlight, [
    { id: live.id, generation: live.generation },
  ]);
  assert.equal(pool.getState().droppedTiles, 1);
  assert.equal(pool.getState().droppedGenerations, 1);
  pool.destroy();
});

test("an error reply from the pool frees its tile", () => {
  const FakeWorker = makeFakeWorkerClass();
  const drops = [];
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 1,
    workerUrl: "fake",
    scheduler: { levels: 1, tileCells: 8 },
    onDrop: (cmd) => drops.push(cmd),
  });
  pool.setViewport(VIEWPORT);
  const worker = FakeWorker.instances[0];
  const tile = worker.posted[0];
  assert.equal(pool.getState().inFlight.length, 1);
  worker.deliver({
    type: "error",
    id: tile.id,
    generation: tile.generation,
    message: "empty tile",
    computeMs: 0.5,
  });
  assert.equal(pool.getState().inFlight.length, 0);
  assert.equal(drops.length, 1);
  assert.equal(drops[0].type, "drop");
  assert.equal(drops[0].reason, "error");
  assert.equal(drops[0].id, tile.id);
  pool.destroy();
});

test("droppedGenerations and droppedTiles move independently", () => {
  const FakeWorker = makeFakeWorkerClass();
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 1,
    workerUrl: "fake",
    scheduler: { levels: 2, tileCells: 4 },
  });
  pool.setViewport(VIEWPORT);
  assert.equal(pool.getState().droppedGenerations, 0);
  assert.equal(pool.getState().droppedTiles, 0);
  const worker = FakeWorker.instances[0];
  const first = worker.posted[0];
  pool.setViewport({ ...VIEWPORT, omegaMax: 0.5 });
  assert.equal(pool.getState().droppedGenerations, 1);
  assert.equal(pool.getState().droppedTiles, 0);
  worker.deliver({
    type: "result",
    id: first.id,
    generation: first.generation,
    data: [0],
    computeMs: 1,
  });
  assert.equal(pool.getState().droppedGenerations, 1);
  assert.equal(pool.getState().droppedTiles, 1);
  const current = worker.posted[worker.posted.length - 1];
  assert.equal(current.generation, 2);
  worker.deliver({
    type: "error",
    id: current.id,
    generation: current.generation,
    message: "empty tile",
  });
  assert.equal(pool.getState().droppedGenerations, 1);
  assert.equal(pool.getState().droppedTiles, 2);
  pool.destroy();
});

test("a worker-level failure frees its tile and is not handed more work", () => {
  const FakeWorker = makeFakeWorkerClass();
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 1,
    workerUrl: "fake",
    scheduler: { levels: 2, tileCells: 4 },
  });
  pool.setViewport(VIEWPORT);
  const worker = FakeWorker.instances[0];
  const tile = worker.posted[0];
  const posted = worker.posted.length;
  assert.equal(pool.getState().inFlight.length, 1);
  assert.equal(pool.getState().droppedTiles, 0);
  failWorker(worker);
  // Reverted-repair check: if onerror still only busy.delete + feed, this fails.
  assert.equal(pool.getState().inFlight.length, 0);
  assert.equal(
    pool.getState().inFlight.some(
      (item) => item.id === tile.id && item.generation === tile.generation,
    ),
    false,
  );
  assert.equal(pool.getState().droppedTiles, 1);
  assert.equal(worker.posted.length, posted);
  pool.setViewport({ ...VIEWPORT, omegaMax: 0.5 });
  assert.equal(worker.posted.length, posted);
  pool.destroy();
});

test("a malformed message frees its tile and is not handed more work", () => {
  for (const raw of [undefined, { type: "nonsense" }]) {
    const FakeWorker = makeFakeWorkerClass();
    const pool = createPool({
      Worker: FakeWorker,
      hardwareConcurrency: 1,
      workerUrl: "fake",
      scheduler: { levels: 2, tileCells: 4 },
    });
    pool.setViewport(VIEWPORT);
    const worker = FakeWorker.instances[0];
    const tile = worker.posted[0];
    const posted = worker.posted.length;
    assert.equal(pool.getState().inFlight.length, 1);
    assert.equal(pool.getState().droppedTiles, 0);
    worker.deliverRaw(raw);
    // Reverted-repair check: if onMessage still ignores unknown/missing types
    // and feeds the worker, this fails.
    assert.equal(pool.getState().inFlight.length, 0);
    assert.equal(
      pool.getState().inFlight.some(
        (item) => item.id === tile.id && item.generation === tile.generation,
      ),
      false,
    );
    assert.equal(pool.getState().droppedTiles, 1);
    assert.equal(worker.posted.length, posted);
    pool.setViewport({ ...VIEWPORT, omegaMax: 0.5 });
    assert.equal(worker.posted.length, posted);
    pool.destroy();
  }
});

test("a message from a worker no longer in the pool is ignored", () => {
  const FakeWorker = makeFakeWorkerClass();
  const paints = [];
  const drops = [];
  const pool = createPool({
    Worker: FakeWorker,
    hardwareConcurrency: 1,
    workerUrl: "fake",
    scheduler: { levels: 1, tileCells: 8 },
    onPaint: (cmd) => paints.push(cmd),
    onDrop: (cmd) => drops.push(cmd),
  });
  pool.setViewport(VIEWPORT);
  const worker = FakeWorker.instances[0];
  const tile = worker.posted[0];
  const posted = worker.posted.length;
  const inFlight = pool.getState().inFlight.slice();
  pool.destroy();
  worker.deliver({
    type: "result",
    id: tile.id,
    generation: tile.generation,
    data: [1],
    computeMs: 1,
  });
  assert.equal(paints.length, 0);
  assert.equal(drops.length, 0);
  assert.equal(worker.posted.length, posted);
  assert.deepEqual(pool.getState().inFlight, inFlight);
});
