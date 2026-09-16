/**
 * Browser glue around the pure scheduler: a pool of independent workers,
 * each holding its own WebAssembly instance.
 *
 * N = min(navigator.hardwareConcurrency, 8). Pan or zoom bumps the generation
 * stamp; workers stay alive so the in-flight tile can finish, and a late
 * result is dropped rather than painted. terminate() runs only in destroy().
 *
 * Workers do not share memory. GitHub Pages cannot set COOP/COEP, so each
 * worker keeps its own wasm instance and only postMessage the finished tile.
 */

import { initialState, reduce } from "./scheduler.js";

const MAX_WORKERS = 8;

/**
 * @param {number} [hardwareConcurrency]
 * @returns {number}
 */
export function workerCount(hardwareConcurrency) {
  const raw = Number(hardwareConcurrency);
  const cores = Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : 1;
  return Math.min(cores, MAX_WORKERS);
}

/**
 * True when `?debug=1` is on the page URL or `localStorage.dynachaosDebug`
 * is set. Used by the e2e smoke and by a bug report from the wild.
 *
 * @param {object} [env]
 * @returns {boolean}
 */
export function debugEnabled(env = globalThis) {
  try {
    const storage = env.localStorage;
    if (storage && storage.dynachaosDebug) return true;
  } catch {
    // localStorage can throw in a locked-down document.
  }
  try {
    const search = env.location && env.location.search;
    if (typeof search === "string") {
      return new URLSearchParams(search).get("debug") === "1";
    }
  } catch {
    // location may be inaccessible.
  }
  return false;
}

/**
 * @param {object} [options]
 * @param {typeof Worker} [options.Worker]
 * @param {number} [options.hardwareConcurrency]
 * @param {URL|string} [options.workerUrl]
 * @param {(cmd: object) => void} [options.onPaint]
 * @param {(cmd: object) => void} [options.onDrop]
 * @param {object} [options.scheduler]
 * @returns {object}
 */
export function createPool(options = {}) {
  const WorkerImpl = options.Worker || globalThis.Worker;
  if (typeof WorkerImpl !== "function") {
    throw new Error("createPool needs Worker");
  }
  const n = workerCount(
    options.hardwareConcurrency ??
      (typeof navigator !== "undefined" ? navigator.hardwareConcurrency : 1),
  );
  const workerUrl = options.workerUrl || new URL("./tile-worker.js", import.meta.url);
  let state = initialState(options.scheduler);
  let workers = [];
  const busy = new Map();
  const failed = new Set();

  function telemetry(extra) {
    return {
      workerCount: n,
      // A worker that fails is retired, so the pool can lose capacity and
      // eventually stop with a non-empty queue. Report what is left, or that
      // stop looks the same as a figure that finished.
      liveWorkers: n - failed.size,
      queueDepth: state.queueDepth,
      droppedGenerations: state.droppedGenerations,
      droppedTiles: state.droppedTiles,
      generation: state.generation,
      ...extra,
    };
  }

  function log(extra) {
    if (!debugEnabled()) return;
    console.info("[dynachaos-live]", telemetry(extra));
  }

  function dispatch(commands) {
    for (const cmd of commands) {
      if (cmd.type === "paint") {
        log({ event: "paint", tile: cmd.id, computeMs: cmd.computeMs });
        if (options.onPaint) options.onPaint(cmd);
      } else if (cmd.type === "drop") {
        log({ event: "drop", tile: cmd.id, computeMs: cmd.computeMs });
        if (options.onDrop) options.onDrop(cmd);
      }
    }
  }

  function reduceEvent(event) {
    const { state: next, commands } = reduce(state, event);
    state = next;
    dispatch(commands);
  }

  function spawn() {
    const worker = new WorkerImpl(workerUrl, { type: "module" });
    worker.onmessage = (event) => onMessage(worker, event);
    worker.onerror = () => onWorkerFailure(worker);
    return worker;
  }

  function spawnAll() {
    workers = Array.from({ length: n }, spawn);
    busy.clear();
    failed.clear();
  }

  function feed(worker) {
    if (!workers.includes(worker) || failed.has(worker) || busy.has(worker)) return;
    const { state: next, commands } = reduce(state, { type: "idle" });
    state = next;
    const issue = commands.find((cmd) => cmd.type === "issue");
    if (!issue) return;
    busy.set(worker, { id: issue.tile.id, generation: issue.tile.generation });
    worker.postMessage({ type: "tile", ...issue.tile });
    log({ event: "issue", tile: issue.tile.id, computeMs: undefined });
  }

  function pump() {
    for (const worker of workers) {
      feed(worker);
    }
  }

  function onWorkerFailure(worker) {
    if (!workers.includes(worker) || failed.has(worker)) return;
    const tile = busy.get(worker);
    busy.delete(worker);
    failed.add(worker);
    log({ event: "error" });
    if (!tile) return;
    reduceEvent({
      type: "error",
      id: tile.id,
      generation: tile.generation,
    });
  }

  function onMessage(worker, event) {
    if (!workers.includes(worker) || failed.has(worker)) return;
    const msg = event && event.data;
    if (msg && (msg.type === "result" || msg.type === "error")) {
      busy.delete(worker);
      reduceEvent({
        type: msg.type,
        id: msg.id,
        generation: msg.generation,
        data: msg.data,
        header: msg.header,
        computeMs: msg.computeMs,
        message: msg.message,
      });
      feed(worker);
      return;
    }
    onWorkerFailure(worker);
  }

  spawnAll();

  return {
    workerCount: n,
    get liveWorkers() {
      return n - failed.size;
    },
    setViewport(viewport) {
      const { state: next, commands } = reduce(state, { type: "viewport", viewport });
      state = next;
      if (commands.some((cmd) => cmd.type === "cancel")) {
        log({ event: "cancel" });
      }
      pump();
    },
    destroy() {
      for (const worker of workers) {
        worker.terminate();
      }
      workers = [];
      busy.clear();
      failed.clear();
    },
    getState() {
      return state;
    },
  };
}
