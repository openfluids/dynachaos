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
 * @param {(info: object) => void} [options.onCapacityLost]
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
  let capacityAnnounced = false;

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
    // Build incrementally: if spawn throws partway, the workers already
    // created are terminated here. Nothing else holds a reference to them,
    // so without this they would run until the page goes away.
    const spawned = [];
    try {
      for (let i = 0; i < n; i++) {
        spawned.push(spawn());
      }
    } catch (err) {
      for (const worker of spawned) {
        try {
          worker.terminate();
        } catch {
          // A half-dead worker must not mask the original spawn error.
        }
      }
      throw err;
    }
    workers = spawned;
    busy.clear();
    failed.clear();
    capacityAnnounced = false;
  }

  function feed(worker) {
    if (!workers.includes(worker) || failed.has(worker) || busy.has(worker)) return;
    const { state: next, commands } = reduce(state, { type: "idle" });
    state = next;
    const issue = commands.find((cmd) => cmd.type === "issue");
    if (!issue) return;
    busy.set(worker, issue.tile);
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
    if (tile) {
      reduceEvent({
        type: "workerError",
        id: tile.id,
        generation: tile.generation,
        tile,
        liveWorkers: n - failed.size,
      });
      // The failure may have requeued the tile; hand it to an idle worker.
      pump();
    }
    if (!capacityAnnounced && failed.size === workers.length) {
      capacityAnnounced = true;
      if (options.onCapacityLost) options.onCapacityLost(telemetry());
    }
  }

  function onMessage(worker, event) {
    if (!workers.includes(worker) || failed.has(worker)) return;
    const msg = event && event.data;
    if (msg && (msg.type === "result" || msg.type === "error")) {
      const remembered = busy.get(worker);
      busy.delete(worker);
      if (
        remembered &&
        msg.id === remembered.id &&
        msg.generation === remembered.generation
      ) {
        reduceEvent({
          type: msg.type,
          id: remembered.id,
          generation: remembered.generation,
          data: msg.data,
          header: msg.header,
          computeMs: msg.computeMs,
          message: msg.message,
        });
      } else {
        // The reply's identity disagrees with what this worker was given.
        // It is never painted; the remembered tile is freed and lost, so it
        // goes through the same once-per-generation retry as a dead worker.
        log({ event: "mismatch", tile: remembered ? remembered.id : undefined });
        if (remembered) {
          reduceEvent({
            type: "workerError",
            id: remembered.id,
            generation: remembered.generation,
            tile: remembered,
            liveWorkers: n - failed.size,
          });
        }
      }
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
