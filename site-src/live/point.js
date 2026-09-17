/**
 * Main-thread sample of one (Omega, K) point.
 *
 * Loads the wasm glue lazily and answers sample() synchronously once
 * ready. The readout path is called per pointermove; an async round trip
 * through the worker pool would buy nothing for ~30 microseconds.
 *
 * Lock detection stays on; the display-tolerance stop does not. Unlocked
 * cells run the full n_iter average.
 *
 * This file is copied to site/live/, so ../wasm/ is site/wasm/.
 */

import init, { initSync, rotation_number_point } from "../wasm/dynachaos_wasm.js";

let ready = false;
let loading = null;

export function isReady() {
  return ready;
}

export function ensureLoaded() {
  if (loading == null) loading = loadWasm();
  return loading;
}

async function loadWasm() {
  // The browser glue fetches the .wasm next to itself. Node cannot fetch a
  // file:// URL, so a smoke run reads the bytes and instantiates them.
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  if (!isNode) {
    await init();
  } else {
    const { readFile } = await import("node:fs/promises");
    const bytes = await readFile(new URL("../wasm/dynachaos_wasm_bg.wasm", import.meta.url));
    initSync({ module: bytes });
  }
  ready = true;
}

/**
 * Rotation number of one point. `null` until the module is ready, or if
 * the arguments are not finite.
 *
 * @param {number} omega
 * @param {number} K
 * @param {number} [nTransient]
 * @param {number} [nIter]
 * @param {number} [theta0]
 * @returns {number | null}
 */
export function sample(omega, K, nTransient, nIter, theta0) {
  if (!ready) {
    ensureLoaded();
    return null;
  }
  if (!Number.isFinite(omega) || !Number.isFinite(K)) return null;
  const transient = Number.isFinite(nTransient) ? nTransient : 200;
  const iter = Number.isFinite(nIter) ? nIter : 2000;
  const theta = Number.isFinite(theta0) ? theta0 : 0.1;
  const out = rotation_number_point(omega, K, transient, iter, theta);
  if (out == null || out.length < 1) return null;
  const value = out[0];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
