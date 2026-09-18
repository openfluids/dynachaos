import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { ensureLoaded, isReady, sample } from "../../site/live/point.js";

const N_TRANSIENT = 200;
const N_ITER = 2000;
const THETA0 = 0.1;

async function loadPointIsolate(tag) {
  const url = new URL("../../site/live/point.js", import.meta.url);
  url.searchParams.set("isolate", tag);
  return import(url.href);
}

async function writeStubGlue(source) {
  const dir = await mkdtemp(join(tmpdir(), "dc-point-"));
  const file = join(dir, "glue.js");
  await writeFile(file, source);
  return pathToFileURL(file).href;
}

test("point path matches native locked and unlocked references", async () => {
  await ensureLoaded();
  assert.equal(isReady(), true);

  // These three inputs cannot tell the point kernel from the tile kernel:
  // two locked tongues and K = 0 all return the same number either way.
  // rust/core/src/circle_map.rs::point_kernel_unlocked_cell_matches_full
  // at (Omega 0.08, K 0.03) is the guard that the display stop is off.
  // Native locked cells return the exact rational (error 0).
  assert.equal(sample(0.05, 0.12, N_TRANSIENT, N_ITER, THETA0), 0);
  assert.equal(sample(0.5, 0.2, N_TRANSIENT, N_ITER, THETA0), 0.5);

  // Unlocked K = 0: the drift per step is Omega, matching rotation_number_full.
  const golden = (Math.sqrt(5) - 1) / 2;
  const unlocked = sample(golden, 0, N_TRANSIENT, N_ITER, THETA0);
  assert.equal(typeof unlocked, "number");
  assert.ok(
    Math.abs(unlocked - golden) <= 1e-12,
    `unlocked golden vs Omega: |${unlocked} - ${golden}| > 1e-12`,
  );
});

test("a glue without rotation_number_point leaves sample() null", async () => {
  const glue = await writeStubGlue("export function other() { return 1; }\n");
  const point = await loadPointIsolate("missing");
  await assert.rejects(() => point.ensureLoaded(glue));
  assert.equal(point.isReady(), false);
  assert.equal(point.sample(0.08, 0.03, N_TRANSIENT, N_ITER, THETA0), null);
});

test("a failed load retries once instead of poisoning the module", async () => {
  const glue = await writeStubGlue("export function other() { return 1; }\n");
  const point = await loadPointIsolate("retry");
  const first = point.ensureLoaded(glue);
  await assert.rejects(first);
  const second = point.ensureLoaded(glue);
  assert.notEqual(second, first);
  await assert.rejects(second);
  assert.equal(point.isReady(), false);
  assert.equal(point.sample(0.5, 0.2, N_TRANSIENT, N_ITER, THETA0), null);
});
