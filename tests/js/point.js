import { test } from "node:test";
import assert from "node:assert/strict";
import { ensureLoaded, isReady, sample } from "../../site/live/point.js";

const N_TRANSIENT = 200;
const N_ITER = 2000;
const THETA0 = 0.1;

test("point path matches native locked and unlocked references", async () => {
  await ensureLoaded();
  assert.equal(isReady(), true);

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
