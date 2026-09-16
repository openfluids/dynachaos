import { test } from "node:test";
import assert from "node:assert/strict";
import {
  HEADER,
  liveTileColorKey,
  sampleAt,
  tileCellsFor,
  tilePixelRect,
  tileWorld,
} from "../../site-src/live/raster.js";

test("tilePixelRect puts kMin (row 0) at the bottom of the canvas", () => {
  const plot = { x0: 0, x1: 1, y0: 0, y1: 1, left: 10, top: 20, width: 100, height: 80 };
  const lower = tilePixelRect(
    { omegaMin: 0, omegaMax: 1, kMin: 0, kMax: 0.5 },
    plot,
  );
  const upper = tilePixelRect(
    { omegaMin: 0, omegaMax: 1, kMin: 0.5, kMax: 1 },
    plot,
  );
  assert.equal(lower.x, 10);
  assert.equal(lower.w, 100);
  // y grows down: the kMin half sits below the kMax half.
  assert.equal(lower.y, 20 + 40);
  assert.equal(lower.h, 40);
  assert.equal(upper.y, 20);
  assert.equal(upper.h, 40);
  assert.ok(lower.y > upper.y);
});

test("tilePixelRect maps omegaMin to the left edge", () => {
  const plot = { x0: 0, x1: 2, y0: 0, y1: 1, left: 0, top: 0, width: 200, height: 50 };
  const right = tilePixelRect(
    { omegaMin: 1, omegaMax: 2, kMin: 0, kMax: 1 },
    plot,
  );
  assert.equal(right.x, 100);
  assert.equal(right.w, 100);
});

function tile(overrides) {
  return {
    id: "0:0:0",
    generation: 1,
    level: 0,
    omegaMin: 0,
    omegaMax: 1,
    kMin: 0,
    kMax: 1,
    header: [2, 2, 200, 2000],
    // row 0 = kMin: 10, 11; row 1 = kMax: 20, 21
    data: Float64Array.of(2, 2, 200, 2000, 10, 11, 20, 21),
    ...overrides,
  };
}

test("sampleAt reads row 0 as kMin, not the top of the canvas", () => {
  const tiles = [tile()];
  assert.equal(sampleAt(tiles, 1, 0.25, 0.25), 10);
  assert.equal(sampleAt(tiles, 1, 0.75, 0.25), 11);
  assert.equal(sampleAt(tiles, 1, 0.25, 0.75), 20);
  assert.equal(sampleAt(tiles, 1, 0.75, 0.75), 21);
});

test("sampleAt uses the finest current-generation tile covering the point", () => {
  const coarse = tile({ level: 0, id: "0:0:0" });
  const fine = tile({
    id: "1:0:0",
    level: 1,
    omegaMin: 0,
    omegaMax: 0.5,
    kMin: 0,
    kMax: 0.5,
    header: [1, 1, 200, 2000],
    data: Float64Array.of(1, 1, 200, 2000, 42),
  });
  assert.equal(sampleAt([coarse, fine], 1, 0.25, 0.25), 42);
  assert.equal(sampleAt([coarse, fine], 1, 0.75, 0.25), 11);
  assert.equal(sampleAt([coarse, fine], 2, 0.25, 0.25), null);
  assert.equal(sampleAt([coarse], 1, 2, 0.5), null);
});

test("sampleAt returns null when no tile of this generation covers the point", () => {
  assert.equal(sampleAt([], 1, 0.5, 0.5), null);
  assert.equal(sampleAt([tile({ generation: 3 })], 1, 0.5, 0.5), null);
});

test("tileWorld recovers the scheduler rectangle from an id", () => {
  const view = { omegaMin: 0, omegaMax: 1, kMin: 0, kMax: 0.4 };
  const t = tileWorld("1:1:0", view);
  assert.equal(t.level, 1);
  assert.equal(t.ix, 1);
  assert.equal(t.iy, 0);
  assert.equal(t.omegaMin, 0.5);
  assert.equal(t.omegaMax, 1);
  assert.equal(t.kMin, 0);
  assert.equal(t.kMax, 0.2);
});

test("tileCellsFor stays inside the kernel caps and the step budget", () => {
  const nIter = 2000;
  const nTransient = 200;
  const cells = tileCellsFor(1600, 400, { levels: 3, nIter, nTransient });
  assert.ok(cells >= 1);
  assert.ok(cells <= 512);
  assert.ok(cells * cells * (nTransient + nIter) <= 250_000_000);
  const tiny = tileCellsFor(8, 8, { levels: 3, nIter, nTransient });
  assert.equal(tiny, 2);
  assert.equal(HEADER, 4);
});

test("liveTileColorKey ignores theme and changes with tile identity", () => {
  const data = new Float64Array(HEADER + 4);
  data[0] = 2;
  data[1] = 2;
  data[HEADER] = 0.1;
  data[HEADER + 3] = 0.9;
  const tile = {
    id: "0:0:0",
    generation: 3,
    header: [2, 2, 200, 2000],
    data,
    theme: "dark",
  };
  const key = liveTileColorKey(tile);
  assert.equal(liveTileColorKey({ ...tile, theme: "light" }), key);
  assert.notEqual(liveTileColorKey({ ...tile, generation: 4 }), key);
  assert.notEqual(liveTileColorKey({ ...tile, id: "1:0:0" }), key);
  const other = new Float64Array(data);
  other[HEADER] = 0.5;
  assert.notEqual(liveTileColorKey({ ...tile, data: other }), key);
  assert.equal(liveTileColorKey.length, 1);
});
