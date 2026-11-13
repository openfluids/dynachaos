import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  const levels = 5;
  const cells = tileCellsFor(1600, 400, { levels, nIter, nTransient });
  assert.equal(cells, 100);
  assert.equal(cells * 2 ** (levels - 1), 1600);
  assert.ok(cells * cells * (nTransient + nIter) <= 250_000_000);
  const tiny = tileCellsFor(8, 8, { levels, nIter, nTransient });
  assert.equal(tiny, 1);
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
  const last = new Float64Array(data);
  last[last.length - 1] = 0.3;
  assert.notEqual(liveTileColorKey({ ...tile, data: last }), key);
  const longer = new Float64Array(HEADER + 8);
  longer.set(data);
  assert.notEqual(liveTileColorKey({ ...tile, data: longer }), key);
  assert.notEqual(liveTileColorKey({ ...tile, header: [4, 2, 200, 2000] }), key);
});

test("liveTileColorKey misses on a replaced record and hits an unchanged one", () => {
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
    paintSeq: 7,
  };
  const key = liveTileColorKey(tile);
  // The same record keeps its key: the bitmap cache still hits.
  assert.equal(liveTileColorKey(tile), key);
  // A record that replaces it inside the same generation — identical id,
  // header, and samples — must key differently or the stale bitmap shows.
  const replaced = { ...tile, paintSeq: 8 };
  assert.notEqual(liveTileColorKey(replaced), key);
  // A record with no paint sequence still keys on the old fingerprint.
  const { paintSeq, ...unstamped } = tile;
  assert.equal(liveTileColorKey(unstamped), liveTileColorKey({ ...unstamped }));
});

// The onPaint paint-sequence assertions moved to tests/js/live-figure.js,
// which imports the handler from site-src/live/live-figure.js.

// The page's colour ramp is code inside the inline script of
// scripts/paper_shell.py, which no module imports. Cut it out of the source
// and run it against the definition it replaced: linear interpolation
// between the colour stops, rounded to the nearest integer. Every comparison
// is exact (tolerance 0): the channels are integers.
function pageRamp(cmap, zmin = 0, zmax = 1, css = () => "#000000") {
  const text = readFileSync(new URL("../../scripts/paper_shell.py", import.meta.url), "utf8");
  const start = text.indexOf("const hex=c=>");
  const marker = text.indexOf("function rampRGB(t)", start);
  assert.ok(start >= 0 && marker > start, "the ramp code moved in paper_shell.py: update this test");
  const body = text.slice(start, text.indexOf("\n", marker));
  const make = new Function(
    "panel",
    "zmin",
    "zmax",
    "css",
    `${body}\nreturn { rampRGB, rampInto, round8, STOPS };`,
  );
  return make({ cmap }, zmin, zmax, css);
}

function referenceRamp(stops, t) {
  const c = Math.max(0, Math.min(1, t));
  const k = c * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(k));
  return [0, 1, 2].map((j) => Math.round(stops[i][j] + (stops[i + 1][j] - stops[i][j]) * (k - i)));
}

for (const cmap of ["viridis", "magma"]) {
  test(`the page's ${cmap} ramp hits every colour stop and clamps outside [0, 1]`, () => {
    const { rampRGB, STOPS } = pageRamp(cmap);
    STOPS.forEach((stop, i) => {
      assert.deepEqual(rampRGB(i / (STOPS.length - 1)), stop, `stop ${i}`);
    });
    assert.deepEqual(rampRGB(-0.5), STOPS[0]);
    assert.deepEqual(rampRGB(1.5), STOPS[STOPS.length - 1]);
  });

  test(`the page's ${cmap} ramp equals rounded linear interpolation between the stops`, () => {
    const { rampRGB, STOPS } = pageRamp(cmap);
    const n = STOPS.length - 1;
    const ts = [];
    for (let i = 0; i <= 4000; i++) ts.push(i / 4000);
    // The stop boundaries, a few ulps either side.
    for (let i = 0; i <= n; i++) for (const d of [-3e-16, -1e-16, 0, 1e-16, 3e-16]) ts.push(i / n + d);
    for (const t of ts) assert.deepEqual(rampRGB(t), referenceRamp(STOPS, t), `t = ${t}`);
  });

  test(`the page's ${cmap} ramp writes its three channels at the given offset only`, () => {
    const { rampInto, rampRGB } = pageRamp(cmap);
    for (const t of [0, 0.25, 0.5, 0.999, 1]) {
      const buf = new Uint8ClampedArray(12).fill(7);
      rampInto(t, buf, 4);
      assert.deepEqual(Array.from(buf.subarray(4, 7)), rampRGB(t), `t = ${t}`);
      assert.deepEqual(Array.from(buf.subarray(0, 4)), [7, 7, 7, 7]);
      assert.deepEqual(Array.from(buf.subarray(7)), [7, 7, 7, 7, 7]);
    }
  });
}

test("the page's channel rounding equals Math.round on every non-negative value tried", () => {
  // floor(x + 0.5) is wrong for one double, 0.5 - 2^-54, so the page tests
  // x < 0.5 first. Try that value and the doubles around every integer and
  // half-integer from 0 to 256.
  const { round8 } = pageRamp("magma");
  const f64 = new Float64Array(1);
  const u64 = new BigUint64Array(f64.buffer);
  const nudge = (x, n) => {
    f64[0] = x;
    u64[0] += BigInt(n);
    return f64[0];
  };
  assert.equal(round8(0.49999999999999994), 0);
  for (let h = 1; h <= 256; h++) {
    for (const base of [h - 0.5, h]) {
      for (let n = -8; n <= 8; n++) {
        const x = nudge(base, n);
        assert.equal(round8(x), Math.round(x), `x = ${x}`);
      }
    }
  }
  for (let n = 0; n <= 8; n++) assert.equal(round8(nudge(0, n)), 0);
});

test("the page's signed ramp interpolates between the three theme colours", () => {
  // The signed map reads its stops from the theme at every call; stub three
  // distinct colours and check the ends, the middle and both half-way points.
  const theme = { "--locked": "#102030", "--torus": "#405060", "--chaotic": "#708090" };
  const stops = [
    [0x10, 0x20, 0x30],
    [0x40, 0x50, 0x60],
    [0x70, 0x80, 0x90],
  ];
  const { rampRGB, rampInto } = pageRamp("signed", -1, 1, (name) => theme[name]);
  assert.deepEqual(rampRGB(0), stops[0]);
  assert.deepEqual(rampRGB(0.5), stops[1]);
  assert.deepEqual(rampRGB(1), stops[2]);
  for (const t of [0.25, 0.75, 0.1, 0.9, 0.3333]) {
    assert.deepEqual(rampRGB(t), referenceRamp(stops, t), `t = ${t}`);
  }
  const buf = new Uint8ClampedArray(9).fill(7);
  rampInto(0.25, buf, 3);
  assert.deepEqual(Array.from(buf), [7, 7, 7, ...referenceRamp(stops, 0.25), 7, 7, 7]);
});
