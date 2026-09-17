// Live Arnold-tongues figure, driven over the Chrome DevTools protocol.
//
//   node tests/e2e/live_figure.mjs <site dir>
//
// Serves the built site, launches headless Chrome, and checks the live path.
// The script uses only node built-ins and the globals that node 22 provides.
// It does not use --virtual-time-budget: that hang waits forever with workers.
//
// Anchor values, from the Python kernel, all below K_c = 1/(2π):
//   rho = 0 at (Omega 0.05, K 0.12) and at (Omega 0.08, K 0.12)
//   rho = 1 at (Omega 0.95, K 0.12)
//   rho ~ 0.0743 at (Omega 0.08, K 0.03)  — this pair catches a flipped K axis

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SITE = resolve(process.argv[2] || "site");
const F = "document.getElementById('fig:arnold_tongues')";
const CHROME_CMDS = ["google-chrome", "google-chrome-stable"];
const PAGE_READY_MS = 20_000;
const FIGURE_MS = 20_000;
const FIRST_PAINT_MS = 30_000;
const REFINE_MS = 45_000;
const SAMPLE_MS = 20_000;
const VIEW_MS = 20_000;
const REDUCED_WAIT_MS = 3_000;
const DEADLINE_SUM_MS =
  2 * (PAGE_READY_MS + FIGURE_MS) +
  FIRST_PAINT_MS +
  REFINE_MS +
  4 * SAMPLE_MS +
  VIEW_MS +
  REDUCED_WAIT_MS;
const WATCHDOG_MS = DEADLINE_SUM_MS + 90_000;
const WASM_RE = /dynachaos_wasm_bg\.wasm/;
const TELEMETRY_EVENTS = ["paint", "drop", "error", "cancel"];
const failures = [];
const pageLogs = [];
const debugLogs = [];
const chromeTried = [];

const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail === undefined ? "" : ` -- ${detail}`}`);
  if (!ok) failures.push(name);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const freePort = () =>
  new Promise((res) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => res(p));
    });
  });

if (!existsSync(SITE)) {
  console.log(`FAIL the site directory does not exist: ${SITE}`);
  process.exit(1);
}

function formatConsole(event) {
  const args = (event.params && event.params.args) || [];
  return args
    .map((a) => {
      if (a.value !== undefined) return typeof a.value === "string" ? a.value : JSON.stringify(a.value);
      if (a.preview && a.preview.properties) {
        const fields = a.preview.properties.map((p) => `${p.name}:${p.value}`).join(",");
        return `{${fields}}`;
      }
      return a.description || a.type || "";
    })
    .join(" ");
}

function isTelemetry(line) {
  return TELEMETRY_EVENTS.some(
    (event) => line.includes(`"event":"${event}"`) || line.includes(`event:${event}`),
  );
}

function dumpConsole() {
  const lines = debugLogs.length ? debugLogs.concat(pageLogs) : pageLogs;
  console.log("--- page console ---");
  if (lines.length === 0) {
    console.log("(empty)");
    return;
  }
  const interesting = lines.filter(isTelemetry);
  console.log(`${lines.length} debug lines; ${interesting.length} paint/drop/error/cancel`);
  const shown =
    interesting.length <= 12 ? interesting : interesting.slice(0, 6).concat(["..."], interesting.slice(-6));
  for (const line of shown) console.log(line);
  console.log(lines[lines.length - 1]);
}

let chrome = null;
let server = null;
let profile = null;
let chromeArgs = [];
let cleaned = false;

function chromeCommandLine(cmd) {
  return [cmd, ...chromeArgs].join(" ");
}

function failChromeStart(extra) {
  const tried = chromeTried.length ? chromeTried.join(" ; ") : chromeCommandLine(CHROME_CMDS[0]);
  console.log(`FAIL Chrome did not start. Command tried: ${tried}${extra ? ` -- ${extra}` : ""}`);
}

async function stopChild(child, { group = false } = {}) {
  if (!child || child.pid == null) return;
  if (child.exitCode != null || child.signalCode != null) return;
  const pid = child.pid;
  const exited = new Promise((resolve) => child.once("exit", resolve));
  try {
    if (group) process.kill(-pid, "SIGTERM");
    else child.kill("SIGTERM");
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {}
  }
  const outcome = await Promise.race([exited.then(() => "exited"), sleep(2_000).then(() => "timeout")]);
  if (outcome === "timeout") {
    try {
      if (group) process.kill(-pid, "SIGKILL");
    } catch {}
    try {
      child.kill("SIGKILL");
    } catch {}
    await Promise.race([exited, sleep(1_000)]);
  }
}

async function removeProfile() {
  if (!profile) return;
  for (let i = 0; i < 8; i++) {
    try {
      rmSync(profile, { recursive: true, force: true });
      if (!existsSync(profile)) return;
    } catch {}
    await sleep(250);
  }
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {}
}

async function cleanup() {
  if (cleaned) return;
  cleaned = true;
  await stopChild(chrome, { group: true });
  await stopChild(server);
  await removeProfile();
}

const watchdog = setTimeout(() => {
  console.log(`FAIL the browser check ran past ${WATCHDOG_MS / 1000} s`);
  try {
    dumpConsole();
  } catch {}
  cleanup().finally(() => process.exit(1));
}, WATCHDOG_MS);

const httpPort = await freePort();
const cdpPort = await freePort();
profile = mkdtempSync(join(tmpdir(), "dc-115-"));
chromeArgs = [
  // Chrome ignores an emulated prefers-reduced-data value unless this Blink
  // feature is on. Without the flag the media query stays false.
  "--headless=new",
  "--disable-gpu",
  "--no-sandbox",
  "--no-first-run",
  "--enable-blink-features=PrefersReducedData",
  `--user-data-dir=${profile}`,
  `--remote-debugging-port=${cdpPort}`,
  "--window-size=1400,1000",
  "about:blank",
];

async function startChrome() {
  for (const cmd of CHROME_CMDS) {
    chromeTried.push(chromeCommandLine(cmd));
    const child = spawn(cmd, chromeArgs, { stdio: "ignore", detached: true });
    const err = await new Promise((resolve) => {
      const onError = (e) => resolve(e);
      child.once("error", onError);
      child.once("spawn", () => {
        child.off("error", onError);
        resolve(null);
      });
    });
    if (!err) return child;
  }
  return null;
}

server = spawn(
  "python3",
  ["-m", "http.server", String(httpPort), "--bind", "127.0.0.1", "--directory", SITE],
  { stdio: "ignore" },
);
chrome = await startChrome();

async function up(url) {
  for (let i = 0; i < 150; i++) {
    try {
      await fetch(url);
      return;
    } catch {
      await sleep(100);
    }
  }
  throw new Error(`not reachable: ${url}`);
}

let exitCode = 1;
try {
  if (!chrome) {
    failChromeStart();
    throw new Error("chrome-missing");
  }
  try {
    await up(`http://127.0.0.1:${cdpPort}/json/version`);
  } catch (err) {
    failChromeStart(err.message);
    throw err;
  }
  await up(`http://127.0.0.1:${httpPort}/index.html`);
  const targets = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
  const page = targets.find((t) => t.type === "page");
  if (!page) throw new Error("Chrome has no page target");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => {
    ws.onopen = r;
    ws.onerror = j;
  });
  let nextId = 0;
  const waiters = new Map();
  const events = [];
  const attached = new Map();
  let attachChain = Promise.resolve();
  const send = (method, params = {}, sessionId) =>
    new Promise((r) => {
      const id = ++nextId;
      waiters.set(id, r);
      const payload = { id, method, params };
      if (sessionId) payload.sessionId = sessionId;
      ws.send(JSON.stringify(payload));
    });
  const onAttached = (params) => {
    attached.set(params.sessionId, params.targetInfo);
    attachChain = attachChain
      .then(async () => {
        await send("Network.enable", {}, params.sessionId);
        await send("Runtime.enable", {}, params.sessionId);
        if (params.waitingForDebugger) {
          await send("Runtime.runIfWaitingForDebugger", {}, params.sessionId);
        }
      })
      .catch(() => {});
  };
  ws.onmessage = (m) => {
    let d;
    try {
      d = JSON.parse(m.data);
    } catch {
      return;
    }
    if (d.id != null && waiters.has(d.id)) {
      waiters.get(d.id)(d);
      waiters.delete(d.id);
    } else if (d.method) {
      events.push(d);
      if (d.method === "Runtime.consoleAPICalled") pageLogs.push(formatConsole(d));
      if (d.method === "Target.attachedToTarget") onAttached(d.params);
      if (d.method === "Target.detachedFromTarget") attached.delete(d.params.sessionId);
    }
  };
  const ev = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.result && r.result.exceptionDetails) {
      const x = r.result.exceptionDetails;
      throw new Error((x.exception && x.exception.description) || x.text);
    }
    return r.result && r.result.result ? r.result.result.value : undefined;
  };
  const waitFor = async (expression, ms, step = 100) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      await attachChain;
      try {
        if (await ev(expression)) return true;
      } catch {}
      await sleep(step);
    }
    return false;
  };
  const pageUrl = `http://127.0.0.1:${httpPort}/index.html?debug=1`;
  const installLogHook = () =>
    ev(`(() => {
      if (window.__dcHooked) return true;
      window.__dcHooked = true;
      window.__dcLog = [];
      const orig = console.info.bind(console);
      console.info = function () {
        try {
          window.__dcLog.push(
            Array.from(arguments)
              .map((a) => (a && typeof a === "object" ? JSON.stringify(a) : String(a)))
              .join(" ")
          );
        } catch (e) {}
        return orig.apply(console, arguments);
      };
      return true;
    })()`);
  const pullDebug = async () => {
    try {
      const hooked = await ev("window.__dcLog || []");
      if (Array.isArray(hooked)) debugLogs.push(...hooked);
    } catch {}
  };
  const load = async () => {
    await send("Page.navigate", { url: pageUrl });
    await waitFor("document.readyState === 'complete'", PAGE_READY_MS);
    await waitFor(`!!${F}`, FIGURE_MS);
    await installLogHook();
  };
  const stats = () => ev(`${F} && ${F}._live ? ${F}._live.stats() : null`);
  const wasmHits = (from) =>
    events.slice(from).filter(
      (e) =>
        e.method === "Network.requestWillBeSent" &&
        e.params &&
        e.params.request &&
        WASM_RE.test(e.params.request.url),
    );
  const attachedWorkers = () => [...attached.values()].filter((t) => t && t.type === "worker");
  const imgShown = () =>
    ev(`(() => {
      const img = ${F} && ${F}.querySelector(".fig-body img");
      if (!img) return false;
      const s = getComputedStyle(img);
      return s.display !== "none" && s.visibility !== "hidden";
    })()`);

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Network.enable");
  await send("Network.setCacheDisabled", { cacheDisabled: true });
  await send("Target.setAutoAttach", {
    autoAttach: true,
    waitForDebuggerOnStart: true,
    flatten: true,
  });

  await load();
  check(
    "the figure is marked live-capable",
    await ev(`!!${F} && !!${F}.dataset.live`),
    await ev(`${F} ? JSON.stringify(${F}.dataset) : "no figure"`),
  );
  check("it has an interact button", await ev(`!!(${F} && ${F}.querySelector(".act-interact"))`));

  await ev(`(${F}.querySelector(".act-interact").click(), true)`);
  const firstPaint = await waitFor(`!!${F}._live && ${F}._live.stats().painted >= 1`, FIRST_PAINT_MS, 50);
  await attachChain;
  const sFirst = await stats();
  check("the first coarse tile is painted", firstPaint && sFirst && sFirst.painted >= 1, JSON.stringify(sFirst));
  check("pressing interact spawns workers", !!(sFirst && sFirst.workerCount >= 1), JSON.stringify(sFirst));

  const captured = firstPaint
    ? await ev(`(() => {
        const c = ${F}.querySelector("canvas.plot");
        if (!c) return false;
        window.__dcCoarse = c.getContext("2d").getImageData(0, 0, c.width, c.height);
        return true;
      })()`)
    : false;
  const coarsePainted = sFirst && sFirst.painted ? sFirst.painted : 0;

  let lastPainted = coarsePainted;
  let stableSince = Date.now();
  let refined = false;
  if (firstPaint) {
    const refineT0 = Date.now();
    while (Date.now() - refineT0 < REFINE_MS) {
      await attachChain;
      const p = ((await stats()) || {}).painted || 0;
      if (p !== lastPainted) {
        lastPainted = p;
        stableSince = Date.now();
      } else if (lastPainted > coarsePainted && Date.now() - stableSince >= 2_000) {
        refined = true;
        break;
      }
      await sleep(150);
    }
  }
  const compare = captured
    ? await ev(`(() => {
        const c = ${F}.querySelector("canvas.plot");
        const coarse = window.__dcCoarse;
        if (!c || !coarse) return { comparable: false, reason: "missing canvas" };
        if (c.width !== coarse.width || c.height !== coarse.height) {
          return {
            comparable: false,
            reason: "canvas resized",
            coarseWidth: coarse.width,
            coarseHeight: coarse.height,
            width: c.width,
            height: c.height,
          };
        }
        const a = coarse.data;
        const b = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
        let differ = false;
        for (let i = 0; i < a.length; i++) {
          if (a[i] !== b[i]) {
            differ = true;
            break;
          }
        }
        return { comparable: true, differ };
      })()`)
    : { comparable: false, reason: "no coarse capture" };

  check(
    "the picture changes between the coarse pass and the refined pass",
    Boolean(captured && refined && lastPainted > coarsePainted && compare && compare.comparable && compare.differ),
    JSON.stringify({ captured, coarsePainted, refinedPainted: lastPainted, refined, compare }),
  );

  const sLive = await stats();
  const nIter = sLive && sLive.nIter ? sLive.nIter : 2000;
  const tol = 2 / nIter;
  const sampleMs = firstPaint ? SAMPLE_MS : 2_000;
  const sampleWhenReady = async (omega, K, ms = sampleMs) => {
    const hasLive = await ev(`${F} && ${F}._live`);
    if (!hasLive) return null;
    await waitFor(`${F}._live.sample(${omega}, ${K}) !== null`, ms);
    return ev(`${F}._live ? ${F}._live.sample(${omega}, ${K}) : null`);
  };
  const a = await sampleWhenReady(0.05, 0.12);
  check("rho = 0 at (Omega 0.05, K 0.12)", a !== null && Math.abs(a) <= tol, a);
  const b = await sampleWhenReady(0.95, 0.12);
  check("rho = 1 at (Omega 0.95, K 0.12)", b !== null && Math.abs(b - 1) <= tol, b);
  const c = await sampleWhenReady(0.08, 0.12);
  check("rho = 0 at (Omega 0.08, K 0.12)", c !== null && Math.abs(c) <= tol, c);
  const d = await sampleWhenReady(0.08, 0.03);
  check("rho ~ 0.0743 at (Omega 0.08, K 0.03)", d !== null && Math.abs(d - 0.0743) < 0.03, d);
  check(
    "readout at locked (Omega 0.05, K 0.12) is 0 to 1e-12",
    a !== null && Math.abs(a) <= 1e-12,
    a,
  );
  const displayTol = 1 / 256;
  const readoutPairs = [
    [0.05, 0.12, a],
    [0.95, 0.12, b],
    [0.08, 0.12, c],
    [0.08, 0.03, d],
  ];
  const vsRaster = [];
  for (const [omega, K, value] of readoutPairs) {
    if (value == null) {
      vsRaster.push({ omega, K, value, raster: null, ok: false });
      continue;
    }
    await waitFor(
      `${F}._live && typeof ${F}._live.rasterSample === "function" && ${F}._live.rasterSample(${omega}, ${K}) !== null`,
      sampleMs,
    );
    const rasterVal = await ev(
      `${F}._live && typeof ${F}._live.rasterSample === "function" ? ${F}._live.rasterSample(${omega}, ${K}) : null`,
    );
    const ok =
      rasterVal !== null &&
      Number.isFinite(rasterVal) &&
      Math.abs(value - rasterVal) <= displayTol;
    vsRaster.push({ omega, K, value, raster: rasterVal, ok });
  }
  check(
    "readout never differs from the raster sample by more than the display tolerance",
    vsRaster.length > 0 && vsRaster.every((row) => row.ok),
    JSON.stringify(vsRaster),
  );

  // Every check above passes on the raster value alone, because a locked cell
  // reads the same either way. This one does not: it imports the same module
  // instance the page imported and demands the readout return that value, at
  // an unlocked point where the two paths differ.
  await waitFor(`${F}._live && ${F}._live.pointReady && ${F}._live.pointReady()`, sampleMs);
  const wired = await ev(`(async () => {
    const m = await import(new URL("live/point.js", document.baseURI).href);
    const direct = m.sample(0.08, 0.03, 200, 2000, 0.1);
    return {
      ready: m.isReady(),
      direct,
      readout: ${F}._live.sample(0.08, 0.03),
      raster: ${F}._live.rasterSample(0.08, 0.03),
    };
  })()`);
  check(
    "the readout is the point kernel, not the tile lookup",
    wired != null && wired.ready === true && wired.direct !== null && wired.readout === wired.direct,
    JSON.stringify(wired),
  );

  const sBeforeView = await stats();
  if (sBeforeView && (await ev(`!!${F}._live && typeof ${F}._live.setView === "function"`))) {
    await ev(`(${F}._live.setView({ omegaMin: 0, omegaMax: 0.3, kMin: 0, kMax: 0.15 }), true)`);
    const grew = await waitFor(
      `${F}._live.stats().generation > ${sBeforeView.generation} && ${F}._live.stats().painted >= 1`,
      firstPaint ? VIEW_MS : 3_000,
    );
    check(
      "at least one refinement generation completes after a view change",
      grew,
      JSON.stringify({ before: sBeforeView, after: await stats() }),
    );
  } else {
    check("at least one refinement generation completes after a view change", false, "no _live.setView");
  }

  const sEnd = await stats();
  check(
    "every worker is still alive",
    Boolean(sEnd && sEnd.liveWorkers >= 1 && sEnd.liveWorkers === sEnd.workerCount),
    JSON.stringify(sEnd),
  );
  await pullDebug();

  await send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-data", value: "reduce" }],
  });
  await load();
  const reducedMatches = await ev(`matchMedia("(prefers-reduced-data: reduce)").matches`);
  if (!reducedMatches) {
    check(
      "Chrome honours prefers-reduced-data with --enable-blink-features=PrefersReducedData",
      false,
      "matchMedia stayed false, so the reduced-data checks prove nothing",
    );
  } else {
    const from = events.length;
    await ev(`(${F}.querySelector(".act-interact") && ${F}.querySelector(".act-interact").click(), true)`);
    await sleep(REDUCED_WAIT_MS);
    await attachChain;
    const st = await stats();
    const hasLive = await ev(`!!(${F} && ${F}._live)`);
    const liveCanvas = await ev(`!!(${F} && ${F}.querySelector("canvas.plot.live"))`);
    const jsonChart = await ev(
      `!!(${F} && Array.from(${F}.querySelectorAll("canvas.plot")).some((el) => !el.classList.contains("live")))`,
    );
    const pngAfter = await imgShown();
    const shown = Boolean(pngAfter || jsonChart);
    const workers = attachedWorkers();
    const hits = wasmHits(from);
    const reducedState = shown && !liveCanvas;
    check(
      "under prefers-reduced-data no tile is computed",
      workers.length === 0 && reducedState && (!hasLive || (st != null && st.painted === 0)),
      JSON.stringify({
        st,
        hasLive,
        workers: workers.map((t) => t.url || t.type),
        shown,
        liveCanvas,
        pngAfter,
        jsonChart,
        state: await ev(`${F} && ${F}.dataset.state`),
      }),
    );
    check(
      "under prefers-reduced-data the WebAssembly module is never downloaded",
      hits.length === 0,
      hits
        .map((e) => e.params.request.url)
        .slice(0, 4)
        .join(" | "),
    );
    check(
      "under prefers-reduced-data no worker target is attached",
      workers.length === 0,
      JSON.stringify(workers.map((t) => ({ type: t.type, url: t.url }))),
    );
    check(
      "under prefers-reduced-data a PNG or JSON chart is shown and no live canvas is created",
      reducedState,
      JSON.stringify({
        liveCanvas,
        pngAfter,
        jsonChart,
        state: await ev(`${F} && ${F}.dataset.state`),
      }),
    );
  }
  await pullDebug();

  const thrown = events.filter((x) => x.method === "Runtime.exceptionThrown");
  check(
    "the page raises no uncaught exception",
    thrown.length === 0,
    thrown
      .map((x) => (x.params.exceptionDetails.exception || {}).description || x.params.exceptionDetails.text)
      .join(" | ")
      .slice(0, 400),
  );

  dumpConsole();
  ws.close();
  exitCode = failures.length === 0 ? 0 : 1;
  console.log(exitCode === 0 ? "BROWSER PASS" : `BROWSER FAIL (${failures.length})`);
} catch (err) {
  if (!(err && err.message === "chrome-missing")) {
    console.log(`FAIL the browser check crashed: ${err && err.message}`);
  }
  dumpConsole();
} finally {
  clearTimeout(watchdog);
  await cleanup();
}
process.exit(exitCode);
