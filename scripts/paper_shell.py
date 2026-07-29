"""Presentation shell for the interactive paper: fonts, tokens, CSS and JS.

Kept separate from ``build_paper.py`` so the design can be edited without
touching the conversion pipeline. Everything here is inlined into the generated
page except the webfonts, which are served from ``site/fonts/``.

Two decisions worth stating, because they drive the rest:

* The three accent hues encode the Lyapunov classification used throughout the
  study --- locked, quasiperiodic, chaotic. The hero canvas colours the logistic
  attractor by that same rule, so the legend teaches the palette before the
  reader meets a figure.
* Figures render compact by default. A reader scanning the argument should not
  wade through full-bleed plots; interaction and magnification are opt-in, one
  click away, and the heavy chart payload is never fetched until asked for.
"""

# ruff: noqa: E501 -- inlined CSS and JS are kept in their authored form.

from __future__ import annotations

FONT_FACES = (
    ("TeX Gyre Pagella", "pagella-regular.woff2", "normal", "400"),
    ("TeX Gyre Pagella", "pagella-italic.woff2", "italic", "400"),
    ("TeX Gyre Pagella", "pagella-bold.woff2", "normal", "700"),
    ("TeX Gyre Pagella", "pagella-bolditalic.woff2", "italic", "700"),
    ("TeX Gyre Pagella Math", "pagella-math.woff2", "normal", "400"),
)


def _font_css() -> str:
    return "\n".join(
        f"@font-face{{font-family:'{family}';src:url('fonts/{file}') format('woff2');"
        f"font-style:{style};font-weight:{weight};font-display:swap;}}"
        for family, file, style, weight in FONT_FACES
    )


CSS = (
    _font_css()
    + r"""
:root{
  --locked:#a8620a;   /* lambda < 0  mode-locked / periodic */
  --torus:#54489f;    /* lambda ~ 0  quasiperiodic */
  --chaotic:#0d6f6a;  /* lambda > 0  chaotic */

  --ground:#faf8f3;
  --raised:#ffffff;
  --sunken:#efece3;
  --ink:#14161c;
  --ink-mid:#454b59;
  --ink-low:#828899;
  --rule:#e0dccf;
  --rule-soft:#ebe7dc;

  --serif:'TeX Gyre Pagella', Palatino, 'Palatino Linotype', Georgia, serif;
  --math:'TeX Gyre Pagella Math', 'TeX Gyre Pagella', serif;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;

  --base:1.155rem;
  --measure:clamp(30rem,16.5rem + 24vw,42rem);
  --rail:clamp(12rem,14vw,17rem);
  --bleed:clamp(34rem,6rem + 44vw,66rem);
  --gutter:clamp(1.15rem,3.5vw,2.75rem);
  --shadow:0 1px 2px rgba(20,22,28,.05), 0 8px 28px rgba(20,22,28,.07);
}

:root[data-size="s"]{--base:1.0rem;--measure:clamp(28rem,16.75rem + 20vw,38rem);}
:root[data-size="m"]{--base:1.075rem;--measure:clamp(29rem,16.6rem + 22vw,40rem);}
:root[data-size="l"]{--base:1.155rem;--measure:clamp(30rem,16.5rem + 24vw,42rem);}
:root[data-size="xl"]{--base:1.26rem;--measure:clamp(31rem,16.4rem + 26vw,44rem);}

@media (prefers-color-scheme: dark){
  :root{
    --locked:#f0a830; --torus:#9086dc; --chaotic:#3fd0c8;
    --ground:#0a0d15; --raised:#111624; --sunken:#070911;
    --ink:#e9edf7; --ink-mid:#a6b0c7; --ink-low:#6b748f;
    --rule:#1e2537; --rule-soft:#161c2b;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.45);
  }
}
:root[data-theme="dark"]{
  --locked:#f0a830; --torus:#9086dc; --chaotic:#3fd0c8;
  --ground:#0a0d15; --raised:#111624; --sunken:#070911;
  --ink:#e9edf7; --ink-mid:#a6b0c7; --ink-low:#6b748f;
  --rule:#1e2537; --rule-soft:#161c2b;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.45);
}
:root[data-theme="light"]{
  --locked:#a8620a; --torus:#54489f; --chaotic:#0d6f6a;
  --ground:#faf8f3; --raised:#ffffff; --sunken:#efece3;
  --ink:#14161c; --ink-mid:#454b59; --ink-low:#828899;
  --rule:#e0dccf; --rule-soft:#ebe7dc;
  --shadow:0 1px 2px rgba(20,22,28,.05), 0 8px 28px rgba(20,22,28,.07);
}

*{box-sizing:border-box;}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%;}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font-family:var(--serif);font-size:var(--base);line-height:1.6;
  text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;
  overflow-x:hidden;
  transition:background-color .35s ease,color .35s ease;
}
@media (prefers-reduced-motion:reduce){body{transition:none;}}

math{font-family:var(--math);}
math[display="block"]{display:block;margin:0;overflow-x:auto;overflow-y:hidden;max-width:100%;}
/* the raw-LaTeX annotation is stripped at build time; belt and braces */
annotation,annotation-xml{display:none;}
.eqn{display:grid;grid-template-columns:1fr auto;align-items:center;gap:1rem;
  margin:1.65em 0;padding:0.15em 0;}
.eqn > math{min-width:0;}
.eqn .eqno{font-variant-numeric:tabular-nums;color:var(--ink-low);font-size:0.92em;flex:none;}
math[display="block"]:not(.eqn > math){margin:1.5em auto;}

/* ------------------------------ type ------------------------------ */
.eyebrow{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.26em;text-transform:uppercase;color:var(--ink-low);margin:0;}
h1{font-size:clamp(2.1rem,0.9rem+4.6vw,4.6rem);line-height:1.0;letter-spacing:-0.032em;font-weight:700;
  margin:0.38em 0 0;max-width:17ch;text-wrap:balance;}
h2{font-size:clamp(1.35rem,1.1rem+0.9vw,1.75rem);line-height:1.2;letter-spacing:-0.014em;font-weight:700;margin:0 0 0.6em;text-wrap:balance;}
h3{font-size:1.06rem;font-weight:700;margin:2em 0 0.35em;text-wrap:balance;}
p{margin:0 0 0.95em;}
article ul,article ol{margin:0 0 1em;padding-left:1.35em;}
article li{margin-bottom:0.4em;}
article li > p{margin:0 0 0.4em;}
article li > p:last-child{margin-bottom:0;}
article li > ul,article li > ol{margin-top:0.4em;}
/* pandoc leaves these wrappers behind for LaTeX environments it half-converts */
.center,.minipage,.tabularx{margin:1.2rem 0;}
.tabularx{overflow-x:auto;font-size:0.85em;}
a{color:var(--chaotic);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 35%,transparent);}
a:hover{border-bottom-color:currentColor;}
:focus-visible{outline:2px solid var(--chaotic);outline-offset:2px;border-radius:2px;}
code,.num{font-family:var(--mono);font-size:0.85em;font-variant-numeric:tabular-nums;}
pre{background:var(--sunken);border:1px solid var(--rule);border-radius:4px;padding:0.8rem 0.95rem;overflow-x:auto;font-size:0.78rem;line-height:1.55;}

/* ------------------------------ hero ------------------------------ */
.hero{position:relative;min-height:100svh;display:grid;grid-template-rows:1fr auto;overflow:hidden;border-bottom:1px solid var(--rule);}
#bifurcation{position:absolute;inset:-8% 0 -8% 0;width:100%;height:116%;display:block;will-change:transform;}
.hero::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(100deg,var(--ground) 0%,color-mix(in oklab,var(--ground) 92%,transparent) 38%,color-mix(in oklab,var(--ground) 22%,transparent) 68%,transparent 100%);}
.hero-inner{position:relative;z-index:2;align-self:center;width:100%;max-width:min(112rem,96vw);
  margin:0 auto;padding:clamp(2rem,8vh,6rem) var(--gutter);}
.hero-inner > *{max-width:min(52rem,90%);}
.hero-rise{opacity:0;transform:translateY(18px);animation:rise .85s cubic-bezier(.22,.68,.28,1) forwards;}
@keyframes rise{to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.hero-rise{opacity:1;transform:none;animation:none;}}
.byline{margin:1.6em 0 0;font-size:1rem;color:var(--ink);}
.byline .affil{display:block;color:var(--ink-low);font-size:0.86rem;margin-top:0.25em;max-width:44ch;}
.lede{font-size:clamp(1.06rem,0.9rem+0.62vw,1.4rem);line-height:1.55;color:var(--ink-mid);max-width:48ch;margin-top:1.5em;}
.stats{display:flex;flex-wrap:wrap;gap:2.25rem;margin-top:2.4rem;padding:0;list-style:none;}
.stats li{margin:0;}
.stats b{display:block;font-size:clamp(1.6rem,1.1rem+1.5vw,2.6rem);line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;}
.stats span{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-low);}
.legend{position:relative;z-index:2;display:flex;flex-wrap:wrap;gap:0.3rem clamp(1rem,2.5vw,2.4rem);
  padding:0.9rem var(--gutter) 1.5rem;max-width:min(112rem,96vw);margin:0 auto;width:100%;font-family:var(--mono);font-size:0.66rem;letter-spacing:0.04em;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
.legend b{font-weight:400;color:var(--ink);}
.swatch{display:inline-block;width:0.62rem;height:0.62rem;margin-right:0.4rem;vertical-align:-1px;border-radius:2px;}

/* ------------------------------ controls ------------------------------ */
.controls{
  position:fixed;top:0;right:0;z-index:60;display:flex;align-items:center;gap:0.15rem;
  padding:0.4rem 0.55rem;background:color-mix(in oklab,var(--ground) 88%,transparent);
  backdrop-filter:blur(10px);border-left:1px solid var(--rule);border-bottom:1px solid var(--rule);
  border-bottom-left-radius:8px;
}
.controls button{
  appearance:none;background:none;border:1px solid transparent;border-radius:5px;
  color:var(--ink-mid);font-family:var(--mono);font-size:0.66rem;letter-spacing:0.06em;
  padding:0.34rem 0.5rem;cursor:pointer;line-height:1;transition:color .15s,background .15s,border-color .15s;
}
.controls button:hover{color:var(--ink);background:var(--sunken);}
.controls button[aria-pressed="true"]{color:var(--chaotic);border-color:color-mix(in oklab,var(--chaotic) 40%,transparent);}
.controls .sep{width:1px;height:1.1rem;background:var(--rule);margin:0 0.25rem;}
@media print{.controls,.progress{display:none;}}

.progress{position:fixed;top:0;left:0;height:2px;background:var(--chaotic);z-index:61;width:0;transition:width .1s linear;}

/* ------------------------------ layout ------------------------------ */
.shell{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);gap:clamp(1.5rem,3.5vw,4rem);
  max-width:min(112rem,96vw);margin:0 auto;padding:clamp(2rem,4vw,4rem) var(--gutter) 6rem;align-items:start;
  transition:grid-template-columns .45s cubic-bezier(.22,.68,.28,1),max-width .45s cubic-bezier(.22,.68,.28,1);}
:root[data-rail="off"] .shell{grid-template-columns:0 minmax(0,1fr);max-width:min(96rem,94vw);}
:root[data-rail="off"] .spine{opacity:0;pointer-events:none;overflow:hidden;}

.spine{position:sticky;top:3.6rem;max-height:calc(100svh - 5.5rem);overflow-y:auto;overscroll-behavior:contain;
  font-family:var(--sans);font-size:0.775rem;line-height:1.35;scrollbar-width:thin;
  scrollbar-color:var(--rule) transparent;padding-right:0.4rem;}
.spine::-webkit-scrollbar{width:6px;}
.spine::-webkit-scrollbar-thumb{background:var(--rule);border-radius:3px;}
.toc,.toc ul{list-style:none;margin:0;padding:0;}
.toc > li.top{margin-bottom:0.1rem;}
.toc a{display:block;padding:0.26rem 0 0.26rem 0.75rem;border-left:1.5px solid var(--rule-soft);
  color:var(--ink-low);text-decoration:none;border-bottom:none;
  transition:color .16s ease,border-color .16s ease;}
.toc a:hover{color:var(--ink);border-left-color:var(--ink-low);}
.toc a i{font-style:normal;font-variant-numeric:tabular-nums;color:var(--ink-low);
  margin-right:0.4em;font-size:0.92em;letter-spacing:0.01em;}
.toc > li.top > a{color:var(--ink-mid);font-weight:700;letter-spacing:-0.005em;}
.toc > li.top > a.on{color:var(--ink);border-left-color:var(--chaotic);}
.toc > li.top > a.on i{color:var(--chaotic);}
.toc .sub{max-height:0;overflow:hidden;opacity:0;transition:max-height .3s ease,opacity .2s ease;}
.toc > li.top.open > .sub{max-height:40rem;opacity:1;}
.toc .sub a{padding-left:1.55rem;font-size:0.735rem;}
.toc .sub a.on{color:var(--ink);border-left-color:var(--chaotic);}
@media (max-width:66rem){.shell{grid-template-columns:minmax(0,1fr);}.spine{display:none;}}

.secno{color:var(--ink-low);font-variant-numeric:tabular-nums;font-weight:400;margin-right:0.5em;letter-spacing:0.01em;}
h2 .secno{font-size:0.78em;}
h3 .secno{font-size:0.85em;}

details.backmatter{margin:0;}
details.backmatter > summary{list-style:none;cursor:pointer;display:block;}
details.backmatter > summary::-webkit-details-marker{display:none;}
details.backmatter > summary h2{display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.9em;}
details.backmatter > summary h2::after{
  content:"show";font-family:var(--mono);font-size:0.58rem;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--ink-low);border:1px solid var(--rule);
  border-radius:4px;padding:0.18rem 0.42rem;margin-left:auto;flex:none;transition:all .15s;}
details.backmatter[open] > summary h2::after{content:"hide";}
details.backmatter > summary:hover h2::after{color:var(--chaotic);border-color:var(--chaotic);}

article{max-width:var(--measure);margin:0 auto;}
article section{scroll-margin-top:3.5rem;}
article > section > h2,
article > section > details > summary > h2{border-top:1px solid var(--rule);padding-top:1.4rem;margin-top:2.8rem;}
article > section:first-child > h2{margin-top:0;border-top:none;padding-top:0;}

/* ------------------------------ figures ------------------------------ */
figure{margin:2.6rem 0;width:var(--bleed);max-width:calc(100vw - 2*var(--gutter));
  position:relative;left:50%;transform:translateX(-50%);background:var(--raised);border:1px solid var(--rule);border-radius:6px;overflow:hidden;
  box-shadow:var(--shadow);transition:box-shadow .3s ease,border-color .3s ease,transform .3s ease;}
figure:hover{border-color:color-mix(in oklab,var(--chaotic) 30%,var(--rule));
  box-shadow:0 2px 4px rgba(20,22,28,.05),0 18px 50px rgba(20,22,28,.12);
  transform:translateX(-50%) translateY(-2px);}
@media (prefers-reduced-motion:reduce){figure,figure:hover{transition:none;transform:translateX(-50%);}}
.fig-head{display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.7rem;border-bottom:1px solid var(--rule-soft);}
.fig-head .name{font-family:var(--mono);font-size:0.63rem;color:var(--ink-low);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.fig-head .acts{margin-left:auto;display:flex;gap:0.25rem;flex:none;}
.fig-head button{appearance:none;background:none;border:1px solid var(--rule);border-radius:4px;color:var(--ink-mid);
  font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.24rem 0.45rem;cursor:pointer;transition:all .15s;}
.fig-head button:hover{color:var(--chaotic);border-color:var(--chaotic);}
.fig-body{position:relative;background:var(--raised);}
figure img{display:block;width:100%;height:auto;max-height:clamp(14rem,34vh,26rem);object-fit:contain;object-position:center;
  cursor:zoom-in;padding:0.5rem;transition:max-height .35s ease;}
:root[data-figsize="tall"] figure img{max-height:min(78vh,46rem);}
figcaption{padding:0.6rem 0.8rem 0.75rem;font-size:0.82rem;line-height:1.5;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
figcaption .num{color:var(--ink);font-weight:700;}
.plot-wrap{position:relative;padding:0.35rem;}
.plot-title{margin:0 0 0.15rem;text-align:center;font-family:var(--mono);font-size:0.62rem;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-low);}
canvas.plot{width:100%;display:block;touch-action:pan-y;}
.hint{margin:0;padding:0 0.8rem 0.6rem;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.06em;color:var(--ink-low);}
/* research-arc overview: a timeline, not a table */
figure.arc{background:var(--raised);}
.arc-track{list-style:none;margin:0;padding:1.4rem 1rem 1.2rem;display:grid;
  grid-template-columns:repeat(6,minmax(0,1fr));gap:0;position:relative;}
.arc-track::before{content:"";position:absolute;left:1.6rem;right:1.6rem;top:2.55rem;height:1.5px;
  background:linear-gradient(90deg,var(--locked),var(--torus),var(--chaotic));opacity:.45;}
.arc-track li{display:flex;flex-direction:column;gap:0.18rem;padding:0 0.45rem;margin:0;
  text-align:left;position:relative;}
.arc-track li::before{content:"";position:absolute;left:0.45rem;top:1.72rem;width:7px;height:7px;
  border-radius:50%;border:1.5px solid var(--raised);z-index:1;
  background:color-mix(in oklab,var(--locked) calc((5 - var(--i)) * 20%),var(--chaotic));}
.arc-track .yr{font-family:var(--mono);font-size:0.63rem;letter-spacing:0.1em;color:var(--ink-low);}
.arc-track .topic{margin-top:1.35rem;font-weight:700;font-size:0.9rem;line-height:1.2;}
.arc-track .mech{font-size:0.78rem;color:var(--ink-mid);line-height:1.25;}
@media (max-width:44rem){
  .arc-track{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.1rem 0;}
  .arc-track::before{display:none;}
  .arc-track li::before{display:none;}
  .arc-track .topic{margin-top:0.15rem;}
}

.readout{position:absolute;pointer-events:none;z-index:4;background:var(--sunken);border:1px solid var(--rule);
  border-radius:4px;padding:0.3rem 0.45rem;font-family:var(--mono);font-size:0.66rem;line-height:1.45;
  color:var(--ink);white-space:pre;opacity:0;transition:opacity .1s;box-shadow:var(--shadow);}
.readout.on{opacity:1;}

/* ------------------------------ tables + refs ------------------------------ */
.table-wrap{overflow-x:auto;margin:2.2rem 0;border:1px solid var(--rule);border-radius:6px;
  background:var(--raised);box-shadow:var(--shadow);
  width:var(--bleed);max-width:calc(100vw - 2*var(--gutter));position:relative;left:50%;transform:translateX(-50%);}
table{border-collapse:collapse;width:100%;font-size:0.9rem;line-height:1.45;
  font-variant-numeric:tabular-nums;}
caption{caption-side:top;text-align:left;padding:0.85rem 1rem 0.75rem;
  font-size:0.86rem;line-height:1.45;color:var(--ink-mid);border-bottom:1px solid var(--rule);}
caption .num{color:var(--ink);font-weight:700;}
thead th{
  position:sticky;top:0;z-index:1;
  background:var(--sunken);
  font-family:var(--sans);font-size:0.7rem;font-weight:700;
  letter-spacing:0.07em;text-transform:uppercase;color:var(--ink-mid);
  padding:0.6rem 1rem;white-space:nowrap;
  border-bottom:1px solid var(--rule);}
tbody td{padding:0.52rem 1rem;border-bottom:1px solid var(--rule-soft);vertical-align:baseline;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover td{background:color-mix(in oklab,var(--chaotic) 5%,transparent);}
/* first column of a reference table is the symbol being defined */
tbody td:first-child{font-weight:700;white-space:nowrap;}
tbody td math{font-size:1em;}
table th[style],table td[style]{text-align:left !important;}

/* 144 citations: they must be reachable without shouting over the prose */
.citation{white-space:nowrap;}
.citation a,a[role="doc-biblioref"]{
  color:inherit;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 30%,transparent);
  transition:color .15s ease,border-color .15s ease;}
.citation a:hover,a[role="doc-biblioref"]:hover{color:var(--chaotic);border-bottom-color:currentColor;}
/* the entry you jumped to should be findable when you land on it */
.csl-entry:target{background:color-mix(in oklab,var(--chaotic) 9%,transparent);
  border-radius:4px;box-shadow:0 0 0 0.5rem color-mix(in oklab,var(--chaotic) 9%,transparent);}
.csl-entry a{color:var(--ink-mid);border-bottom-color:color-mix(in oklab,var(--ink-low) 40%,transparent);
  overflow-wrap:anywhere;}
.csl-entry a:hover{color:var(--chaotic);}
#refs,.references{margin-top:1.2rem;font-size:0.82rem;}
.csl-entry{margin:0 0 0.6em;padding-left:1.5em;text-indent:-1.5em;color:var(--ink-mid);line-height:1.45;}

/* ------------------------------ lightbox ------------------------------ */
.lb{position:fixed;inset:0;z-index:70;display:none;place-items:center;background:color-mix(in oklab,var(--sunken) 94%,transparent);backdrop-filter:blur(8px);padding:2.5rem 1.5rem;}
.lb.on{display:grid;animation:lbin .22s ease forwards;}
@keyframes lbin{from{opacity:0;}to{opacity:1;}}
.lb.on img{animation:lbimg .28s cubic-bezier(.22,.68,.28,1) forwards;}
@keyframes lbimg{from{opacity:0;transform:scale(.975);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.lb.on,.lb.on img{animation:none;}}
.lb img{max-width:100%;max-height:88svh;object-fit:contain;border-radius:3px;box-shadow:0 24px 70px rgba(0,0,0,.45);}
.lb-cap{position:absolute;left:0;right:0;bottom:0;padding:0.9rem 1.5rem;font-size:0.8rem;color:var(--ink-mid);text-align:center;max-width:60rem;margin:0 auto;}
.lb-close{position:absolute;top:0.9rem;right:1.1rem;background:var(--raised);color:var(--ink);border:1px solid var(--rule);border-radius:5px;font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.4rem 0.65rem;cursor:pointer;}
.lb-nav{
  position:absolute;top:50%;transform:translateY(-50%);
  appearance:none;background:var(--raised);color:var(--ink);
  border:1px solid var(--rule);border-radius:5px;
  font-family:var(--mono);font-size:1rem;line-height:1;
  width:2.2rem;height:2.2rem;padding:0;cursor:pointer;
  opacity:.72;transition:opacity .15s,border-color .15s,color .15s;
}
.lb-nav:hover{opacity:1;border-color:var(--chaotic);color:var(--chaotic);}
.lb-prev{left:0.9rem;}
.lb-next{right:0.9rem;}
/* A position in the sequence rather than a figure number. Kept visually apart
   from the caption so neither can be read as the other. */
.lb-pos{position:absolute;top:1rem;left:50%;transform:translateX(-50%);margin:0;
  font-family:var(--mono);font-size:0.68rem;letter-spacing:0.12em;
  color:var(--ink-low);pointer-events:none;}

/* ------------------------------ xref popover + return pill ------------------------------ */
.xref-pop{
  position:fixed;z-index:65;max-width:min(32rem,calc(100vw - 2rem));
  background:var(--raised);color:var(--ink);border:1px solid var(--rule);
  border-radius:6px;box-shadow:var(--shadow);padding:0.7rem 0.85rem;
  font-size:0.82rem;line-height:1.45;
  opacity:0;transform:scale(.98) translateY(4px);visibility:hidden;
  transition:opacity .12s ease,transform .12s ease,visibility .12s;
  pointer-events:none;
}
.xref-pop.on{opacity:1;transform:none;visibility:visible;pointer-events:auto;}
.xref-pop .xref-body{max-height:min(40vh,22rem);overflow:auto;}
.xref-pop .xref-body .csl-entry{margin:0;padding-left:0;text-indent:0;color:var(--ink-mid);}
.xref-pop .xref-body .eqn{margin:0.35em 0;}
.xref-pop .xref-label{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--ink-low);margin:0 0 0.35rem;}
.xref-pop .xref-figimg{display:block;max-width:100%;height:auto;max-height:12rem;
  object-fit:contain;margin:0 0 0.4rem;border-radius:3px;}
.xref-pop .xref-cap{color:var(--ink-mid);font-size:0.78rem;margin:0;line-height:1.45;}
.xref-pop .xref-jump{display:inline-block;margin-top:0.55rem;padding-top:0.45rem;
  border-top:1px solid var(--rule-soft);font-family:var(--mono);font-size:0.66rem;
  letter-spacing:0.04em;color:var(--chaotic);border-bottom:none;text-decoration:none;}
.xref-pop .xref-jump:hover{color:var(--ink);}

.return-pill{
  position:fixed;left:1rem;bottom:1.2rem;z-index:55;
  appearance:none;background:var(--raised);color:var(--ink);
  border:1px solid var(--rule);border-radius:999px;box-shadow:var(--shadow);
  font-family:var(--mono);font-size:0.7rem;letter-spacing:0.02em;
  padding:0.55rem 0.95rem;cursor:pointer;line-height:1;
  opacity:0;transform:translateY(8px);visibility:hidden;pointer-events:none;
  transition:opacity .18s ease,transform .18s ease,visibility .18s;
}
.return-pill.on{opacity:1;transform:none;visibility:visible;pointer-events:auto;}
.return-pill:hover{border-color:var(--chaotic);color:var(--chaotic);}

.cited-in{
  margin:0.2em 0 0.65em;padding-left:1.5em;text-indent:0;
  font-family:var(--mono);font-size:0.72em;color:var(--ink-low);line-height:1.4;
}
.cited-in a{color:var(--ink-low);border-bottom-color:color-mix(in oklab,var(--ink-low) 35%,transparent);}
.cited-in a:hover{color:var(--chaotic);}

html.js .reveal{opacity:0;transform:translateY(14px);transition:opacity .6s cubic-bezier(.22,.68,.28,1),transform .6s cubic-bezier(.22,.68,.28,1);}
html.js figure{opacity:0;transform:translateX(-50%) translateY(22px) scale(.985);}
html.js figure.seen{opacity:1;transform:translateX(-50%) translateY(0) scale(1);
  transition:opacity .7s cubic-bezier(.22,.68,.28,1),transform .7s cubic-bezier(.22,.68,.28,1),box-shadow .3s ease,border-color .3s ease;}
@media (prefers-reduced-motion:reduce){html.js figure{opacity:1;transform:translateX(-50%);}}
html.js .reveal.in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){
  html.js .reveal{opacity:1;transform:none;transition:none;}
  html{scroll-behavior:auto;}.progress{transition:none;}
  .xref-pop,.return-pill{transition:none;transform:none;}
}

@media print{
  :root{--ground:#fff;--raised:#fff;--ink:#000;--ink-mid:#333;--ink-low:#666;--rule:#bbb;--rule-soft:#ddd;}
  .hero{min-height:auto;border-bottom:2px solid #000;page-break-after:avoid;}
  #bifurcation,.hero::after,.legend,.spine,.controls,.progress,.lb,.fig-head .acts,
  .xref-pop,.return-pill,.lb-nav,.lb-pos{display:none !important;}
  .shell{display:block;max-width:none;padding:0;}
  article{max-width:none;}
  figure{break-inside:avoid;box-shadow:none;border:1px solid #bbb;}
  figure img{max-height:none;}
  details.backmatter > summary h2::after{display:none;}
  details.backmatter{display:block;}
  details.backmatter > summary{list-style:none;}
  h2,h3{page-break-after:avoid;}
  a{color:#000;border-bottom:none;}
  .foot{border-top:1px solid #bbb;}
}

.foot{border-top:1px solid var(--rule);padding:2.2rem var(--gutter);max-width:72rem;margin:0 auto;color:var(--ink-mid);font-size:0.83rem;}
"""
)


JS = r"""
"use strict";
document.documentElement.classList.add("js");
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
const root=document.documentElement;
const store=(k,v)=>{try{v===null?localStorage.removeItem(k):localStorage.setItem(k,v);}catch(e){}};
const load=k=>{try{return localStorage.getItem(k);}catch(e){return null;}};

/* ---------------- reading controls ---------------- */
(function(){
  const SIZES=["s","m","l","xl"];
  let size=load("dc-size")||"l";
  let theme=load("dc-theme")||"auto";
  let rail=load("dc-rail")||"on";
  let fig=load("dc-fig")||"short";

  const apply=()=>{
    root.setAttribute("data-size",size);
    if(theme==="auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme",theme);
    root.setAttribute("data-rail",rail);
    root.setAttribute("data-figsize",fig);
  };
  apply();

  const bar=document.querySelector(".controls");
  if(!bar) return;
  const $=s=>bar.querySelector(s);
  const sync=()=>{
    $(".c-theme").textContent=theme==="auto"?"auto":theme;
    $(".c-theme").setAttribute("aria-pressed",theme!=="auto");
    $(".c-rail").setAttribute("aria-pressed",rail==="on");
    $(".c-fig").setAttribute("aria-pressed",fig==="tall");
    $(".c-smaller").disabled=size==="s";
    $(".c-larger").disabled=size==="xl";
  };
  $(".c-theme").addEventListener("click",()=>{
    theme=theme==="auto"?"light":theme==="light"?"dark":"auto";
    store("dc-theme",theme==="auto"?null:theme);apply();sync();repaint();
  });
  $(".c-rail").addEventListener("click",()=>{
    rail=rail==="on"?"off":"on";store("dc-rail",rail);apply();sync();
    setTimeout(()=>MOUNTED.forEach(p=>p.resize()),60);
  });
  $(".c-fig").addEventListener("click",()=>{
    fig=fig==="short"?"tall":"short";store("dc-fig",fig);apply();sync();
  });
  const step=d=>{
    const i=Math.max(0,Math.min(SIZES.length-1,SIZES.indexOf(size)+d));
    size=SIZES[i];store("dc-size",size);apply();sync();
    setTimeout(()=>MOUNTED.forEach(p=>p.resize()),60);
  };
  $(".c-smaller").addEventListener("click",()=>step(-1));
  $(".c-larger").addEventListener("click",()=>step(1));
  sync();
})();

/* ---------------- reading progress ---------------- */
(function(){
  const bar=document.querySelector(".progress");
  const art=document.querySelector("article");
  if(!bar||!art) return;
  const upd=()=>{
    const top=art.offsetTop, h=art.offsetHeight-innerHeight;
    const p=h>0?Math.min(1,Math.max(0,(scrollY-top)/h)):0;
    bar.style.width=(p*100).toFixed(2)+"%";
  };
  addEventListener("scroll",upd,{passive:true});
  addEventListener("resize",upd);upd();
})();

/* ---------------- hero: logistic attractor coloured by sign of lambda ----------------
   After the initial sweep the map keeps iterating: random columns gain a few
   more points every frame while a faint wash of the ground colour holds the
   density at equilibrium. The shimmer is the computation continuing, not an
   effect layered on top of it. */
function hero(){
  const cv=document.getElementById("bifurcation");
  if(!cv) return ()=>{};
  const ctx=cv.getContext("2d",{alpha:false});
  let raf=null,col=0,W=0,H=0,live=false,tick=0;

  function column(px,iters,alpha){
    const r=2.85+(4.0-2.85)*(px/W);
    let x=0.35+0.3*((px*2654435761)%1000)/1000,l=0;
    for(let i=0;i<380;i++) x=r*x*(1-x);
    for(let i=0;i<160;i++){x=r*x*(1-x);l+=Math.log(Math.abs(r*(1-2*x))+1e-12);}
    l/=160;
    ctx.fillStyle=l>0.005?css("--chaotic"):l<-0.005?css("--locked"):css("--torus");
    ctx.globalAlpha=alpha;
    for(let i=0;i<iters;i++){x=r*x*(1-x);ctx.fillRect(px,(1-x)*H,1.15,1.15);}
    ctx.globalAlpha=1;
  }

  function sweep(target){
    const end=Math.min(W,target||col+22);
    for(;col<end;col++){
      ctx.globalAlpha=0.10;ctx.fillStyle=css('--ground');ctx.fillRect(col,0,1.6,H);ctx.globalAlpha=1;
      column(col,330,0.46);
    }
    if(col<W){raf=requestAnimationFrame(()=>sweep(0));}
    else if(!reduced){live=true;raf=requestAnimationFrame(shimmer);}
  }

  function shimmer(){
    tick++;
    // hold the density steady so the plate never saturates to a solid block
    if(tick%3===0){
      ctx.globalAlpha=0.016;ctx.fillStyle=css("--ground");
      ctx.fillRect(0,0,W,H);ctx.globalAlpha=1;
    }
    for(let k=0;k<10;k++) column(Math.floor(Math.random()*W),70,0.16);
    if(live) raf=requestAnimationFrame(shimmer);
  }

  // Resizing used to clear to the ground colour and redraw, which flashed.
  // Stretch the previous frame into the new size first so the plate deforms
  // continuously, then refine it column by column underneath.
  function reset(carry){
    const dpr=Math.min(devicePixelRatio||1,2);
    const nw=Math.round(cv.clientWidth*dpr),nh=Math.round(cv.clientHeight*dpr);
    if(!nw||!nh||(nw===W&&nh===H&&carry!==undefined)) return;
    let prev=null;
    if(carry&&W&&H){
      prev=document.createElement("canvas");prev.width=W;prev.height=H;
      prev.getContext("2d").drawImage(cv,0,0);
    }
    W=nw;H=nh;cv.width=W;cv.height=H;
    ctx.fillStyle=css("--ground");ctx.fillRect(0,0,W,H);
    if(prev){ctx.globalAlpha=0.85;ctx.drawImage(prev,0,0,W,H);ctx.globalAlpha=1;}
    col=0;live=false;
    if(raf)cancelAnimationFrame(raf);
    sweep(reduced?W:0);
  }

  // pause when off-screen; an attractor nobody can see should not burn a core
  new IntersectionObserver(es=>{
    for(const e of es){
      if(e.isIntersecting){ if(!live&&col>=W&&!reduced){live=true;raf=requestAnimationFrame(shimmer);} }
      else { live=false; if(raf)cancelAnimationFrame(raf); }
    }
  },{threshold:0.01}).observe(cv);

  // slow parallax: the plate drifts against the type as the reader leaves
  if(!reduced){
    addEventListener("scroll",()=>{
      const y=Math.min(scrollY,innerHeight);
      cv.style.transform="translate3d(0,"+(y*0.16).toFixed(1)+"px,0)";
    },{passive:true});
  }

  let rrq=false;
  addEventListener("resize",()=>{
    if(rrq) return;
    rrq=true;
    requestAnimationFrame(()=>{rrq=false;reset(true);});
  });
  reset();
  return ()=>reset(false);
}

/* ---------------- canvas plotting ---------------- */
// Tick text is formatted against the axis *span*, not the value: an axis
// running 1.4..2.5 wants one decimal, not the four that "1.4000" wasted.
function fmtSpan(v,span){
  const a=Math.abs(v);
  if((a!==0&&(a<1e-3||a>=1e5))||(span&&span<1e-3))return v.toExponential(2);
  const d=span?Math.max(0,Math.min(6,1-Math.floor(Math.log10(span/4)))):3;
  const s=v.toFixed(d);
  return s==="-0"?"0":s;
}
function fmt(v){return fmtSpan(v,Math.abs(v)||1);}
function ticks(lo,hi,n){
  const raw=(hi-lo||1)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),nm=raw/mag;
  const st=(nm<1.5?1:nm<3?2:nm<7?5:10)*mag,out=[];
  for(let t=Math.ceil(lo/st)*st;t<=hi+1e-9;t+=st) out.push(t);
  return out;}
const PAL=["--chaotic","--locked","--torus"];
const MOUNTED=[];

function Plot(canvas,panel,meta){
  const ctx=canvas.getContext("2d");
  const multi=panel.traces&&panel.traces.length>1;
  const pad={l:58,r:12,t:multi?24:10,b:34};
  const wrap=canvas.parentElement;
  const tip=document.createElement("div");tip.className="readout";wrap.appendChild(tip);
  let W=0,H=0,dom=null,base=null,drag=null;
  const heat=meta.kind==="heatmap";

  function extent(){
    if(heat) return {x0:Math.min(...panel.x),x1:Math.max(...panel.x),y0:Math.min(...panel.y),y1:Math.max(...panel.y)};
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    for(const s of panel.traces) for(let i=0;i<s.x.length;i++){
      if(s.x[i]<x0)x0=s.x[i]; if(s.x[i]>x1)x1=s.x[i];
      const lo=s.yerr?s.y[i]-s.yerr[i]:s.y[i], hi=s.yerr?s.y[i]+s.yerr[i]:s.y[i];
      if(lo<y0)y0=lo; if(hi>y1)y1=hi;
    }
    const p=(y1-y0)*0.08||0.5;
    y0-=p;y1+=p;
    // Reference lines widen the axis, exactly as matplotlib's axvline/axhline
    // do -- otherwise a threshold the caption depends on falls off the edge.
    for(const m of (meta.marks||[])){
      if(m.axis==="x"){x0=Math.min(x0,m.value);x1=Math.max(x1,m.value);}
      else{y0=Math.min(y0,m.value);y1=Math.max(y1,m.value);}
    }
    return {x0,x1,y0,y1};
  }
  const sx=v=>pad.l+(v-dom.x0)/(dom.x1-dom.x0)*(W-pad.l-pad.r);
  const sy=v=>H-pad.b-(v-dom.y0)/(dom.y1-dom.y0)*(H-pad.t-pad.b);
  const ux=p=>dom.x0+(p-pad.l)/(W-pad.l-pad.r)*(dom.x1-dom.x0);

  let zmin=0,zmax=1;
  if(heat){zmin=Infinity;zmax=-Infinity;
    for(const row of panel.z) for(const v of row){
      if(v===null||Number.isNaN(v))continue;
      if(v<zmin)zmin=v;if(v>zmax)zmax=v;}}
  const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];
  // Same viridis the matplotlib figures use, so toggling to the interactive
  // view never changes what a colour means.
  const VIRIDIS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                 [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
  const signed=(panel.cmap||"viridis")==="signed"&&zmin<0&&zmax>0;
  function lerp3(a,b,f){return [0,1,2].map(j=>Math.round(a[j]+(b[j]-a[j])*f));}
  function rampRGB(t){
    t=Math.max(0,Math.min(1,t));
    if(signed){
      const stops=[hex(css("--locked")),hex(css("--torus")),hex(css("--chaotic"))];
      const k=t*2,i=Math.min(1,Math.floor(k));
      return lerp3(stops[i],stops[i+1],k-i);
    }
    const k=t*(VIRIDIS.length-1),i=Math.min(VIRIDIS.length-2,Math.floor(k));
    return lerp3(VIRIDIS[i],VIRIDIS[i+1],k-i);
  }
  // Two-slope norm, as matplotlib's TwoSlopeNorm: z = 0 sits at the ramp's
  // midpoint (the locked/chaotic boundary) while each side still spans its half
  // of the ramp, so a lopsided field does not collapse into one hue.
  const znorm=v=>signed
    ? (v<0 ? 0.5*(1-v/zmin) : 0.5+0.5*(v/zmax))
    : (v-zmin)/((zmax-zmin)||1);
  const zlo=zmin, zhi=zmax;
  function ramp(t){const c=rampRGB(t);return `rgb(${c[0]},${c[1]},${c[2]})`;}

  const CBAR=13;   // colourbar strip width
  // Reserve exactly the room the tick text needs, so the rotated axis label
  // stops landing on top of it.
  function measurePads(){
    ctx.font="10px "+css("--mono");
    let w=0;
    const yspan=dom.y1-dom.y0;
    for(const t of ticks(dom.y0,dom.y1,4)) w=Math.max(w,ctx.measureText(fmtSpan(t,yspan)).width);
    pad.l=Math.ceil(w)+(meta.ylabel?26:12);
    pad.r=12;
    if(heat){
      let zw=0;
      for(const t of ticks(zlo,zhi,4)) zw=Math.max(zw,ctx.measureText(fmtSpan(t,zhi-zlo)).width);
      pad.r=12+CBAR+8+Math.ceil(zw)+(zlabel?16:4);
    }
  }
  const zlabel=panel.zlabel||"";

  function drawColorbar(){
    const x=W-pad.r+12,y0=pad.t,y1=H-pad.b,h=y1-y0;
    if(h<20) return;
    // Painted linearly in value, not in ramp position, so a two-slope norm
    // still reads against evenly spaced ticks.
    for(let p=0;p<h;p++){
      ctx.fillStyle=ramp(znorm(zlo+(1-p/(h-1))*(zhi-zlo)));
      ctx.fillRect(x,y0+p,CBAR,1);
    }
    ctx.strokeStyle=css("--rule");ctx.lineWidth=1;
    ctx.strokeRect(x+0.5,y0+0.5,CBAR-1,h-1);
    ctx.fillStyle=css("--ink-low");ctx.textAlign="left";ctx.textBaseline="middle";
    const zspan=zhi-zlo;
    for(const t of ticks(zlo,zhi,4)){
      if(t<zlo-1e-12||t>zhi+1e-12)continue;
      const y=y1-(t-zlo)/(zspan||1)*h;
      ctx.fillText(fmtSpan(t,zspan),x+CBAR+5,y);
    }
    if(zlabel){
      ctx.save();ctx.fillStyle=css("--ink-mid");
      ctx.translate(W-3,y0+h/2);ctx.rotate(Math.PI/2);
      ctx.textAlign="center";ctx.textBaseline="top";ctx.fillText(zlabel,0,0);ctx.restore();
    }
  }

  // Reference lines the captions promise (critical lines, bifurcation
  // thresholds). Values come from the payload, never recomputed here.
  function drawMarks(){
    const marks=meta.marks||[];
    if(!marks.length) return;
    ctx.save();ctx.setLineDash([5,4]);ctx.lineWidth=1;
    ctx.font="10px "+css("--mono");ctx.textBaseline="bottom";
    for(const m of marks){
      const vertical=m.axis==="x";
      const p=vertical?sx(m.value):sy(m.value);
      if(vertical?(p<pad.l||p>W-pad.r):(p<pad.t||p>H-pad.b))continue;
      ctx.strokeStyle=css("--ink-mid");ctx.globalAlpha=.75;
      ctx.beginPath();
      if(vertical){ctx.moveTo(p+0.5,pad.t);ctx.lineTo(p+0.5,H-pad.b);}
      else{ctx.moveTo(pad.l,p+0.5);ctx.lineTo(W-pad.r,p+0.5);}
      ctx.stroke();
      if(m.label){
        ctx.globalAlpha=1;ctx.fillStyle=css("--ink-mid");
        if(vertical){ctx.textAlign="left";ctx.fillText(m.label,p+4,pad.t+11);}
        else{ctx.textAlign="right";ctx.fillText(m.label,W-pad.r-4,p-3);}
      }
    }
    ctx.restore();
  }

  function draw(){
    if(!W||!H) return;
    measurePads();
    ctx.clearRect(0,0,W,H);
    const rule=css("--rule"),low=css("--ink-low"),mid=css("--ink-mid");
    ctx.font="10px "+css("--mono");ctx.strokeStyle=rule;ctx.lineWidth=1;ctx.fillStyle=low;
    ctx.textAlign="right";ctx.textBaseline="middle";
    const yspan=dom.y1-dom.y0,xspan=dom.x1-dom.x0;
    for(const t of ticks(dom.y0,dom.y1,4)){
      const y=Math.round(sy(t))+0.5;if(y<pad.t-1||y>H-pad.b+1)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmtSpan(t,yspan),pad.l-7,y);
    }
    ctx.textAlign="center";ctx.textBaseline="top";
    for(const t of ticks(dom.x0,dom.x1,5)){
      const x=Math.round(sx(t))+0.5;if(x<pad.l-1||x>W-pad.r+1)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmtSpan(t,xspan),x,H-pad.b+7);
    }
    ctx.fillStyle=mid;ctx.textAlign="center";ctx.textBaseline="bottom";
    ctx.fillText(meta.xlabel||"",pad.l+(W-pad.l-pad.r)/2,H-2);
    ctx.save();ctx.translate(11,pad.t+(H-pad.t-pad.b)/2);ctx.rotate(-Math.PI/2);
    ctx.textBaseline="top";ctx.fillText(meta.ylabel||"",0,0);ctx.restore();

    ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b);ctx.clip();
    if(heat){
      const nx=panel.x.length,ny=panel.y.length;
      // Fill to the next sample's edge rather than a fixed +1 fudge, so cells
      // tile exactly instead of smearing over one another.
      const edgeX=i=>i<nx-1?(sx(panel.x[i])+sx(panel.x[i+1]))/2:sx(panel.x[i])+(sx(panel.x[i])-sx(panel.x[i-1]))/2;
      const edgeY=j=>j<ny-1?(sy(panel.y[j])+sy(panel.y[j+1]))/2:sy(panel.y[j])+(sy(panel.y[j])-sy(panel.y[j-1]))/2;
      for(let j=0;j<ny;j++){
        const yb=j>0?edgeY(j-1):sy(panel.y[0])-(edgeY(0)-sy(panel.y[0]));
        const ya=edgeY(j);
        const yTop=Math.min(ya,yb),yh=Math.abs(ya-yb)+1;
        if(yTop>H-pad.b||yTop+yh<pad.t)continue;
        const row=panel.z[j];
        for(let i=0;i<nx;i++){
          const v=row[i];if(v===null||Number.isNaN(v))continue;
          const xb=i>0?edgeX(i-1):sx(panel.x[0])-(edgeX(0)-sx(panel.x[0]));
          const xa=edgeX(i);
          const xL=Math.min(xa,xb),xw=Math.abs(xa-xb)+1;
          if(xL>W-pad.r||xL+xw<pad.l)continue;
          ctx.fillStyle=ramp(znorm(v));ctx.fillRect(xL,yTop,xw,yh);
        }
      }
    }else{
      panel.traces.forEach((s,k)=>{
        const c=css(s.color||PAL[k%PAL.length]);
        if(s.yerr){
          ctx.fillStyle=c;ctx.globalAlpha=.15;ctx.beginPath();
          for(let i=0;i<s.x.length;i++) ctx.lineTo(sx(s.x[i]),sy(s.y[i]+s.yerr[i]));
          for(let i=s.x.length-1;i>=0;i--) ctx.lineTo(sx(s.x[i]),sy(s.y[i]-s.yerr[i]));
          ctx.closePath();ctx.fill();ctx.globalAlpha=1;
        }
        if(meta.kind==="scatter"){
          ctx.fillStyle=c;ctx.globalAlpha=.72;
          for(let i=0;i<s.x.length;i++) ctx.fillRect(sx(s.x[i])-1,sy(s.y[i])-1,2,2);
          ctx.globalAlpha=1;
        }else{
          ctx.strokeStyle=c;ctx.lineWidth=1.5;ctx.lineJoin="round";ctx.beginPath();
          for(let i=0;i<s.x.length;i++) ctx.lineTo(sx(s.x[i]),sy(s.y[i]));
          ctx.stroke();
        }
      });
    }
    drawMarks();
    if(drag&&drag.cur!==null){
      ctx.fillStyle=css("--chaotic");ctx.globalAlpha=.12;
      const a=Math.min(drag.start,drag.cur),b=Math.max(drag.start,drag.cur);
      ctx.fillRect(a,pad.t,b-a,H-pad.t-pad.b);ctx.globalAlpha=1;
    }
    ctx.restore();
    ctx.strokeStyle=css("--rule");ctx.lineWidth=1;
    ctx.strokeRect(pad.l+0.5,pad.t+0.5,W-pad.l-pad.r-1,H-pad.t-pad.b-1);
    if(heat) drawColorbar();
    if(!heat&&multi){
      ctx.font="10px "+css("--mono");ctx.textAlign="left";ctx.textBaseline="middle";
      let lx=pad.l+4;
      panel.traces.forEach((s,k)=>{
        ctx.fillStyle=css(s.color||PAL[k%PAL.length]);ctx.fillRect(lx,pad.t-10,12,2);
        ctx.fillStyle=mid;ctx.fillText(s.name||"",lx+17,pad.t-9);
        lx+=25+ctx.measureText(s.name||"").width;
      });
    }
  }

  function resize(){
    const dpr=Math.min(devicePixelRatio||1,2);
    const cw=canvas.clientWidth;if(!cw) return;
    const chh=Math.round(Math.min(340,Math.max(200,cw*(heat?0.5:0.42))));
    canvas.style.height=chh+"px";
    canvas.width=cw*dpr;canvas.height=chh*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);W=cw;H=chh;draw();
  }

  canvas.addEventListener("pointermove",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;
    if(drag){drag.cur=Math.max(pad.l,Math.min(W-pad.r,px));draw();return;}
    if(px<pad.l||px>W-pad.r||py<pad.t||py>H-pad.b){tip.classList.remove("on");return;}
    const xv=ux(px);const lines=[];
    if(heat){
      const yv=dom.y0+(H-pad.b-py)/(H-pad.t-pad.b)*(dom.y1-dom.y0);
      const near=(arr,v)=>{let b=0,d=Infinity;
        for(let i=0;i<arr.length;i++){const q=Math.abs(arr[i]-v);if(q<d){d=q;b=i;}}return b;};
      const i=near(panel.x,xv),j=near(panel.y,yv);
      lines.push((meta.xlabel||"x")+" = "+fmt(panel.x[i]));
      lines.push((meta.ylabel||"y")+" = "+fmt(panel.y[j]));
      lines.push((zlabel||"value")+" = "+fmt(panel.z[j][i]));
    }else{
      lines.push((meta.xlabel||"x")+" = "+fmt(xv));
      for(const s of panel.traces){
        let b=0,d=Infinity;
        for(let i=0;i<s.x.length;i++){const q=Math.abs(s.x[i]-xv);if(q<d){d=q;b=i;}}
        lines.push((s.name||"y")+" = "+fmt(s.y[b]));
      }
    }
    tip.textContent=lines.join("\n");tip.classList.add("on");
    const tw=tip.offsetWidth,th=tip.offsetHeight;
    tip.style.left=Math.min(px+12,W-tw-4)+"px";
    tip.style.top=Math.max(4,Math.min(py-th-8,H-th-4))+"px";
  });
  canvas.addEventListener("pointerleave",()=>tip.classList.remove("on"));
  canvas.addEventListener("pointerdown",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left;
    if(px<pad.l||px>W-pad.r)return;
    drag={start:px,cur:null};canvas.setPointerCapture(e.pointerId);tip.classList.remove("on");
  });
  canvas.addEventListener("pointerup",()=>{
    if(drag&&drag.cur!==null&&Math.abs(drag.cur-drag.start)>10){
      const a=ux(Math.min(drag.start,drag.cur)),b=ux(Math.max(drag.start,drag.cur));
      if(heat){dom={...dom,x0:a,x1:b};}
      else{
        const ys=[];
        for(const s of panel.traces) for(let i=0;i<s.x.length;i++)
          if(s.x[i]>=a&&s.x[i]<=b) ys.push(s.y[i]);
        if(ys.length>1){const y0=Math.min(...ys),y1=Math.max(...ys),p=(y1-y0)*0.08||0.5;
          dom={x0:a,x1:b,y0:y0-p,y1:y1+p};}
      }
    }
    drag=null;draw();
  });
  canvas.addEventListener("dblclick",()=>{dom={...base};draw();});

  base=extent();dom={...base};
  new ResizeObserver(resize).observe(canvas);resize();
  return {redraw:draw,resize};
}

/* ---------------- figures: static by default, interaction on demand ---------------- */
async function mountInteractive(fig){
  const body=fig.querySelector(".fig-body");
  const src=fig.dataset.src;
  const btn=fig.querySelector(".act-interact");
  if(fig.dataset.state==="live"){                     // toggle back to the image
    fig.dataset.state="static";
    body.querySelectorAll(".plot-wrap,.hint").forEach(n=>n.remove());
    body.querySelector("img").style.display="";
    btn.textContent="interact";
    return;
  }
  btn.textContent="loading";btn.disabled=true;
  try{
    const spec=await (await fetch(src)).json();
    body.querySelector("img").style.display="none";
    spec.panels.forEach(panel=>{
      const w=document.createElement("div");w.className="plot-wrap";
      const c=document.createElement("canvas");c.className="plot";
      w.appendChild(c);body.appendChild(w);
      if(spec.panels.length>1&&panel.title){
        const h=document.createElement("p");h.className="plot-title";
        h.textContent=panel.title;w.insertBefore(h,c);
      }
      MOUNTED.push(Plot(c,panel,{
        kind:spec.kind,
        marks:spec.marks||[],
        xlabel:(spec.axes&&spec.axes.x&&spec.axes.x.label)||"",
        ylabel:(spec.axes&&spec.axes.y&&spec.axes.y.label)||""}));
    });
    const d=spec.decimation,n=document.createElement("p");n.className="hint";
    n.textContent="drag to zoom · double-click to reset · hover to read values"
      +(d&&d.method!=="none"?" · resampled to "+d.to.toLocaleString()+" of "+d.from.toLocaleString()+" grid points; the static image is at full resolution":"");
    body.appendChild(n);
    fig.dataset.state="live";btn.textContent="image";
  }catch(err){
    btn.textContent="unavailable";
  }finally{btn.disabled=false;}
}

document.querySelectorAll("figure[data-src]").forEach(fig=>{
  const btn=fig.querySelector(".act-interact");
  if(btn) btn.addEventListener("click",()=>mountInteractive(fig));
});

/* ---------------- lightbox + figure nav ---------------- */
(function(){
  const lb=document.querySelector(".lb");if(!lb)return;
  const img=lb.querySelector("img"),cap=lb.querySelector(".lb-cap");
  const figs=[...document.querySelectorAll("figure")].filter(f=>f.querySelector(".fig-body img, img"));
  let idx=-1;
  const prev=document.createElement("button");
  prev.type="button";prev.className="lb-nav lb-prev";
  prev.setAttribute("aria-label","Previous figure");prev.textContent="\u2190";
  const next=document.createElement("button");
  next.type="button";next.className="lb-nav lb-next";
  next.setAttribute("aria-label","Next figure");next.textContent="\u2192";
  const pos=document.createElement("p");
  pos.className="lb-pos";pos.setAttribute("aria-label","Position in the figure sequence");
  lb.appendChild(prev);lb.appendChild(next);lb.appendChild(pos);

  function show(i){
    if(!figs.length) return;
    idx=((i%figs.length)+figs.length)%figs.length;
    const f=figs[idx],t=f.querySelector("img");
    img.src=t.dataset.full||t.src;img.alt=t.alt||"";
    const c=f.querySelector("figcaption");
    // "Figure N of M" would invent a second numbering authority: N is the
    // figure's own number, M the size of the enlargeable set, and the two only
    // agree while the one figure without an image happens to sit last. Name the
    // figure exactly as its caption does, and report the position separately as
    // a position, so neither can be read as the other.
    const n=f.dataset.fignum||"";
    const head=n?("Figure "+n):("Figure "+(idx+1));
    let body=c?c.textContent.trim():"";
    body=body.replace(/^Figure\s*\d+\.\s*/i,"");
    if(body.length>220) body=body.slice(0,217)+"\u2026";
    cap.textContent=body?head+". "+body:head;
    pos.textContent=(idx+1)+" / "+figs.length;
    lb.classList.add("on");
  }
  const open=t=>{
    const f=t.closest("figure");
    const i=f?figs.indexOf(f):-1;
    show(i>=0?i:0);
    lb.querySelector(".lb-close").focus();
  };
  const close=()=>lb.classList.remove("on");
  document.addEventListener("click",e=>{
    const t=e.target;
    if(t.closest&&t.closest(".lb-nav")) return;
    if(t.tagName==="IMG"&&t.closest("figure")&&!t.closest(".xref-pop")){open(t);return;}
    if(t.classList.contains("act-zoom")){
      const i=t.closest("figure").querySelector("img");if(i)open(i);return;}
    if(t.closest(".lb")) close();
  });
  prev.addEventListener("click",e=>{e.stopPropagation();show(idx-1);});
  next.addEventListener("click",e=>{e.stopPropagation();show(idx+1);});
  addEventListener("keydown",e=>{
    if(!lb.classList.contains("on")) return;
    if(e.key==="Escape") close();
    else if(e.key==="ArrowLeft") show(idx-1);
    else if(e.key==="ArrowRight") show(idx+1);
  });
})();

/* ---------------- hover / focus / tap previews ---------------- */
(function(){
  const coarse=matchMedia("(pointer:coarse)").matches;
  const pop=document.createElement("div");
  pop.className="xref-pop";pop.id="xref-pop";pop.setAttribute("role","tooltip");
  pop.hidden=true;document.body.appendChild(pop);
  let cur=null,hideT=null,armed=null;

  /* The equation anchor is an empty span that pandoc emits *inside* the
     paragraph, immediately before the display equation. A <div> cannot live in
     a <p>, so the parser closes the paragraph and hoists the .eqn out: in the
     DOM the anchor is the paragraph's last child and the equation is the
     paragraph's next sibling, one level up. Checking nextElementSibling alone
     finds nothing and the preview comes up blank -- which is how this shipped
     looking correct. Climb until an .eqn turns up, bounded so a stray anchor
     cannot drag in an unrelated equation from further down the section. */
  function eqnAfter(anchor){
    let node=anchor;
    for(let up=0;node&&up<3;up++,node=node.parentElement){
      for(let s=node.nextElementSibling;s;s=s.nextElementSibling){
        if(s.classList&&s.classList.contains("eqn")) return s;
        if(s.querySelector){const inner=s.querySelector(".eqn");if(inner) return inner;}
        if(s.textContent&&s.textContent.trim()) return null;   // real prose intervened
      }
    }
    return null;
  }

  function classify(a){
    const href=a.getAttribute("href")||"";
    if(href[0]!=="#") return null;
    let id;try{id=decodeURIComponent(href.slice(1));}catch(e){id=href.slice(1);}
    if(!id) return null;
    const el=document.getElementById(id);
    if(!el) return null;
    if(a.getAttribute("role")==="doc-biblioref"||el.classList.contains("csl-entry")||id.startsWith("ref-"))
      return {kind:"cite",el,href};
    if(a.dataset.referenceType==="eqref"||el.hasAttribute("data-eqno")||el.classList.contains("eqn")
       ||id.startsWith("eq:")||id.startsWith("eq-u")){
      let content=el;
      if(el.hasAttribute("data-eqno")) content=eqnAfter(el)||el;
      return {kind:"eq",el:content,href};
    }
    if((el.matches&&el.matches("figure"))||id.startsWith("fig:"))
      return {kind:"fig",el:el.matches&&el.matches("figure")?el:(el.closest("figure")||el),href};
    return null;
  }

  function fill(info){
    pop.replaceChildren();
    const body=document.createElement("div");body.className="xref-body";
    if(info.kind==="cite"){
      const c=info.el.cloneNode(true);
      c.removeAttribute("id");
      c.querySelectorAll(".cited-in").forEach(n=>n.remove());
      body.appendChild(c);pop.appendChild(body);
      const jump=document.createElement("a");
      jump.className="xref-jump";jump.href=info.href;jump.textContent="\u2192 jump";
      pop.appendChild(jump);
    }else if(info.kind==="eq"){
      const c=info.el.cloneNode(true);
      if(c.removeAttribute) c.removeAttribute("id");
      body.appendChild(c);pop.appendChild(body);
    }else{
      const lab=document.createElement("p");lab.className="xref-label";
      lab.textContent="Figure "+(info.el.dataset.fignum||"");
      body.appendChild(lab);
      const im=info.el.querySelector("img");
      if(im){
        const c=document.createElement("img");
        c.className="xref-figimg";c.src=im.getAttribute("src")||im.src;c.alt=im.alt||"";
        body.appendChild(c);
      }
      const fc=info.el.querySelector("figcaption");
      if(fc){
        // The caption opens with its own "Figure N." span; keeping it here
        // would print the number twice under the label we just added.
        const c=fc.cloneNode(true);
        c.querySelectorAll(".num").forEach(n=>n.remove());
        const p=document.createElement("p");p.className="xref-cap";
        let t=c.textContent.trim();
        if(t.length>200) t=t.slice(0,197)+"\u2026";
        p.textContent=t;body.appendChild(p);
      }
      pop.appendChild(body);
    }
  }

  function place(a){
    const r=a.getBoundingClientRect();
    const pw=pop.offsetWidth,ph=pop.offsetHeight;
    let top=r.top-ph-10;
    if(top<8) top=Math.min(r.bottom+10,innerHeight-ph-8);
    top=Math.max(8,Math.min(top,innerHeight-ph-8));
    let left=r.left+r.width/2-pw/2;
    left=Math.max(8,Math.min(left,innerWidth-pw-8));
    pop.style.top=top+"px";pop.style.left=left+"px";
  }

  function hide(){
    clearTimeout(hideT);hideT=null;
    pop.classList.remove("on");
    if(cur){cur.removeAttribute("aria-describedby");cur=null;}
    const done=()=>{if(!pop.classList.contains("on")) pop.hidden=true;};
    if(reduced) done(); else setTimeout(done,120);
  }
  function scheduleHide(){clearTimeout(hideT);hideT=setTimeout(hide,120);}
  function cancelHide(){clearTimeout(hideT);hideT=null;}

  function open(a,info){
    cancelHide();
    fill(info);
    pop.hidden=false;
    if(cur&&cur!==a) cur.removeAttribute("aria-describedby");
    cur=a;a.setAttribute("aria-describedby","xref-pop");
    // measure then fade in
    pop.style.left="0";pop.style.top="0";
    place(a);
    if(reduced) pop.classList.add("on");
    else requestAnimationFrame(()=>{place(a);pop.classList.add("on");});
  }

  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    if(!classify(a)) return;
    a.addEventListener("pointerenter",()=>{if(!coarse) open(a,classify(a));});
    a.addEventListener("pointerleave",()=>{if(!coarse) scheduleHide();});
    a.addEventListener("focus",()=>open(a,classify(a)));
    a.addEventListener("blur",scheduleHide);
  });
  pop.addEventListener("pointerenter",cancelHide);
  pop.addEventListener("pointerleave",scheduleHide);

  // first tap on coarse pointers opens the preview instead of navigating
  document.addEventListener("click",e=>{
    const a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(!a||a.closest(".xref-pop")) return;
    const info=classify(a);
    if(!info){armed=null;return;}
    if(coarse){
      if(armed!==a){
        e.preventDefault();
        armed=a;open(a,info);return;
      }
      armed=null;hide();
    }
  },true);

  addEventListener("keydown",e=>{if(e.key==="Escape") hide();});
  addEventListener("scroll",()=>{if(cur) hide();},{passive:true});
  addEventListener("resize",()=>{if(cur) hide();});
})();

/* ---------------- return pill ---------------- */
(function(){
  const pill=document.createElement("button");
  pill.type="button";pill.className="return-pill";pill.hidden=true;
  document.body.appendChild(pill);
  let origin=null,landY=null,hideTimer=null,landTimer=null;

  function currentSecno(){
    const on=document.querySelector(".spine a.on i");
    if(on&&on.textContent.trim()) return on.textContent.trim();
    const secs=[...document.querySelectorAll("article section[id]")];
    let cur=null;
    for(const s of secs){
      if(s.getBoundingClientRect().top<=innerHeight*0.35) cur=s;
    }
    if(!cur) return "";
    const h=cur.querySelector("h2, h3");
    const sn=h&&h.querySelector(".secno");
    return sn?sn.textContent.trim():"";
  }

  function hide(){
    clearTimeout(hideTimer);hideTimer=null;
    clearTimeout(landTimer);landTimer=null;
    origin=null;landY=null;
    pill.classList.remove("on");
    const done=()=>{if(!pill.classList.contains("on")) pill.hidden=true;};
    if(reduced) done(); else setTimeout(done,180);
  }
  function show(y,sec){
    origin={y,sec};landY=null;
    pill.textContent=sec?("\u21A9 back to \u00A7"+sec):"\u21A9 back";
    pill.hidden=false;
    // Reading offsetWidth flushes layout, which gives the transition the frame
    // boundary it needs. requestAnimationFrame would do the same but does not
    // run in a hidden tab -- a reader who jumps and immediately switches tabs
    // would come back to a pill that never appeared and never armed its timeout.
    void pill.offsetWidth;
    pill.classList.add("on");
    clearTimeout(hideTimer);
    hideTimer=setTimeout(hide,45000);
    // snapshot landing position after smooth-scroll settles
    clearTimeout(landTimer);
    landTimer=setTimeout(()=>{landY=scrollY;},reduced?0:550);
  }

  pill.addEventListener("click",()=>{
    if(!origin) return;
    const y=origin.y;hide();
    scrollTo({top:y,behavior:reduced?"auto":"smooth"});
  });

  document.addEventListener("click",e=>{
    const a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(!a) return;
    // skip the coarse first-tap that only opens a preview
    if(e.defaultPrevented) return;
    const href=a.getAttribute("href")||"";
    if(href[0]!=="#") return;
    let id;try{id=decodeURIComponent(href.slice(1));}catch(err){id=href.slice(1);}
    if(!id||!document.getElementById(id)) return;
    show(scrollY,currentSecno());
  });

  addEventListener("scroll",()=>{
    if(!origin||landY===null) return;
    if(Math.abs(scrollY-landY)>innerHeight*0.7) hide();
  },{passive:true});
  // never cover the lightbox
  const lb=document.querySelector(".lb");
  if(lb) new MutationObserver(()=>{
    if(lb.classList.contains("on")) pill.style.visibility="hidden";
    else if(pill.classList.contains("on")) pill.style.visibility="";
  }).observe(lb,{attributes:true,attributeFilter:["class"]});
})();

/* ---------------- cited-in back-links ---------------- */
(function(){
  // Walk out of unnumbered subsections to the nearest numbered section so a
  // citation in "Notation" still lists under §1 rather than vanishing.
  function numberedSection(from){
    let sec=from&&from.closest("section");
    while(sec){
      if(sec.id){
        let h=null;
        for(const c of sec.children){
          if(c.matches("h2, h3")){h=c;break;}
          if(c.matches("details")){
            const s=c.querySelector(":scope > summary > h2");
            if(s){h=s;break;}
          }
        }
        const sn=h&&h.querySelector(".secno");
        const label=sn&&sn.textContent.trim();
        if(label) return {id:sec.id,label};
      }
      sec=sec.parentElement&&sec.parentElement.closest("section");
    }
    return null;
  }
  const map=new Map();
  document.querySelectorAll('a[role="doc-biblioref"]').forEach(a=>{
    const href=a.getAttribute("href")||"";
    if(href[0]!=="#") return;
    const key=href.slice(1);
    const where=numberedSection(a);
    if(!where) return;
    if(!map.has(key)) map.set(key,[]);
    const list=map.get(key);
    if(!list.some(x=>x.id===where.id)) list.push(where);
  });
  const rank=s=>s.split(".").map(n=>parseInt(n,10)||0);
  const cmp=(a,b)=>{
    const A=rank(a.label),B=rank(b.label);
    for(let i=0;i<Math.max(A.length,B.length);i++){
      const d=(A[i]||0)-(B[i]||0);if(d) return d;
    }
    return 0;
  };
  for(const [key,cites] of map){
    const el=document.getElementById(key);
    if(!el) continue;
    cites.sort(cmp);
    const cap=cites.slice(0,6),rest=cites.length-cap.length;
    const line=document.createElement("div");
    line.className="cited-in";
    line.appendChild(document.createTextNode("Cited in "));
    cap.forEach((c,i)=>{
      if(i) line.appendChild(document.createTextNode(", "));
      const a=document.createElement("a");
      a.href="#"+c.id;a.textContent="\u00A7"+c.label;
      line.appendChild(a);
    });
    if(rest) line.appendChild(document.createTextNode(" + "+rest+" more"));
    el.appendChild(line);
  }
})();

/* ---------------- print: open closed details ---------------- */
(function(){
  // CSS display:block does not open a closed <details>; force the open attribute.
  let opened=null;
  const openAll=()=>{
    if(opened) return;
    opened=[];
    document.querySelectorAll("details").forEach(d=>{
      if(!d.open){d.open=true;opened.push(d);}
    });
  };
  const restore=()=>{
    if(!opened) return;
    opened.forEach(d=>{d.open=false;});
    opened=null;
  };
  addEventListener("beforeprint",openAll);
  addEventListener("afterprint",restore);
  const mq=matchMedia("print");
  const onMq=e=>{if(e.matches) openAll(); else restore();};
  if(mq.addEventListener) mq.addEventListener("change",onMq);
  else if(mq.addListener) mq.addListener(onMq);
})();

/* ---------------- cross-references into folded back matter ---------------- */
function revealTarget(){
  const id=decodeURIComponent(location.hash.slice(1));
  if(!id) return;
  const el=document.getElementById(id);
  if(!el) return;
  let d=el.closest("details");
  while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
  requestAnimationFrame(()=>el.scrollIntoView({block:"start",behavior:"auto"}));
}
addEventListener("hashchange",revealTarget);
if(location.hash) addEventListener("load",revealTarget);
document.addEventListener("click",e=>{
  const a=e.target.closest&&e.target.closest('a[href^="#"]');
  if(!a) return;
  const el=document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));
  if(!el) return;
  let d=el.closest("details");
  while(d){d.open=true;d=d.parentElement&&d.parentElement.closest("details");}
});

/* ---------------- reveal + spine ---------------- */
const io=new IntersectionObserver(es=>{
  for(const e of es) if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target);}
},{rootMargin:"-20px"});
document.querySelectorAll(".reveal").forEach(n=>io.observe(n));

const figIn=new IntersectionObserver(es=>{
  for(const e of es) if(e.isIntersecting){e.target.classList.add("seen");figIn.unobserve(e.target);}
},{rootMargin:"0px 0px -8% 0px"});
document.querySelectorAll("figure").forEach(n=>figIn.observe(n));

const links=[...document.querySelectorAll(".spine a")];
function openBranch(a){
  if(!a) return;
  const li=a.closest("li.top");
  document.querySelectorAll(".toc > li.top").forEach(n=>{if(n!==li)n.classList.remove("open");});
  if(li) li.classList.add("open");
}
const spy=new IntersectionObserver(es=>{
  for(const e of es){
    if(!e.isIntersecting)continue;
    const id=e.target.id;
    links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+id));
    const on=links.find(a=>a.classList.contains("on"));
    openBranch(on);
    if(on&&on.offsetParent) on.scrollIntoView({block:"nearest"});
  }
},{rootMargin:"-20% 0px -70%"});
document.querySelectorAll("article section[id]").forEach(s=>spy.observe(s));
links.forEach(a=>a.addEventListener("click",()=>openBranch(a)));
openBranch(links[0]);

document.querySelectorAll(".hero-inner > *").forEach((n,i)=>{
  n.classList.add("hero-rise");
  n.style.animationDelay=(0.06*i+0.05).toFixed(2)+"s";
});
const rebuildHero=hero();
function repaint(){rebuildHero();MOUNTED.forEach(p=>p.redraw());}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",repaint);
new MutationObserver(repaint).observe(root,{attributes:true,attributeFilter:["data-theme"]});
"""
