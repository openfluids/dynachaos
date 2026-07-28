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
  --measure:35rem;
  --rail:14rem;
  --gutter:clamp(1.15rem,3.5vw,2.75rem);
  --shadow:0 1px 2px rgba(20,22,28,.05), 0 8px 28px rgba(20,22,28,.07);
}

:root[data-size="s"]{--base:1.0rem;--measure:32.5rem;}
:root[data-size="m"]{--base:1.075rem;--measure:34rem;}
:root[data-size="l"]{--base:1.155rem;--measure:35rem;}
:root[data-size="xl"]{--base:1.26rem;--measure:36.5rem;}

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
math[display="block"]{display:block;margin:1.4em auto;overflow-x:auto;overflow-y:hidden;max-width:100%;}

/* ------------------------------ type ------------------------------ */
.eyebrow{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.26em;text-transform:uppercase;color:var(--ink-low);margin:0;}
h1{font-size:clamp(2rem,1.1rem+3.6vw,3.6rem);line-height:1.03;letter-spacing:-0.028em;font-weight:700;margin:0.4em 0 0;max-width:19ch;text-wrap:balance;}
h2{font-size:clamp(1.35rem,1.1rem+0.9vw,1.75rem);line-height:1.2;letter-spacing:-0.014em;font-weight:700;margin:0 0 0.6em;text-wrap:balance;}
h3{font-size:1.06rem;font-weight:700;margin:2em 0 0.35em;text-wrap:balance;}
p{margin:0 0 0.95em;}
article li{margin-bottom:0.35em;}
a{color:var(--chaotic);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 35%,transparent);}
a:hover{border-bottom-color:currentColor;}
:focus-visible{outline:2px solid var(--chaotic);outline-offset:2px;border-radius:2px;}
code,.num{font-family:var(--mono);font-size:0.85em;font-variant-numeric:tabular-nums;}
pre{background:var(--sunken);border:1px solid var(--rule);border-radius:4px;padding:0.8rem 0.95rem;overflow-x:auto;font-size:0.78rem;line-height:1.55;}

/* ------------------------------ hero ------------------------------ */
.hero{position:relative;min-height:min(100svh,54rem);display:grid;grid-template-rows:1fr auto;overflow:hidden;border-bottom:1px solid var(--rule);}
#bifurcation{position:absolute;inset:-8% 0 -8% 0;width:100%;height:116%;display:block;will-change:transform;}
.hero::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(100deg,var(--ground) 0%,color-mix(in oklab,var(--ground) 92%,transparent) 38%,color-mix(in oklab,var(--ground) 22%,transparent) 68%,transparent 100%);}
.hero-inner{position:relative;z-index:2;align-self:center;width:100%;max-width:74rem;
  margin:0 auto;padding:clamp(2rem,7vh,5rem) var(--gutter);}
.hero-inner > *{max-width:min(46rem,92%);}
.hero-rise{opacity:0;transform:translateY(18px);animation:rise .85s cubic-bezier(.22,.68,.28,1) forwards;}
@keyframes rise{to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.hero-rise{opacity:1;transform:none;animation:none;}}
.byline{margin:1.6em 0 0;font-size:1rem;color:var(--ink);}
.byline .affil{display:block;color:var(--ink-low);font-size:0.86rem;margin-top:0.25em;max-width:44ch;}
.lede{font-size:clamp(1.04rem,0.95rem+0.4vw,1.24rem);line-height:1.55;color:var(--ink-mid);max-width:48ch;margin-top:1.5em;}
.stats{display:flex;flex-wrap:wrap;gap:2.25rem;margin-top:2.4rem;padding:0;list-style:none;}
.stats li{margin:0;}
.stats b{display:block;font-size:1.7rem;line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;}
.stats span{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.16em;text-transform:uppercase;color:var(--ink-low);}
.legend{position:relative;z-index:2;display:flex;flex-wrap:wrap;gap:0.3rem 1.4rem;padding:0.85rem var(--gutter) 1.4rem;font-family:var(--mono);font-size:0.66rem;letter-spacing:0.04em;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
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
.shell{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);gap:clamp(1.5rem,4vw,3.5rem);
  max-width:72rem;margin:0 auto;padding:clamp(2rem,4vw,3.5rem) var(--gutter) 5rem;align-items:start;}
:root[data-rail="off"] .shell{grid-template-columns:minmax(0,1fr);max-width:62rem;}
:root[data-rail="off"] .spine{display:none;}

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
figure{margin:2.1rem 0;background:var(--raised);border:1px solid var(--rule);border-radius:6px;overflow:hidden;
  box-shadow:var(--shadow);transition:box-shadow .3s ease,border-color .3s ease,transform .3s ease;}
figure:hover{border-color:color-mix(in oklab,var(--chaotic) 30%,var(--rule));
  box-shadow:0 2px 4px rgba(20,22,28,.05),0 14px 40px rgba(20,22,28,.10);transform:translateY(-1px);}
@media (prefers-reduced-motion:reduce){figure,figure:hover{transition:none;transform:none;}}
.fig-head{display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.7rem;border-bottom:1px solid var(--rule-soft);}
.fig-head .name{font-family:var(--mono);font-size:0.63rem;color:var(--ink-low);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.fig-head .acts{margin-left:auto;display:flex;gap:0.25rem;flex:none;}
.fig-head button{appearance:none;background:none;border:1px solid var(--rule);border-radius:4px;color:var(--ink-mid);
  font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.24rem 0.45rem;cursor:pointer;transition:all .15s;}
.fig-head button:hover{color:var(--chaotic);border-color:var(--chaotic);}
.fig-body{position:relative;background:var(--raised);}
figure img{display:block;width:100%;height:auto;max-height:16rem;object-fit:contain;object-position:center;
  cursor:zoom-in;padding:0.5rem;transition:max-height .35s ease;}
:root[data-figsize="tall"] figure img{max-height:30rem;}
figcaption{padding:0.6rem 0.8rem 0.75rem;font-size:0.82rem;line-height:1.5;color:var(--ink-mid);border-top:1px solid var(--rule-soft);}
figcaption .num{color:var(--ink);font-weight:700;}
.plot-wrap{position:relative;padding:0.35rem;}
canvas.plot{width:100%;display:block;touch-action:pan-y;}
.hint{margin:0;padding:0 0.8rem 0.6rem;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.06em;color:var(--ink-low);}
figure.plain{padding:0.8rem;font-size:0.85rem;}

.readout{position:absolute;pointer-events:none;z-index:4;background:var(--sunken);border:1px solid var(--rule);
  border-radius:4px;padding:0.3rem 0.45rem;font-family:var(--mono);font-size:0.66rem;line-height:1.45;
  color:var(--ink);white-space:pre;opacity:0;transition:opacity .1s;box-shadow:var(--shadow);}
.readout.on{opacity:1;}

/* ------------------------------ tables + refs ------------------------------ */
.table-wrap{overflow-x:auto;margin:1.5rem 0;border:1px solid var(--rule);border-radius:5px;}
table{border-collapse:collapse;font-size:0.83rem;width:100%;font-variant-numeric:tabular-nums;}
th,td{padding:0.4rem 0.65rem;text-align:left;border-bottom:1px solid var(--rule-soft);}
thead th{border-bottom:1px solid var(--ink-mid);font-weight:700;white-space:nowrap;}
tbody tr:last-child td{border-bottom:none;}
caption{caption-side:top;text-align:left;padding:0.6rem 0.65rem;font-size:0.82rem;color:var(--ink-mid);border-bottom:1px solid var(--rule-soft);}
.citation{white-space:nowrap;}
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

.reveal{opacity:0;transform:translateY(14px);transition:opacity .6s cubic-bezier(.22,.68,.28,1),transform .6s cubic-bezier(.22,.68,.28,1);}
.reveal.in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){.reveal{opacity:1;transform:none;transition:none;}html{scroll-behavior:auto;}.progress{transition:none;}}

@media print{
  :root{--ground:#fff;--raised:#fff;--ink:#000;--ink-mid:#333;--ink-low:#666;--rule:#bbb;--rule-soft:#ddd;}
  .hero{min-height:auto;border-bottom:2px solid #000;page-break-after:avoid;}
  #bifurcation,.hero::after,.legend,.spine,.controls,.progress,.lb,.fig-head .acts{display:none !important;}
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
    for(;col<end;col++) column(col,330,0.46);
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

  function reset(){
    const dpr=Math.min(devicePixelRatio||1,2);
    W=cv.clientWidth*dpr;H=cv.clientHeight*dpr;
    if(!W||!H) return;
    cv.width=W;cv.height=H;
    ctx.fillStyle=css("--ground");ctx.fillRect(0,0,W,H);
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

  addEventListener("resize",()=>{clearTimeout(cv._t);cv._t=setTimeout(reset,200);});
  reset();return reset;
}

/* ---------------- canvas plotting ---------------- */
function fmt(v){const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);
  return v.toFixed(a<1?4:3);}
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
    return {x0,x1,y0:y0-p,y1:y1+p};
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
  function ramp(t){
    const stops=[css("--locked"),css("--torus"),css("--chaotic")];
    const k=Math.max(0,Math.min(0.999,t))*(stops.length-1);
    const i=Math.floor(k),f=k-i,a=hex(stops[i]),b=hex(stops[i+1]||stops[i]);
    return `rgb(${a.map((v,j)=>Math.round(v+(b[j]-v)*f)).join(",")})`;
  }

  function draw(){
    if(!W||!H) return;
    ctx.clearRect(0,0,W,H);
    const rule=css("--rule"),low=css("--ink-low"),mid=css("--ink-mid");
    ctx.font="10px "+css("--mono");ctx.strokeStyle=rule;ctx.lineWidth=1;ctx.fillStyle=low;
    ctx.textAlign="right";ctx.textBaseline="middle";
    for(const t of ticks(dom.y0,dom.y1,4)){
      const y=Math.round(sy(t))+0.5;if(y<pad.t||y>H-pad.b)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmt(t),pad.l-7,y);
    }
    ctx.textAlign="center";ctx.textBaseline="top";
    for(const t of ticks(dom.x0,dom.x1,5)){
      const x=Math.round(sx(t))+0.5;if(x<pad.l||x>W-pad.r)continue;
      if(!heat){ctx.globalAlpha=.4;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmt(t),x,H-pad.b+7);
    }
    ctx.fillStyle=mid;ctx.textAlign="center";ctx.textBaseline="bottom";
    ctx.fillText(meta.xlabel||"",pad.l+(W-pad.l-pad.r)/2,H-2);
    ctx.save();ctx.translate(11,pad.t+(H-pad.t-pad.b)/2);ctx.rotate(-Math.PI/2);
    ctx.textBaseline="top";ctx.fillText(meta.ylabel||"",0,0);ctx.restore();

    ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b);ctx.clip();
    if(heat){
      const nx=panel.x.length,ny=panel.y.length;
      const cw=Math.abs(sx(panel.x[1])-sx(panel.x[0]))+1;
      const ch=Math.abs(sy(panel.y[1])-sy(panel.y[0]))+1;
      for(let j=0;j<ny;j++){
        const yv=sy(panel.y[j]);if(yv<pad.t-ch||yv>H-pad.b+ch)continue;
        const row=panel.z[j];
        for(let i=0;i<nx;i++){
          const v=row[i];if(v===null||Number.isNaN(v))continue;
          const xv=sx(panel.x[i]);if(xv<pad.l-cw||xv>W-pad.r+cw)continue;
          ctx.fillStyle=ramp((v-zmin)/((zmax-zmin)||1));ctx.fillRect(xv,yv-ch,cw,ch);
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
    if(drag&&drag.cur!==null){
      ctx.fillStyle=css("--chaotic");ctx.globalAlpha=.12;
      const a=Math.min(drag.start,drag.cur),b=Math.max(drag.start,drag.cur);
      ctx.fillRect(a,pad.t,b-a,H-pad.t-pad.b);ctx.globalAlpha=1;
    }
    ctx.restore();
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
      lines.push("value = "+fmt(panel.z[j][i]));
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
      MOUNTED.push(Plot(c,panel,{
        kind:spec.kind,
        xlabel:(spec.axes&&spec.axes.x&&spec.axes.x.label)||"",
        ylabel:(spec.axes&&spec.axes.y&&spec.axes.y.label)||""}));
    });
    const d=spec.decimation,n=document.createElement("p");n.className="hint";
    n.textContent="drag to zoom · double-click to reset · hover to read values"
      +(d&&d.method!=="none"?" · showing "+d.to.toLocaleString()+" of "+d.from.toLocaleString()+" points":"");
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

/* ---------------- lightbox ---------------- */
(function(){
  const lb=document.querySelector(".lb");if(!lb)return;
  const img=lb.querySelector("img"),cap=lb.querySelector(".lb-cap");
  const open=t=>{
    img.src=t.dataset.full||t.src;img.alt=t.alt;
    const f=t.closest("figure"),c=f&&f.querySelector("figcaption");
    cap.textContent=c?c.textContent.trim().slice(0,240):"";
    lb.classList.add("on");lb.querySelector(".lb-close").focus();
  };
  document.addEventListener("click",e=>{
    const t=e.target;
    if(t.tagName==="IMG"&&t.closest("figure")){open(t);return;}
    if(t.classList.contains("act-zoom")){
      const i=t.closest("figure").querySelector("img");if(i)open(i);return;}
    if(t.closest(".lb")) lb.classList.remove("on");
  });
  addEventListener("keydown",e=>{if(e.key==="Escape")lb.classList.remove("on");});
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
