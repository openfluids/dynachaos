"""Presentation shell for the interactive paper: fonts, tokens, CSS and JS.

Kept separate from ``build_paper.py`` so the design can be edited without
touching the pandoc pipeline. Everything here is inlined into the generated
page except the webfonts, which are served from ``site/fonts/``.

Palette note: the three accent hues are not decorative. They encode the
Lyapunov classification used throughout the paper --- locked, quasiperiodic,
chaotic --- and the hero canvas colours the logistic attractor by that same
rule, so the legend teaches the palette before the reader meets a figure.
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
    out = []
    for family, file, style, weight in FONT_FACES:
        out.append(
            f"@font-face{{font-family:'{family}';src:url('fonts/{file}') format('woff2');"
            f"font-style:{style};font-weight:{weight};font-display:swap;}}"
        )
    return "\n".join(out)


CSS = (
    _font_css()
    + r"""
:root{
  /* Lyapunov classification -- the palette IS the taxonomy. */
  --locked:#b46a08;   /* lambda < 0  mode-locked / periodic */
  --torus:#5b4dbe;    /* lambda ~ 0  quasiperiodic */
  --chaotic:#0f7f79;  /* lambda > 0  chaotic */

  --ground:#f8f6f0;
  --raised:#fffefb;
  --sunken:#eeebe1;
  --ink:#191b22;
  --ink-mid:#4a4f5e;
  --ink-low:#7c8395;
  --rule:#dcd8cb;
  --mark:#f0e8cf;

  --serif:'TeX Gyre Pagella', Palatino, 'Palatino Linotype', Georgia, serif;
  --math:'TeX Gyre Pagella Math', 'TeX Gyre Pagella', serif;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;

  --measure:38rem;
  --gutter:clamp(1.1rem,4vw,3rem);
}

@media (prefers-color-scheme: dark){
  :root{
    --locked:#f0a830; --torus:#8b7fd4; --chaotic:#35c8c0;
    --ground:#0b0e18; --raised:#131829; --sunken:#080a12;
    --ink:#e9edf7; --ink-mid:#a9b2c9; --ink-low:#6e7793;
    --rule:#222a3f; --mark:#2a2410;
  }
}
:root[data-theme="dark"]{
  --locked:#f0a830; --torus:#8b7fd4; --chaotic:#35c8c0;
  --ground:#0b0e18; --raised:#131829; --sunken:#080a12;
  --ink:#e9edf7; --ink-mid:#a9b2c9; --ink-low:#6e7793;
  --rule:#222a3f; --mark:#2a2410;
}
:root[data-theme="light"]{
  --locked:#b46a08; --torus:#5b4dbe; --chaotic:#0f7f79;
  --ground:#f8f6f0; --raised:#fffefb; --sunken:#eeebe1;
  --ink:#191b22; --ink-mid:#4a4f5e; --ink-low:#7c8395;
  --rule:#dcd8cb; --mark:#f0e8cf;
}

*{box-sizing:border-box;}
html{scroll-behavior:smooth;}

body{
  margin:0;
  background:var(--ground);
  color:var(--ink);
  font-family:var(--serif);
  font-size:clamp(1.02rem,0.96rem+0.22vw,1.14rem);
  line-height:1.62;
  text-rendering:optimizeLegibility;
  -webkit-font-smoothing:antialiased;
  overflow-x:hidden;
}

/* ---------------- math ---------------- */
math{font-family:var(--math);font-size:1.04em;}
math[display="block"]{
  display:block;
  margin:1.5em auto;
  overflow-x:auto;
  overflow-y:hidden;
  max-width:100%;
  padding:0.2em 0;
}

/* ---------------- type ---------------- */
.eyebrow{
  font-family:var(--mono);
  font-size:0.68rem;
  letter-spacing:0.24em;
  text-transform:uppercase;
  color:var(--ink-low);
  margin:0;
}
h1{
  font-size:clamp(2.5rem,1.1rem+6vw,5.6rem);
  line-height:0.94;
  letter-spacing:-0.035em;
  font-weight:700;
  margin:0.16em 0 0;
  text-wrap:balance;
}
h2{
  font-size:clamp(1.5rem,1.1rem+1.5vw,2.15rem);
  line-height:1.14;
  letter-spacing:-0.018em;
  font-weight:700;
  margin:0 0 0.5em;
  text-wrap:balance;
}
h3{font-size:1.16rem;font-weight:700;margin:2.2em 0 0.4em;letter-spacing:-0.01em;text-wrap:balance;}
h4{font-size:1rem;font-weight:700;font-style:italic;margin:1.8em 0 0.3em;}
p{margin:0 0 1.05em;}
em{font-style:italic;}

a{color:var(--chaotic);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--chaotic) 40%,transparent);}
a:hover{border-bottom-color:currentColor;}
:focus-visible{outline:2px solid var(--chaotic);outline-offset:3px;border-radius:2px;}

code,.num{font-family:var(--mono);font-size:0.86em;font-variant-numeric:tabular-nums;}
pre{
  background:var(--sunken);border:1px solid var(--rule);border-radius:3px;
  padding:0.85rem 1rem;overflow-x:auto;font-size:0.8rem;line-height:1.5;
}
pre code{font-size:inherit;}

/* ---------------- hero ---------------- */
.hero{position:relative;min-height:100svh;display:grid;grid-template-rows:1fr auto;overflow:hidden;border-bottom:1px solid var(--rule);}
#bifurcation{position:absolute;inset:0;width:100%;height:100%;display:block;}
.hero::after{
  content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(102deg,var(--ground) 0%,
    color-mix(in oklab,var(--ground) 90%,transparent) 36%,
    color-mix(in oklab,var(--ground) 20%,transparent) 66%,transparent 100%);
}
.hero-inner{position:relative;z-index:2;align-self:center;padding:var(--gutter);max-width:62rem;}
.authors{margin-top:1.6em;font-size:1.02rem;color:var(--ink-mid);}
.authors b{font-weight:700;color:var(--ink);}
.lede{font-size:clamp(1.02rem,0.9rem+0.4vw,1.2rem);color:var(--ink-mid);max-width:44ch;margin-top:1.1em;}
.legend{
  position:relative;z-index:2;display:flex;flex-wrap:wrap;gap:0.35rem 1.5rem;
  padding:0.9rem var(--gutter) 1.5rem;
  font-family:var(--mono);font-size:0.68rem;letter-spacing:0.05em;color:var(--ink-mid);
}
.legend b{font-weight:400;color:var(--ink);}
.swatch{display:inline-block;width:0.68rem;height:0.68rem;margin-right:0.45rem;vertical-align:-1px;border-radius:2px;}

/* ---------------- shell ---------------- */
.shell{
  display:grid;grid-template-columns:15rem minmax(0,1fr);
  gap:clamp(1.5rem,4vw,4rem);
  max-width:74rem;margin:0 auto;padding:clamp(2rem,5vw,4rem) var(--gutter) 5rem;
}
.spine{
  position:sticky;top:1.6rem;align-self:start;max-height:calc(100svh - 3rem);
  overflow-y:auto;display:flex;flex-direction:column;
  font-family:var(--mono);font-size:0.7rem;border-left:1px solid var(--rule);
}
.spine a{
  padding:0.34rem 0 0.34rem 0.75rem;margin-left:-1px;
  border-left:2px solid transparent;border-bottom:none;
  color:var(--ink-low);line-height:1.35;transition:color .18s,border-color .18s;
}
.spine a:hover,.spine a.on{color:var(--ink);border-left-color:var(--chaotic);}
.spine a.sub{padding-left:1.5rem;font-size:0.66rem;}
@media (max-width:64rem){.shell{grid-template-columns:minmax(0,1fr);}.spine{display:none;}}

article{max-width:var(--measure);}
article section{scroll-margin-top:1.5rem;}
article > section > h2{border-top:1px solid var(--rule);padding-top:1.5rem;margin-top:3rem;}
article > section:first-child > h2{margin-top:0;}

/* full-bleed figures escape the reading measure */
figure{
  margin:2.2rem 0;
  width:min(100%, calc(100vw - 2*var(--gutter)));
  background:var(--raised);border:1px solid var(--rule);border-radius:3px;overflow:hidden;
}
@media (min-width:64rem){figure{width:min(58rem, calc(100vw - 20rem));}}
.fig-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:0.4rem 0.9rem;padding:0.8rem 1rem;border-bottom:1px solid var(--rule);}
.fig-head .name{font-family:var(--mono);font-size:0.7rem;color:var(--ink-low);}
.tag{margin-left:auto;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.13em;text-transform:uppercase;color:var(--chaotic);border:1px solid currentColor;border-radius:2px;padding:0.08rem 0.38rem;}
.tag.static{color:var(--ink-low);}
figure img{display:block;width:100%;height:auto;cursor:zoom-in;background:var(--raised);}
figcaption{padding:0.85rem 1rem;font-size:0.87rem;line-height:1.55;color:var(--ink-mid);}
figcaption .lbl{font-weight:700;color:var(--ink);}
.plot-wrap{position:relative;padding:0.5rem;}
canvas.plot{width:100%;display:block;touch-action:pan-y;}
.hint{padding:0 1rem 0.8rem;font-family:var(--mono);font-size:0.63rem;letter-spacing:0.07em;color:var(--ink-low);}

.readout{
  position:absolute;pointer-events:none;z-index:4;
  background:var(--sunken);border:1px solid var(--rule);border-radius:2px;
  padding:0.32rem 0.5rem;font-family:var(--mono);font-size:0.68rem;line-height:1.5;
  color:var(--ink);white-space:pre;opacity:0;transition:opacity .12s;
}
.readout.on{opacity:1;}

/* tables (booktabs -> pandoc) */
.table-wrap{overflow-x:auto;margin:1.6rem 0;}
table{border-collapse:collapse;font-size:0.88rem;width:100%;font-variant-numeric:tabular-nums;}
th,td{padding:0.42rem 0.7rem;text-align:left;border-bottom:1px solid var(--rule);}
thead th{border-bottom:2px solid var(--ink-mid);font-weight:700;}
caption{caption-side:top;text-align:left;padding-bottom:0.5rem;font-size:0.87rem;color:var(--ink-mid);}

/* citations + references */
.citation{white-space:nowrap;}
#refs,.references{margin-top:1.5rem;font-size:0.86rem;}
.csl-entry{margin:0 0 0.65em;padding-left:1.6em;text-indent:-1.6em;color:var(--ink-mid);line-height:1.5;}

/* lightbox */
.lb{position:fixed;inset:0;z-index:60;display:none;place-items:center;background:color-mix(in oklab,var(--sunken) 92%,transparent);backdrop-filter:blur(6px);padding:2rem;}
.lb.on{display:grid;}
.lb img{max-width:100%;max-height:92svh;object-fit:contain;box-shadow:0 20px 60px rgba(0,0,0,.4);}
.lb-close{position:absolute;top:1rem;right:1.2rem;background:var(--raised);color:var(--ink);border:1px solid var(--rule);border-radius:3px;font-family:var(--mono);font-size:0.72rem;padding:0.4rem 0.7rem;cursor:pointer;}

/* theme toggle */
.theme{
  position:fixed;right:1rem;bottom:1rem;z-index:50;
  background:var(--raised);color:var(--ink);border:1px solid var(--rule);border-radius:999px;
  font-family:var(--mono);font-size:0.66rem;letter-spacing:0.1em;text-transform:uppercase;
  padding:0.5rem 0.85rem;cursor:pointer;box-shadow:0 3px 14px rgba(0,0,0,.14);
}
.theme:hover{border-color:var(--chaotic);color:var(--chaotic);}

.reveal{opacity:0;transform:translateY(12px);transition:opacity .55s,transform .55s;}
.reveal.in{opacity:1;transform:none;}
@media (prefers-reduced-motion:reduce){
  .reveal{opacity:1;transform:none;transition:none;}
  html{scroll-behavior:auto;}
}

.foot{border-top:1px solid var(--rule);padding:2.4rem var(--gutter);max-width:74rem;margin:0 auto;color:var(--ink-mid);font-size:0.85rem;}
"""
)


JS = r"""
"use strict";
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- theme ---------- */
(function(){
  const saved=localStorage.getItem("dc-theme");
  if(saved) document.documentElement.setAttribute("data-theme",saved);
  const btn=document.querySelector(".theme");
  if(!btn) return;
  const cur=()=>document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
  const sync=()=>{btn.textContent=cur()==="dark"?"light mode":"dark mode";};
  btn.addEventListener("click",()=>{
    const next=cur()==="dark"?"light":"dark";
    document.documentElement.setAttribute("data-theme",next);
    localStorage.setItem("dc-theme",next);
    sync();
  });
  sync();
})();

/* ---------- hero: logistic attractor, coloured by sign of lambda ---------- */
function hero(){
  const cv=document.getElementById("bifurcation");
  if(!cv) return ()=>{};
  const ctx=cv.getContext("2d",{alpha:false});
  let raf=null,col=0,W=0,H=0;
  function drawColumn(px){
    const r=2.85+(4.0-2.85)*(px/W);
    let x=0.35,l=0;
    for(let i=0;i<420;i++) x=r*x*(1-x);
    for(let i=0;i<260;i++){x=r*x*(1-x);l+=Math.log(Math.abs(r*(1-2*x))+1e-12);}
    l/=260;
    ctx.fillStyle=l>0.005?css("--chaotic"):l<-0.005?css("--locked"):css("--torus");
    ctx.globalAlpha=0.5;
    x=0.35;
    for(let i=0;i<300;i++) x=r*x*(1-x);
    for(let i=0;i<340;i++){x=r*x*(1-x);ctx.fillRect(px,(1-x)*H,1.15,1.15);}
    ctx.globalAlpha=1;
  }
  function step(target){
    const end=Math.min(W,target||col+26);
    for(;col<end;col++) drawColumn(col);
    if(col<W) raf=requestAnimationFrame(()=>step(0));
  }
  function reset(){
    const dpr=Math.min(devicePixelRatio||1,2);
    W=cv.clientWidth*dpr;H=cv.clientHeight*dpr;
    cv.width=W;cv.height=H;
    ctx.fillStyle=css("--ground");ctx.fillRect(0,0,W,H);
    col=0;if(raf)cancelAnimationFrame(raf);
    step(reduced?W:0);
  }
  addEventListener("resize",()=>{clearTimeout(cv._t);cv._t=setTimeout(reset,180);});
  reset();
  return reset;
}

/* ---------- canvas plotting ---------- */
function fmt(v){const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);
  return v.toFixed(a<1?4:3);}
function ticks(lo,hi,n){
  const raw=(hi-lo||1)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),nm=raw/mag;
  const st=(nm<1.5?1:nm<3?2:nm<7?5:10)*mag,out=[];
  for(let t=Math.ceil(lo/st)*st;t<=hi+1e-9;t+=st) out.push(t);
  return out;}

const PAL=["--chaotic","--locked","--torus"];

function Plot(canvas,panel,meta){
  const ctx=canvas.getContext("2d");
  const pad={l:64,r:14,t:panel.traces&&panel.traces.length>1?26:12,b:40};
  const wrap=canvas.parentElement;
  const tip=document.createElement("div");tip.className="readout";wrap.appendChild(tip);
  let W=0,H=0,dom=null,base=null,drag=null;
  const heat=meta.kind==="heatmap";

  function extent(){
    if(heat){
      return {x0:Math.min(...panel.x),x1:Math.max(...panel.x),
              y0:Math.min(...panel.y),y1:Math.max(...panel.y)};
    }
    let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
    for(const s of panel.traces){
      for(let i=0;i<s.x.length;i++){
        if(s.x[i]<x0)x0=s.x[i]; if(s.x[i]>x1)x1=s.x[i];
        const lo=s.yerr?s.y[i]-s.yerr[i]:s.y[i], hi=s.yerr?s.y[i]+s.yerr[i]:s.y[i];
        if(lo<y0)y0=lo; if(hi>y1)y1=hi;
      }
    }
    const p=(y1-y0)*0.08||0.5;
    return {x0,x1,y0:y0-p,y1:y1+p};
  }
  const sx=v=>pad.l+(v-dom.x0)/(dom.x1-dom.x0)*(W-pad.l-pad.r);
  const sy=v=>H-pad.b-(v-dom.y0)/(dom.y1-dom.y0)*(H-pad.t-pad.b);
  const ux=p=>dom.x0+(p-pad.l)/(W-pad.l-pad.r)*(dom.x1-dom.x0);

  let zmin=0,zmax=1;
  if(heat){
    zmin=Infinity;zmax=-Infinity;
    for(const row of panel.z) for(const v of row){
      if(v===null||Number.isNaN(v))continue;
      if(v<zmin)zmin=v; if(v>zmax)zmax=v;}
  }
  function ramp(t){
    // locked -> torus -> chaotic, through the palette the paper already uses
    const stops=[css("--locked"),css("--torus"),css("--chaotic")];
    const hex=c=>[parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];
    const k=Math.max(0,Math.min(0.999,t))*(stops.length-1);
    const i=Math.floor(k),f=k-i,a=hex(stops[i]),b=hex(stops[i+1]||stops[i]);
    return `rgb(${a.map((v,j)=>Math.round(v+(b[j]-v)*f)).join(",")})`;
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    const rule=css("--rule"),low=css("--ink-low"),mid=css("--ink-mid");
    ctx.font="11px "+css("--mono");
    ctx.strokeStyle=rule;ctx.lineWidth=1;ctx.fillStyle=low;

    ctx.textAlign="right";ctx.textBaseline="middle";
    for(const t of ticks(dom.y0,dom.y1,5)){
      const y=Math.round(sy(t))+0.5; if(y<pad.t||y>H-pad.b)continue;
      if(!heat){ctx.globalAlpha=.45;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmt(t),pad.l-8,y);
    }
    ctx.textAlign="center";ctx.textBaseline="top";
    for(const t of ticks(dom.x0,dom.x1,6)){
      const x=Math.round(sx(t))+0.5; if(x<pad.l||x>W-pad.r)continue;
      if(!heat){ctx.globalAlpha=.45;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();ctx.globalAlpha=1;}
      ctx.fillText(fmt(t),x,H-pad.b+8);
    }
    ctx.fillStyle=mid;
    ctx.textAlign="center";ctx.textBaseline="bottom";
    ctx.fillText(meta.xlabel||"",pad.l+(W-pad.l-pad.r)/2,H-4);
    ctx.save();ctx.translate(13,pad.t+(H-pad.t-pad.b)/2);ctx.rotate(-Math.PI/2);
    ctx.textBaseline="top";ctx.fillText(meta.ylabel||"",0,0);ctx.restore();

    ctx.save();ctx.beginPath();
    ctx.rect(pad.l,pad.t,W-pad.l-pad.r,H-pad.t-pad.b);ctx.clip();

    if(heat){
      const nx=panel.x.length,ny=panel.y.length;
      const cw=Math.abs(sx(panel.x[1])-sx(panel.x[0]))+1;
      const ch=Math.abs(sy(panel.y[1])-sy(panel.y[0]))+1;
      for(let j=0;j<ny;j++){
        const yv=sy(panel.y[j]);
        if(yv<pad.t-ch||yv>H-pad.b+ch)continue;
        const row=panel.z[j];
        for(let i=0;i<nx;i++){
          const v=row[i]; if(v===null||Number.isNaN(v))continue;
          const xv=sx(panel.x[i]);
          if(xv<pad.l-cw||xv>W-pad.r+cw)continue;
          ctx.fillStyle=ramp((v-zmin)/((zmax-zmin)||1));
          ctx.fillRect(xv,yv-ch,cw,ch);
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
          ctx.fillStyle=c;ctx.globalAlpha=.75;
          for(let i=0;i<s.x.length;i++) ctx.fillRect(sx(s.x[i])-1,sy(s.y[i])-1,2,2);
          ctx.globalAlpha=1;
        }else{
          ctx.strokeStyle=c;ctx.lineWidth=1.6;ctx.lineJoin="round";ctx.beginPath();
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

    if(!heat&&panel.traces.length>1){
      ctx.font="11px "+css("--mono");ctx.textAlign="left";ctx.textBaseline="middle";
      let lx=pad.l+6;
      panel.traces.forEach((s,k)=>{
        ctx.fillStyle=css(s.color||PAL[k%PAL.length]);
        ctx.fillRect(lx,pad.t-11,14,2.5);
        ctx.fillStyle=mid;ctx.fillText(s.name||"",lx+20,pad.t-10);
        lx+=28+ctx.measureText(s.name||"").width;
      });
    }
  }

  function resize(){
    const dpr=Math.min(devicePixelRatio||1,2);
    const cw=canvas.clientWidth;
    const chh=Math.round(Math.min(430,Math.max(250,cw*(heat?0.52:0.44))));
    canvas.style.height=chh+"px";
    canvas.width=cw*dpr;canvas.height=chh*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    W=cw;H=chh;draw();
  }

  canvas.addEventListener("pointermove",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;
    if(drag){drag.cur=Math.max(pad.l,Math.min(W-pad.r,px));draw();return;}
    if(px<pad.l||px>W-pad.r||py<pad.t||py>H-pad.b){tip.classList.remove("on");return;}
    const xv=ux(px);const lines=[(meta.xlabel||"x")+" = "+fmt(xv)];
    if(heat){
      const yv=dom.y0+(H-pad.b-py)/(H-pad.t-pad.b)*(dom.y1-dom.y0);
      const near=(arr,v)=>{let b=0,d=Infinity;for(let i=0;i<arr.length;i++){const q=Math.abs(arr[i]-v);if(q<d){d=q;b=i;}}return b;};
      const i=near(panel.x,xv),j=near(panel.y,yv);
      lines[0]=(meta.xlabel||"x")+" = "+fmt(panel.x[i]);
      lines.push((meta.ylabel||"y")+" = "+fmt(panel.y[j]));
      lines.push("value = "+fmt(panel.z[j][i]));
    }else{
      for(const s of panel.traces){
        let b=0,d=Infinity;
        for(let i=0;i<s.x.length;i++){const q=Math.abs(s.x[i]-xv);if(q<d){d=q;b=i;}}
        lines.push((s.name||"y")+" = "+fmt(s.y[b]));
      }
    }
    tip.textContent=lines.join("\n");tip.classList.add("on");
    const tw=tip.offsetWidth,th=tip.offsetHeight;
    tip.style.left=Math.min(px+14,W-tw-6)+"px";
    tip.style.top=Math.max(6,Math.min(py-th-10,H-th-6))+"px";
  });
  canvas.addEventListener("pointerleave",()=>tip.classList.remove("on"));
  canvas.addEventListener("pointerdown",e=>{
    const r=canvas.getBoundingClientRect(),px=e.clientX-r.left;
    if(px<pad.l||px>W-pad.r)return;
    drag={start:px,cur:null};canvas.setPointerCapture(e.pointerId);tip.classList.remove("on");
  });
  canvas.addEventListener("pointerup",()=>{
    if(drag&&drag.cur!==null&&Math.abs(drag.cur-drag.start)>12){
      const a=ux(Math.min(drag.start,drag.cur)),b=ux(Math.max(drag.start,drag.cur));
      if(heat){dom={...dom,x0:a,x1:b};}
      else{
        const ys=[];
        for(const s of panel.traces) for(let i=0;i<s.x.length;i++)
          if(s.x[i]>=a&&s.x[i]<=b) ys.push(s.y[i]);
        if(ys.length>1){
          const y0=Math.min(...ys),y1=Math.max(...ys),p=(y1-y0)*0.08||0.5;
          dom={x0:a,x1:b,y0:y0-p,y1:y1+p};
        }
      }
    }
    drag=null;draw();
  });
  canvas.addEventListener("dblclick",()=>{dom={...base};draw();});

  base=extent();dom={...base};
  new ResizeObserver(resize).observe(canvas);
  resize();
  return {redraw:draw};
}

/* ---------- lazy-mount interactive figures ---------- */
const MOUNTED=[];
const figIO=new IntersectionObserver(async(es)=>{
  for(const e of es){
    if(!e.isIntersecting) continue;
    const host=e.target; figIO.unobserve(host);
    try{
      const spec=await (await fetch(host.dataset.src)).json();
      host.innerHTML="";
      spec.panels.forEach(panel=>{
        if(panel.title){
          const h=document.createElement("p");
          h.className="hint";h.style.padding="0.3rem 0.5rem 0";h.textContent=panel.title;
          host.appendChild(h);
        }
        const w=document.createElement("div");w.className="plot-wrap";
        const c=document.createElement("canvas");c.className="plot";
        w.appendChild(c);host.appendChild(w);
        MOUNTED.push(Plot(c,panel,{
          kind:spec.kind,
          xlabel:(spec.axes&&spec.axes.x&&spec.axes.x.label)||"",
          ylabel:(spec.axes&&spec.axes.y&&spec.axes.y.label)||""
        }));
      });
      const d=spec.decimation;
      if(d&&d.method!=="none"){
        const n=document.createElement("p");
        n.className="hint";
        n.textContent="drag to zoom · double-click to reset · showing "
          +d.to.toLocaleString()+" of "+d.from.toLocaleString()+" points ("+d.method+")";
        host.appendChild(n);
      }else{
        const n=document.createElement("p");
        n.className="hint";n.textContent="drag to zoom · double-click to reset · hover to read values";
        host.appendChild(n);
      }
    }catch(err){
      host.innerHTML='<p class="hint">interactive view unavailable — static figure shown above</p>';
    }
  }
},{rootMargin:"300px"});
document.querySelectorAll("[data-src]").forEach(n=>figIO.observe(n));

/* ---------- lightbox ---------- */
(function(){
  const lb=document.querySelector(".lb");if(!lb)return;
  const img=lb.querySelector("img");
  document.addEventListener("click",e=>{
    const t=e.target;
    if(t.tagName==="IMG"&&t.closest("figure")&&t.dataset.full){
      img.src=t.dataset.full;img.alt=t.alt;lb.classList.add("on");
    }else if(t.closest(".lb")){lb.classList.remove("on");}
  });
  addEventListener("keydown",e=>{if(e.key==="Escape")lb.classList.remove("on");});
})();

/* ---------- reveal + spine ---------- */
const io=new IntersectionObserver(es=>{
  for(const e of es) if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target);}
},{rootMargin:"-30px"});
document.querySelectorAll(".reveal").forEach(n=>io.observe(n));

const links=[...document.querySelectorAll(".spine a")];
const spy=new IntersectionObserver(es=>{
  for(const e of es){
    if(!e.isIntersecting)continue;
    links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+e.target.id));
  }
},{rootMargin:"-25% 0px -65%"});
document.querySelectorAll("article section[id]").forEach(s=>spy.observe(s));

const rebuild=hero();
const repaint=()=>{rebuild();MOUNTED.forEach(p=>p.redraw());};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",repaint);
new MutationObserver(repaint).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
"""
