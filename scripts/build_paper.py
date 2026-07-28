"""Build the interactive paper at ``site/index.html``.

Two modes, and the split matters:

**Import** (local, occasional) --- ``--manuscript <path-to.tex>`` runs pandoc on
a LaTeX source that lives *outside* this repository, converts it to HTML with
native MathML and resolved citations, and caches the result as
``web/paper-body.html``. Requires pandoc.

**Build** (default, and what CI runs) --- reads the cached
``web/paper-body.html`` and assembles the finished page. Requires no pandoc, no
LaTeX, and no manuscript source. This repository therefore holds a website, not
a manuscript: no ``.tex``, no ``.bib``, no PDF.

Equations are native MathML, so they need no JavaScript and no math library;
``TeX Gyre Pagella Math`` makes them match the body text.
"""

# ruff: noqa: E501 -- inlined HTML templates are kept in their authored form.

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_shell import CSS, JS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
BODY = WEB / "paper-body.html"
META = WEB / "paper-meta.json"
SITE = REPO / "site"
FONTS_SRC = REPO / "assets" / "fonts"

FIG_RE = re.compile(r"<figure\b.*?</figure>", re.S)
IMG_RE = re.compile(r'<img[^>]*src="(?:figures/)?([a-z0-9_]+)/([a-z0-9_]+)\.png"[^>]*/?>', re.I)
CAP_RE = re.compile(r"<figcaption>(.*?)</figcaption>", re.S)
SEC_RE = re.compile(r'<section id="([^"]+)" class="([^"]*)"[^>]*>')
HEAD_RE = re.compile(r"<(h[12])([^>]*)>(.*?)</\1>", re.S)
TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)


ANCHORED_ENVS = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|table\*?)\}(.*?)\\end\{\1\}", re.S
)
LABEL_RE = re.compile(r"\\label\{((?:eq|tab):[^}]+)\}")
ID_ATTR = re.compile(r'\sid="([^"]+)"')


def anchor_labels(tex: str) -> tuple[str, int]:
    """Insert ``\\hypertarget`` before every labelled equation and table.

    Pandoc drops ``\\label`` inside display math, and misses it on some table
    environments, so ``\\eqref`` and table cross-references would dangle.
    Emitting an explicit hypertarget gives each one a real anchor. Duplicates
    (where pandoc *did* emit an id) are removed afterwards by
    :func:`dedupe_ids`. Operates on an in-memory copy; the manuscript on disk is
    never touched.
    """
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        label = LABEL_RE.search(match.group(2))
        if not label:
            return match.group(0)
        count += 1
        return f"\\hypertarget{{{label.group(1)}}}{{}}{match.group(0)}"

    return ANCHORED_ENVS.sub(repl, tex), count


def dedupe_ids(body: str) -> tuple[str, int]:
    """Drop repeated ``id`` attributes, keeping the first occurrence of each.

    A duplicate id makes the second target unreachable, so this keeps the
    earliest anchor -- which is the injected hypertarget, sitting immediately
    before the object it names.
    """
    seen: set[str] = set()
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        value = match.group(1)
        if value in seen:
            removed += 1
            return ""
        seen.add(value)
        return match.group(0)

    return ID_ATTR.sub(repl, body), removed


ACCENTS = {
    ("'", "e"): "é", ("'", "E"): "É", ("`", "e"): "è", ("`", "E"): "È",
    ("^", "e"): "ê", ("^", "E"): "Ê", ('"', "e"): "ë", ("'", "a"): "á",
    ("`", "a"): "à", ("^", "a"): "â", ('"', "a"): "ä", ("'", "o"): "ó",
    ("^", "o"): "ô", ('"', "o"): "ö", ("'", "i"): "í", ("^", "i"): "î",
    ('"', "i"): "ï", ("'", "u"): "ú", ("^", "u"): "û", ('"', "u"): "ü",
    ("~", "n"): "ñ", ("'", "c"): "ć", ("~", "a"): "ã", ("~", "o"): "õ",
}  # fmt: skip

ACCENT_RE = re.compile(r"\\(['`^\"~=.])\{?([a-zA-Z])\}?")


def extract_meta(tex: str) -> dict[str, str]:
    """Pull title, byline, affiliation and a lede from the manuscript preamble.

    Cached to ``web/paper-meta.json`` at import so the build never needs the
    LaTeX source. Anything not found is left blank rather than invented --- an
    author list is not something to guess at.
    """

    def clean(raw: str) -> str:
        raw = ACCENT_RE.sub(lambda m: ACCENTS.get((m.group(1), m.group(2)), m.group(2)), raw)
        raw = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", raw)
        raw = raw.replace(r"\&", "&").replace("~", " ").replace("--", "\u2013")
        raw = re.sub(r"\\[a-zA-Z]+", "", raw)
        raw = raw.replace("{", "").replace("}", "")
        return " ".join(raw.split()).strip()

    def find(pattern: str) -> str:
        m = re.search(pattern, tex, re.S)
        return clean(m.group(1)) if m else ""

    abstract = find(r"\\begin\{abstract\}(.*?)\\end\{abstract\}")
    sentences = [x for x in re.split(r"(?<=\.)\s+", abstract) if x]

    # Lead with the claim, not the background: prefer the sentence that states
    # what was investigated over the one describing prior work.
    start = 0
    for i, sentence in enumerate(sentences):
        if re.match(r"\s*We\b", sentence):
            start = i
            break
    lede = ""
    for sentence in sentences[start:]:
        if lede and len(lede) + len(sentence) > 320:
            break
        lede += (" " if lede else "") + sentence

    return {
        "title": find(r"\\title\{(.+?)\}\s*\n"),
        "authors": find(r"\\author\{(.+?)\}"),
        "affil": find(r"\\affil[^{]*\{(.+?)\}"),
        "lede": lede or abstract[:320],
    }


def run_pandoc(source: Path) -> str:
    """Convert a LaTeX manuscript to an HTML fragment with MathML and citations."""
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH; install it to use --manuscript")
    if not source.exists():
        raise SystemExit(f"manuscript not found: {source}")
    bib = source.parent / "references.bib"

    patched, n_anchor = anchor_labels(source.read_text(encoding="utf-8"))
    staged = Path(tempfile.mkdtemp(prefix="dynachaos-paper-")) / source.name
    staged.write_text(patched, encoding="utf-8")
    print(f"  anchored {n_anchor} labelled equations and tables")

    cmd = [
        "pandoc", str(staged), "-f", "latex", "-t", "html5",
        "--mathml", "--citeproc", "--section-divs", "--wrap=none",
        "--resource-path", str(source.parent),
    ]  # fmt: skip
    if bib.exists():
        cmd += ["--bibliography", str(bib)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"pandoc failed:\n{proc.stderr}")
    for line in proc.stderr.splitlines():
        if line.startswith("[WARNING]"):
            print(f"  pandoc: {line}")
    return proc.stdout


def strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def figure_block(
    fig_id: str, section: str, name: str, caption: str, dims: dict[str, tuple[int, int]]
) -> str:
    """Render one figure.

    The static image is always present and is what the reader sees first; the
    interactive chart is fetched only when asked for, so no reader pays for a
    payload they never open.
    """
    thumb = SITE / "thumbs" / section / f"{name}.webp"
    data = SITE / "data" / section / f"{name}.json"
    src = f"thumbs/{section}/{name}.webp" if thumb.exists() else f"full/{section}/{name}.png"
    size = ""
    if (section, name) in dims:
        w, h = dims[(section, name)]
        size = f' width="{w}" height="{h}"'
    alt = html.escape(strip_tags(caption)[:150], quote=True)

    acts = ['<button type="button" class="act-zoom">enlarge</button>']
    attrs = f' id="{fig_id}"'
    if data.exists():
        acts.insert(0, '<button type="button" class="act-interact">interact</button>')
        attrs += f' data-src="data/{section}/{name}.json" data-state="static"'

    return (
        f"<figure{attrs}>"
        f'<div class="fig-head"><span class="name">{section}/{name}</span>'
        f'<span class="acts">{"".join(acts)}</span></div>'
        f'<div class="fig-body">'
        f'<img src="{src}" data-full="full/{section}/{name}.png" alt="{alt}"'
        f' loading="lazy"{size}>'
        f"</div>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def thumb_dims() -> dict[str, tuple[int, int]]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return {}
    out: dict[str, tuple[int, int]] = {}
    for path in (SITE / "thumbs").rglob("*.webp"):
        with Image.open(path) as im:
            out[(path.parent.name, path.stem)] = im.size
    return out


def transform(body: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Rewrite pandoc output into the page's own figure, table and heading shapes."""
    dims = thumb_dims()
    stats = {"interactive": 0, "static": 0, "passthrough": 0}

    def do_figure(match: re.Match[str]) -> str:
        block = match.group(0)
        fid = re.search(r'<figure id="([^"]+)"', block)
        fig_id = fid.group(1) if fid else ""
        img = IMG_RE.search(block)
        cap = CAP_RE.search(block)
        caption = cap.group(1).strip() if cap else ""
        if not img:
            stats["passthrough"] += 1
            return f'<figure class="plain">{block[len("<figure") :]}'.replace(
                "</figure>", "</figure>", 1
            )
        section, name = img.group(1), img.group(2)
        if (SITE / "data" / section / f"{name}.json").exists():
            stats["interactive"] += 1
        else:
            stats["static"] += 1
        return figure_block(fig_id, section, name, caption, dims)

    body = FIG_RE.sub(do_figure, body)
    body = TABLE_RE.sub(lambda m: f'<div class="table-wrap">{m.group(0)}</div>', body)

    # Keep pandoc's ids verbatim: they are the manuscript's own \label anchors,
    # and every internal cross-reference in the text points at them.
    body = SEC_RE.sub(lambda m: f'<section id="{m.group(1)}" class="{m.group(2)} reveal">', body)

    # level1 -> h2, level2 -> h3, so the shell's type scale applies.
    nav: list[tuple[int, str, str]] = []

    def demote(match: re.Match[str]) -> str:
        tag, attrs, text = match.group(1), match.group(2), match.group(3)
        new = "h2" if tag == "h1" else "h3"
        return f"<{new}{attrs}>{text}</{new}>"

    for m in re.finditer(r'<section id="([^"]+)" class="([^"]*)"', body):
        sec_id, cls = m.group(1), m.group(2)
        tail = body[m.end() : m.end() + 4000]
        h = HEAD_RE.search(tail)
        if not h:
            continue
        level = 1 if "level1" in cls else 2
        nav.append((level, sec_id, strip_tags(h.group(3))))

    body = HEAD_RE.sub(demote, body)
    print(
        f"  figures: {stats['interactive']} interactive, {stats['static']} static, "
        f"{stats['passthrough']} passthrough"
    )
    return body, nav


def build_nav(nav: list[tuple[int, str, str]]) -> str:
    out = ['<nav class="spine" aria-label="Contents">']
    for level, sec_id, title in nav:
        cls = ' class="sub"' if level == 2 else ""
        out.append(f'<a href="#{sec_id}"{cls}>{html.escape(title)}</a>')
    out.append("</nav>")
    return "".join(out)


def hero(meta: dict[str, str]) -> str:
    stats = "".join(
        f"<li><b>{b}</b><span>{lab}</span></li>"
        for b, lab in (
            ("6", "mechanisms"),
            ("5&ndash;1000&times;", "finer resolution"),
            ("4", "modern diagnostics"),
        )
    )
    return f"""<header class="hero">
<canvas id="bifurcation" aria-hidden="true"></canvas>
<div class="hero-inner">
<p class="eyebrow">Interactive study &middot; built on dynachaos</p>
<h1>{html.escape(meta["title"])}</h1>
<p class="byline">{html.escape(meta["authors"])}<span class="affil">{html.escape(meta["affil"])}</span></p>
<p class="lede">{html.escape(meta["lede"])}</p>
<ul class="stats">{stats}</ul>
</div>
<div class="legend">
<span><span class="swatch" style="background:var(--locked)"></span><b>&lambda; &lt; 0</b> &nbsp;mode-locked</span>
<span><span class="swatch" style="background:var(--torus)"></span><b>&lambda; &asymp; 0</b> &nbsp;quasiperiodic</span>
<span><span class="swatch" style="background:var(--chaotic)"></span><b>&lambda; &gt; 0</b> &nbsp;chaotic</span>
<span style="color:var(--ink-low)">above &mdash; logistic attractor, computed live in your browser</span>
</div>
</header>"""


CONTROLS = """<div class="progress" aria-hidden="true"></div>
<div class="controls" role="group" aria-label="Reading controls">
<button type="button" class="c-rail" title="Toggle contents sidebar">contents</button>
<span class="sep"></span>
<button type="button" class="c-smaller" title="Smaller text" aria-label="Smaller text">A&minus;</button>
<button type="button" class="c-larger" title="Larger text" aria-label="Larger text">A+</button>
<span class="sep"></span>
<button type="button" class="c-fig" title="Toggle larger figures">figures</button>
<button type="button" class="c-theme" title="Cycle theme: auto, light, dark">auto</button>
</div>"""


def assemble(body: str, nav: str, meta: dict[str, str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(meta["title"])}</title>
<meta name="description" content="{html.escape(meta["lede"][:180])}">
<style>{CSS}</style>
</head>
<body>
{CONTROLS}
{hero(meta)}
<div class="shell">
{nav}
<article>
{body}
</article>
</div>
<footer class="foot">
<p>Every figure is recomputed from scratch by the same public commands, on any machine:</p>
<pre><code>pip install dynachaos
dynachaos list
dynachaos run all</code></pre>
<p><a href="gallery.html">Figure index</a> &middot;
<a href="https://github.com/openfluids/dynachaos">dynachaos on GitHub</a></p>
</footer>
<div class="lb"><button class="lb-close" type="button">close</button><img alt=""><p class="lb-cap"></p></div>
<script>{JS}</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manuscript", type=Path, help="LaTeX source to import (local only)")
    args = ap.parse_args()

    if args.manuscript:
        WEB.mkdir(parents=True, exist_ok=True)
        tex = args.manuscript.read_text(encoding="utf-8")
        raw = run_pandoc(args.manuscript)
        BODY.write_text(raw, encoding="utf-8")
        meta = extract_meta(tex)
        missing = [k for k, v in meta.items() if not v]
        if missing:
            print(f"  WARNING: could not parse from the manuscript: {', '.join(missing)}")
        META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  imported {len(raw):,} bytes -> {BODY.relative_to(REPO)}")
        print(f"  title: {meta['title'][:70]}")

    if not BODY.exists():
        raise SystemExit(
            f"{BODY.relative_to(REPO)} is missing. Import it once with:\n"
            f"  uv run --extra viz python scripts/build_paper.py --manuscript <path/to/main.tex>"
        )

    SITE.mkdir(parents=True, exist_ok=True)
    fonts_dst = SITE / "fonts"
    fonts_dst.mkdir(parents=True, exist_ok=True)
    for face in FONTS_SRC.glob("*.woff2"):
        shutil.copy2(face, fonts_dst / face.name)

    if not META.exists():
        raise SystemExit(f"{META.relative_to(REPO)} is missing. Import it once with --manuscript.")
    meta = json.loads(META.read_text(encoding="utf-8"))

    body, nav = transform(BODY.read_text(encoding="utf-8"))
    body, dropped = dedupe_ids(body)
    if dropped:
        print(f"  removed {dropped} duplicate id attributes")
    page = assemble(body, build_nav(nav), meta)
    out = SITE / "index.html"
    out.write_text(page, encoding="utf-8")

    print(f"  sections in nav: {len(nav)}")
    print(f"  MathML nodes:    {page.count('<math')}")
    print(f"  references:      {page.count('csl-entry')}")
    print(f"  fonts copied:    {len(list(fonts_dst.glob('*.woff2')))}")
    print(f"  wrote {out.relative_to(REPO)} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
