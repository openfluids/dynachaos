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
PROGRAM_ARC_SLOT = "<!--PROGRAM-ARC-->"
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


COLSPEC_NOISE = re.compile(r"[@><!]\{(?:[^{}]|\{[^{}]*\})*\}")
COLSPEC_SIZED = re.compile(r"[pmb]\{[^{}]*\}")
COLSPEC_STAR = re.compile(r"\*\{(\d+)\}\{([^{}]*)\}")


def count_columns(spec: str) -> int:
    """Count real columns in a LaTeX column specification."""
    spec = COLSPEC_STAR.sub(lambda m: m.group(2) * int(m.group(1)), spec)
    spec = COLSPEC_NOISE.sub("", spec)
    spec = COLSPEC_SIZED.sub("l", spec)
    return sum(1 for ch in spec if ch in "lcrXY")


def brace_group(text: str, start: int) -> tuple[str, int]:
    """Read one balanced ``{...}`` group beginning at ``start``.

    Column specifications nest braces (``>{\\raggedright}p{2.2cm}``), so a
    non-greedy regex stops at the first closing brace and captures nonsense.
    """
    if start >= len(text) or text[start] != "{":
        return "", start
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return "", start


def tabularx_to_tabular(tex: str) -> tuple[str, int]:
    """Rewrite ``tabularx`` environments as plain ``tabular``.

    Pandoc converts ``tabular`` correctly but leaves ``tabularx`` untouched, so
    the column specification and every ``&`` separator land on the page as
    prose. The X/Y stretch columns carry no meaning on the web -- the table is
    laid out by CSS -- so collapsing them to ``l`` loses nothing.
    """
    out, pos, count = [], 0, 0
    marker = "\\begin{tabularx}"
    while True:
        i = tex.find(marker, pos)
        if i < 0:
            break
        j = i + len(marker)
        _, j = brace_group(tex, j)  # width argument
        spec, j = brace_group(tex, j)  # column specification
        end = tex.find("\\end{tabularx}", j)
        if not spec or end < 0:
            out.append(tex[pos : i + len(marker)])
            pos = i + len(marker)
            continue
        cols = count_columns(spec)
        if not cols:
            out.append(tex[pos:j])
            pos = j
            continue
        count += 1
        out.append(tex[pos:i])
        body = tex[j:end]
        out.append(f"\\begin{{tabular}}{{{'l' * cols}}}{body}\\end{{tabular}}")
        pos = end + len("\\end{tabularx}")
    out.append(tex[pos:])
    return "".join(out), count


TAGGED_ENV = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\1\}", re.S)
TAG_RE = re.compile(r"\\tag\{(.*?)\}\s*")
TAG_SYMBOLS = {"\\star": "\u2605", "\\dagger": "\u2020", "\\ddagger": "\u2021", "\\ast": "*"}


def tag_text(raw: str) -> str:
    """Turn a LaTeX ``\\tag`` argument into the text it should display."""
    out = raw.strip().strip("$").strip()
    for macro, glyph in TAG_SYMBOLS.items():
        out = out.replace(macro, glyph)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    return out.replace("{", "").replace("}", "").strip()


def extract_tags(tex: str) -> tuple[str, list[str]]:
    """Pull ``\\tag{...}`` out of display maths and anchor each one.

    Pandoc cannot convert ``\\tag`` and drops the marker, so the manuscript's
    starred preview equations arrive with no marker at all. Removing the tag
    lets the maths convert, and the anchor lets the marker be put back beside
    the equation afterwards.
    """
    tags: list[str] = []

    def repl(match: re.Match[str]) -> str:
        body = match.group(2)
        found = TAG_RE.search(body)
        if not found:
            return match.group(0)
        index = len(tags)
        tags.append(tag_text(found.group(1)))
        cleaned = TAG_RE.sub("", body)
        return (
            f"\\hypertarget{{eqtag-{index}}}{{}}"
            f"\\begin{{{match.group(1)}}}{cleaned}\\end{{{match.group(1)}}}"
        )

    return TAGGED_ENV.sub(repl, tex), tags


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


def run_pandoc(source: Path) -> tuple[str, list[str]]:
    """Convert a LaTeX manuscript to an HTML fragment with MathML and citations."""
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH; install it to use --manuscript")
    if not source.exists():
        raise SystemExit(f"manuscript not found: {source}")
    bib = source.parent / "references.bib"

    raw_tex = source.read_text(encoding="utf-8")
    raw_tex, n_tabx = tabularx_to_tabular(raw_tex)
    raw_tex, eq_tags = extract_tags(raw_tex)
    if eq_tags:
        print(f"  preserved {len(eq_tags)} tagged equations: {' '.join(eq_tags)}")
    if n_tabx:
        print(f"  rewrote {n_tabx} tabularx environments as tabular")
    patched, n_anchor = anchor_labels(raw_tex)
    staged = Path(tempfile.mkdtemp(prefix="dynachaos-paper-")) / source.name
    staged.write_text(patched, encoding="utf-8")
    print(f"  anchored {n_anchor} labelled equations and tables")

    cmd = [
        "pandoc", str(staged), "-f", "latex", "-t", "html5",
        "--mathml", "--citeproc", "--section-divs", "--wrap=none",
        # every author-year links to its own bibliography entry, as hyperref does
        "-M", "link-citations=true", "-M", "link-bibliography=true",
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
    return proc.stdout, eq_tags


HYPERTARGET_RE = re.compile(r'<(span|div) id="((?:eq|tab):[^"]+)"[^>]*>\s*</\1>')
BLOCK_MATH_RE = re.compile(r'<math display="block".*?</math>', re.S)
ANNOTATION_RE = re.compile(r"<annotation\b[^>]*>.*?</annotation>", re.S)
XREF_RE = re.compile(
    r'(<a href="#([^"]+)"[^>]*data-reference-type="(?:eqref|ref)"[^>]*>)\[([^\]]+)\]</a>'
)


def apply_eq_tags(body: str, tags: list[str]) -> tuple[str, int]:
    """Put each recovered ``\\tag`` marker beside its equation."""
    if not tags:
        return body, 0
    out, pos, done = [], 0, 0
    for match in re.finditer(r'<span id="eqtag-(\d+)"[^>]*></span>', body):
        index = int(match.group(1))
        if index >= len(tags):
            continue
        eq = BLOCK_MATH_RE.search(body, match.end())
        if not eq or eq.start() - match.end() > 1200:
            continue
        done += 1
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn">{eq.group(0)}<span class="eqno">({tags[index]})</span></div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), done


def number_equations(body: str) -> tuple[str, dict[str, str], int]:
    """Number labelled display equations and show the number beside each.

    Pandoc leaves ``\\eqref`` as a literal ``[eq:foo]`` and gives display maths no
    number at all, so the prose reads "the delayed logistic map [eq:delayed_logistic]".
    Number them here and hand back the mapping so the references can be rewritten.
    """
    numbers: dict[str, str] = {}
    count = 0

    def tag(match: re.Match[str]) -> str:
        nonlocal count
        label = match.group(2)
        if label.startswith("tab:"):
            return match.group(0)
        count += 1
        numbers[label] = str(count)
        return f'<span id="{label}" data-eqno="{count}"></span>'

    body = HYPERTARGET_RE.sub(tag, body)

    # Attach the number to the display equation that follows each anchor.
    out, pos = [], 0
    for m in re.finditer(r'<span id="((?:eq):[^"]+)" data-eqno="(\d+)"></span>', body):
        eq = BLOCK_MATH_RE.search(body, m.end())
        if not eq or eq.start() - m.end() > 1200:
            continue
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn">{eq.group(0)}<span class="eqno">({m.group(2)})</span></div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), numbers, count


def anchor_unnumbered_equations(body: str) -> tuple[str, int]:
    """Give the display equations nobody numbered a stable anchor.

    Run this after ``apply_eq_tags`` and ``number_equations``, so the only
    block maths left bare are the ones the manuscript never labelled. Those get
    an id -- so a reader can link to one and the page can preview it -- but
    deliberately no visible number: numbering them would shift every subsequent
    equation number and the web version would stop agreeing with the source.

    The ids are part of the page's public surface once a reader has shared a
    link, so they are positional and stable rather than content-derived.
    """
    out, pos, count = [], 0, 0
    for eq in BLOCK_MATH_RE.finditer(body):
        # Anything already wrapped carries its number and its own anchor.
        if body.rfind('<div class="eqn">', 0, eq.start()) > body.rfind("</div>", 0, eq.start()):
            continue
        count += 1
        out.append(body[pos : eq.start()])
        out.append(f'<div class="eqn eqn-bare" id="eq-u{count}">{eq.group(0)}</div>')
        pos = eq.end()
    out.append(body[pos:])
    return "".join(out), count


def number_tables(body: str) -> tuple[str, dict[str, str]]:
    """Number the tables in reading order so table references resolve."""
    numbers: dict[str, str] = {}
    count = 0

    def tag(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        numbers[match.group(2)] = str(count)
        return match.group(0)

    HYPERTARGET_RE.sub(lambda m: tag(m) if m.group(2).startswith("tab:") else m.group(0), body)
    return body, numbers


FIGREF_RE = re.compile(
    r'(<a href="#(fig:[^"]+)"[^>]*data-reference-type="ref"[^>]*>)([^<]*)</a>'
)


def renumber_figure_refs(body: str) -> tuple[str, int, int]:
    """Rewrite every figure reference to the number its figure actually shows.

    Pandoc resolved these against the *manuscript's* figure order at import
    time, but this page numbers figures in *page* order and deliberately moves
    the programme-arc figure to the end -- which slid every other figure by one.
    The result was 35 references that named a figure and linked to its
    neighbour.

    Equations and tables already avoid this because ``resolve_refs`` rewrites
    them from the numbers this build assigned. Figures were simply never added
    to that map, so nothing compared the two. Deriving the text from
    ``data-fignum`` here means the caption and the sentence cannot disagree
    again, whatever order the page puts figures in.
    """
    # The id and data-fignum attributes appear in either order on the tag, so
    # read them per figure rather than assuming one regex ordering.
    shown: dict[str, str] = {}
    for tag in re.findall(r"<figure[^>]*>", body):
        num = re.search(r'data-fignum="(\d+)"', tag)
        fid = re.search(r'id="(fig:[^"]+)"', tag)
        if num and fid:
            shown[fid.group(1)] = num.group(1)

    changed = missing = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed, missing
        open_tag, target, text = match.group(1), match.group(2), match.group(3)
        number = shown.get(target)
        if number is None:
            missing += 1
            return match.group(0)
        if text.strip() != number:
            changed += 1
        return f"{open_tag}{number}</a>"

    return FIGREF_RE.sub(repl, body), changed, missing


def resolve_refs(body: str, numbers: dict[str, str]) -> tuple[str, int]:
    """Replace literal ``[eq:foo]`` reference text with the real number."""
    fixed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal fixed
        open_tag, target = match.group(1), match.group(2)
        number = numbers.get(target)
        if not number:
            return match.group(0)
        fixed += 1
        label = f"({number})" if target.startswith("eq:") else number
        return f"{open_tag}{label}</a>"

    return XREF_RE.sub(repl, body), fixed


# Transcribed verbatim from the manuscript's tabularx overview figure, which
# pandoc cannot convert (it emits the raw cell separators as prose). Content is
# the manuscript's, not invented; only the presentation is ours.
PROGRAM_ARC = (
    ("1982", "Circle map", "phase locking"),
    ("1983\u201384", "Torus instabilities", "oscillation, doubling"),
    ("1983", "Coupled maps", "symmetry breaking"),
    ("1985", "CML", "spatial extension"),
    ("1989\u201390", "GCM", "mean-field"),
    ("1994\u201398", "Milnor / biology", "applications"),
)


def program_arc(fig_id: str, caption: str, number: int) -> str:
    """Render the research-arc overview as a timeline rather than a table."""
    cells = "".join(
        f'<li style="--i:{i}"><span class="yr">{year}</span>'
        f'<span class="topic">{topic}</span><span class="mech">{mech}</span></li>'
        for i, (year, topic, mech) in enumerate(PROGRAM_ARC)
    )
    return (
        f'<figure id="{fig_id}" class="arc" data-fignum="{number}">'
        f'<ol class="arc-track">{cells}</ol>'
        f'<figcaption><span class="num">Figure {number}.</span> {caption}</figcaption>'
        f"</figure>"
    )


OPACITY_RULE = re.compile(r"([^{}]+)\{([^{}]*opacity:\s*([01])[^{}]*)\}")


def specificity(selector: str) -> tuple[int, int, int]:
    """Rough CSS specificity: (ids, classes, elements)."""
    sel = selector.strip()
    return (
        sel.count("#"),
        sel.count(".") + sel.count("[") + sel.count(":"),
        len(re.findall(r"(?:^|[\s>+~])[a-z]+", sel)),
    )


def check_reveal_rules(css: str) -> list[str]:
    """Fail the build if something is hidden with no stronger rule to show it.

    A rule that sets ``opacity:0`` needs a more specific rule setting it back to
    1, or the element never appears. Getting this wrong blanks the page while
    every other check still passes, so it is verified rather than trusted.

    Hiders revealed by a CSS animation, and keyframe steps themselves, are not
    the failure this is looking for.
    """
    KEYFRAME_STEP = re.compile(r"^(from|to|\d+%)$")
    hiders: list[tuple[str, tuple[int, int, int]]] = []
    showers: list[tuple[str, tuple[int, int, int]]] = []

    for match in OPACITY_RULE.finditer(css):
        declarations = match.group(2)
        for selector in match.group(1).split(","):
            selector = selector.strip().rsplit("}", 1)[-1].strip()
            if not selector or selector.startswith("@") or KEYFRAME_STEP.match(selector):
                continue
            if match.group(3) == "1":
                showers.append((selector, specificity(selector)))
            elif "animation" not in declarations:
                hiders.append((selector, specificity(selector)))

    problems = []
    for selector, spec in hiders:
        # the rightmost simple selector is what the reveal rule must also target
        key = re.split(r"[\s>+~]", selector.split(":")[0])[-1]
        if not key:
            continue
        if any(key in other and other_spec > spec for other, other_spec in showers):
            continue
        problems.append(selector)
    return problems


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


def transform(
    body: str, eq_tags: list[str] | None = None
) -> tuple[str, list[tuple[int, str, str, str]]]:
    """Rewrite pandoc output into the page's own figure, table and heading shapes."""
    dims = thumb_dims()
    stats: dict[str, object] = {"interactive": 0, "static": 0, "arc": 0, "dropped": []}

    def do_figure(match: re.Match[str]) -> str:
        block = match.group(0)
        fid = re.search(r'<figure id="([^"]+)"', block)
        fig_id = fid.group(1) if fid else ""
        img = IMG_RE.search(block)
        cap = CAP_RE.search(block)
        caption = cap.group(1).strip() if cap else ""
        if not img:
            # pandoc cannot convert the tabularx overview; it emits raw cell
            # separators as prose. Render it as a purpose-built timeline instead.
            if fig_id == "fig:program_map":
                stats["arc"] += 1
                return PROGRAM_ARC_SLOT
            stats["dropped"].append(fig_id or "(unlabelled)")
            return ""
        section, name = img.group(1), img.group(2)
        if (SITE / "data" / section / f"{name}.json").exists():
            stats["interactive"] += 1
        else:
            stats["static"] += 1
        return figure_block(fig_id, section, name, caption, dims)

    body = FIG_RE.sub(do_figure, body)
    body = TABLE_RE.sub(lambda m: f'<div class="table-wrap">{m.group(0)}</div>', body)

    # Number the figures in reading order, so a reader can name what they are
    # looking at. Done after the figure rewrite so the count matches the page.
    fig_no = 0

    def number_figure(match: re.Match[str]) -> str:
        nonlocal fig_no
        block = match.group(0)
        fig_no += 1
        return block.replace(
            "<figcaption>", f'<figcaption><span class="num">Figure {fig_no}.</span> ', 1
        ).replace("<figure ", f'<figure data-fignum="{fig_no}" ', 1)

    body = FIG_RE.sub(number_figure, body)
    if PROGRAM_ARC_SLOT in body:
        arc_caption = (
            "Arc of Kaneko's research programme, from one-dimensional phase-locking "
            "analysis (1982) through coupled map lattices and globally coupled maps "
            "to biological applications (1990s). The columns mark the extension of "
            "the framework to higher complexity. Abbreviations: CML, coupled map "
            "lattice; GCM, globally coupled map."
        )
        fig_no += 1
        body = body.replace(
            PROGRAM_ARC_SLOT, program_arc("fig:program_map", arc_caption, fig_no), 1
        )

    # Keep pandoc's ids verbatim: they are the manuscript's own \label anchors,
    # and every internal cross-reference in the text points at them.
    body = SEC_RE.sub(lambda m: f'<section id="{m.group(1)}" class="{m.group(2)} reveal">', body)

    # Number sections from the heading tree, skipping the ones the manuscript
    # marks unnumbered, then demote level1 -> h2 and level2 -> h3 so the
    # shell's type scale applies.
    nav: list[tuple[int, str, str, str]] = []
    top = sub = 0
    for m in re.finditer(r'<section id="([^"]+)" class="([^"]*)"', body):
        sec_id, cls = m.group(1), m.group(2)
        tail = body[m.end() : m.end() + 4000]
        h = HEAD_RE.search(tail)
        if not h:
            continue
        level = 1 if "level1" in cls else 2
        number = ""
        if "unnumbered" not in cls:
            if level == 1:
                top += 1
                sub = 0
                number = str(top)
            else:
                sub += 1
                number = f"{top}.{sub}"
        nav.append((level, sec_id, strip_tags(h.group(3)), number))

    numbers = iter(n for *_, n in nav)

    def demote(match: re.Match[str]) -> str:
        tag, attrs, text = match.group(1), match.group(2), match.group(3)
        new = "h2" if tag == "h1" else "h3"
        number = next(numbers, "")
        label = f'<span class="secno">{number}</span> ' if number else ""
        return f"<{new}{attrs}>{label}{text}</{new}>"

    body = HEAD_RE.sub(demote, body)
    body, n_tagged = apply_eq_tags(body, eq_tags or [])
    body, eq_numbers, n_eq = number_equations(body)
    body, n_bare = anchor_unnumbered_equations(body)
    body, tab_numbers = number_tables(body)
    body, n_fixed = resolve_refs(body, {**eq_numbers, **tab_numbers})
    body, n_figref, n_figref_missing = renumber_figure_refs(body)
    # The raw-LaTeX annotation duplicates every equation as plain text when a
    # browser does not render MathML; drop it rather than rely on a UA stylesheet.
    n_ann = len(ANNOTATION_RE.findall(body))
    body = ANNOTATION_RE.sub("", body)
    body, folded = fold_back_matter(body)
    print(
        f"  figures: {stats['interactive']} interactive, {stats['static']} static "
        f"(numbered 1-{fig_no})"
    )
    if stats["dropped"]:
        print(f"  DROPPED (pandoc could not convert): {', '.join(stats['dropped'])}")
    print(f"  sections numbered: {sum(1 for *_, n in nav if n)} of {len(nav)}")
    print(
        f"  equations numbered: {n_eq}; tagged: {n_tagged}; "
        f"anchored unnumbered: {n_bare}; references resolved: {n_fixed}"
    )
    print(f"  figure references renumbered to match captions: {n_figref}")
    if n_figref_missing:
        print(f"  WARNING: {n_figref_missing} figure references point at no figure on the page")
    print(f"  latex annotations stripped: {n_ann}")
    print(f"  back matter folded: {folded}")
    return body, nav


# Reference material the reader should be able to skip: it interrupts the
# argument but must stay on the page and stay linkable.
BACK_MATTER = ("sec:glossary", "app:provenance", "app:repro_index", "app:assumptions")


def fold_back_matter(body: str) -> tuple[str, int]:
    """Wrap reference sections in a collapsed ``<details>``.

    The section keeps its id and stays in the document, so every cross-reference
    into it still resolves; browsers open a closed ``<details>`` automatically
    when a link targets something inside it.
    """
    folded = 0
    for sec_id in BACK_MATTER:
        pattern = re.compile(
            rf'(<section id="{re.escape(sec_id)}"[^>]*>)\s*<h2([^>]*)>(.*?)</h2>', re.S
        )
        match = pattern.search(body)
        if not match:
            continue
        folded += 1
        body = pattern.sub(
            lambda m: (
                f"{m.group(1)}<details class='backmatter'>"
                f"<summary><h2{m.group(2)}>{m.group(3)}</h2></summary>"
            ),
            body,
            count=1,
        )
        end = body.index("</section>", body.index(f'id="{sec_id}"'))
        body = body[:end] + "</details>" + body[end:]
    return body, folded


def build_nav(nav: list[tuple[int, str, str, str]]) -> str:
    """Nested contents: top-level entries always visible, children on demand."""
    out = ['<nav class="spine" aria-label="Contents"><ul class="toc">']
    open_sub = False
    for level, sec_id, title, number in nav:
        label = f"<i>{number}</i> {html.escape(title)}" if number else html.escape(title)
        if level == 1:
            if open_sub:
                out.append("</ul></li>")
                open_sub = False
            out.append(f'<li class="top"><a href="#{sec_id}">{label}</a><ul class="sub">')
            open_sub = True
        else:
            if not open_sub:
                out.append('<li class="top"><ul class="sub">')
                open_sub = True
            out.append(f'<li><a href="#{sec_id}">{label}</a></li>')
    if open_sub:
        out.append("</ul></li>")
    out.append("</ul></nav>")
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
        raw, eq_tags = run_pandoc(args.manuscript)
        BODY.write_text(raw, encoding="utf-8")
        meta = extract_meta(tex)
        meta["eq_tags"] = eq_tags
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

    hidden = check_reveal_rules(CSS)
    if hidden:
        raise SystemExit(
            "CSS would hide these with no stronger rule to reveal them, which "
            "blanks the page:\n  " + "\n  ".join(hidden)
        )

    body, nav = transform(BODY.read_text(encoding="utf-8"), meta.get("eq_tags"))
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
