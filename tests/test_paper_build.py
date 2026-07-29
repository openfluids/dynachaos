"""Checks on the page assembly in ``scripts/build_paper.py``.

These exist because a whole class of defect here is invisible: the build
succeeds, the page renders, every link resolves -- and the text still names the
wrong figure. 52 references pointed at a figure one place off from the one they
named, and survived every build, because nothing compared the number in the
sentence against the number in the caption.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_paper.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_paper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def _ref(target: str, text: str) -> str:
    return f'<a href="#{target}" data-reference-type="ref" data-reference="{target}">{text}</a>'


def test_figure_refs_take_the_number_the_caption_shows():
    module = _load()
    # The programme-arc figure is numbered last on the page but was numbered
    # first in the manuscript -- the exact shift that caused the original bug.
    body = (
        _ref("fig:arnold", "2")
        + _ref("fig:arc", "1")
        + '<figure data-fignum="1" id="fig:arnold"></figure>'
        + '<figure id="fig:arc" class="arc" data-fignum="37"></figure>'
    )

    out, changed, missing = module.renumber_figure_refs(body)

    assert changed == 2
    assert missing == 0
    assert _ref("fig:arnold", "1") in out
    assert _ref("fig:arc", "37") in out


def test_figure_refs_are_left_alone_when_they_already_agree():
    module = _load()
    body = _ref("fig:a", "4") + '<figure data-fignum="4" id="fig:a"></figure>'

    out, changed, missing = module.renumber_figure_refs(body)

    assert (changed, missing) == (0, 0)
    assert out == body


def test_a_reference_to_a_figure_that_is_not_on_the_page_is_reported():
    """Silently leaving it alone is right -- but it must be counted, so the
    build can say so rather than shipping a link to nothing."""
    module = _load()
    body = _ref("fig:absent", "9")

    out, changed, missing = module.renumber_figure_refs(body)

    assert (changed, missing) == (0, 1)
    assert out == body


def test_table_captions_get_the_same_number_the_ref_map_assigns():
    """Prose cites "Table N" via the numbers map ``number_tables`` returns; the
    caption must show the identical number or a reader sees a citation and a
    caption that disagree, the same bug the figure numbering test guards."""
    module = _load()
    body = (
        '<div id="tab:notation"></div><table id="tab:notation">'
        "<caption>Global notation.</caption></table>"
        '<div id="tab:atlas"></div><table id="tab:atlas">'
        "<caption>Mechanistic atlas of Kaneko-style systems.</caption></table>"
    )

    out, numbers = module.number_tables(body)

    assert numbers == {"tab:notation": "1", "tab:atlas": "2"}
    assert '<caption><span class="num">Table 1.</span> Global notation.' in out
    assert '<caption><span class="num">Table 2.</span> Mechanistic atlas' in out


def test_build_alt_text_strips_annotations_and_labels_and_cuts_at_a_sentence():
    """The alt builder must not speak MathML twice, must not leak pandoc's
    unresolved ``[eq:...]`` placeholders, and must never cut mid-word."""
    module = _load()
    caption = (
        "Lyapunov spectrum of the delayed logistic map "
        '<a href="#eq:delayed_logistic" data-reference-type="eqref" '
        'data-reference="eq:delayed_logistic">[eq:delayed_logistic]</a> at '
        '<math display="inline"><semantics><mrow><mi>&alpha;</mi><mo>=</mo>'
        '<mn>0.3</mn></mrow><annotation encoding="application/x-tex">'
        "\\alpha = 0.3</annotation></semantics></math> versus D. "
        "The vertical dashed line marks the onset of chaos in the sweep, "
        "well past the two hundred character mark this sentence is padded "
        "out to reach so the cutoff logic actually has to make a choice "
        "here between keeping the first sentence or the second one."
    )

    alt = module.build_alt_text(caption)

    assert "[eq:" not in alt
    assert "\\alpha" not in alt
    assert "annotation" not in alt
    # the doubled math text is gone; only the rendered symbol remains once
    assert alt.count("0.3") <= 1
    assert alt.endswith((".", "…", "!", "?"))
    assert not alt.endswith(("li", "trac", "wedg"))  # no mid-word cuts
    assert len(alt) <= 203  # ~200 chars plus a little slack for the ellipsis


def test_build_alt_text_keeps_a_short_caption_whole():
    module = _load()
    caption = "Devil's staircase of the circle map."

    alt = module.build_alt_text(caption)

    assert alt == "Devil's staircase of the circle map."


def test_search_index_covers_folded_details_and_strips_annotations():
    """The whole point of the index: a browser's own Ctrl+F cannot see inside
    a closed <details>, so the build-time index must -- and it must not speak
    an equation's raw-LaTeX annotation as if it were prose."""
    module = _load()
    body = (
        '<section id="sec:results"><h2>Results</h2>'
        '<details class="backmatter"><summary><h2>Provenance</h2></summary>'
        "<p>Every run is recomputed at "
        '<math display="inline"><semantics><mrow><mi>&alpha;</mi><mo>=</mo>'
        '<mn>0.3</mn></mrow><annotation encoding="application/x-tex">'
        "\\alpha = 0.3</annotation></semantics></math> from fresh seeds.</p>"
        "</details></section>"
    )

    units = module.build_search_index(body)
    para = next(u for u in units if "Every run" in u["text"])

    assert "\\alpha" not in para["text"]
    assert para["text"].count("0.3") == 1
    # the id resolves to the ancestor section's real anchor, not a made-up one
    assert para["id"] == "sec:results"
    # indexed under the nearest heading actually seen, inside the details
    assert para["section"] == "Provenance"
    headings = {u["text"] for u in units if u["tag"] in ("h2", "h3", "h4")}
    assert headings == {"Results", "Provenance"}


def test_search_index_json_is_embeddable_in_a_script_tag():
    module = _load()
    units = [{"id": "sec:a", "section": "A", "tag": "p", "text": "contains </script> literally"}]

    raw, truncated = module.search_index_json(units)
    parsed = json.loads(raw.replace("<\\/", "</"))

    assert truncated is False
    assert "</script" not in raw
    assert parsed[0]["text"] == "contains </script> literally"


def test_unnumbered_display_equations_get_an_anchor_but_no_number():
    """Numbering them would shift every subsequent equation number and the page
    would stop agreeing with the manuscript it reproduces."""
    module = _load()
    numbered = '<div class="eqn"><math display="block">a</math><span class="eqno">(1)</span></div>'
    body = numbered + '<p>text</p><math display="block">b</math>'

    out, count = module.anchor_unnumbered_equations(body)

    assert count == 1
    assert 'id="eq-u1"' in out
    # exactly one visible number, still the one that was already there
    assert len(re.findall(r'class="eqno"', out)) == 1
    assert numbered in out


def test_the_numbering_note_lands_before_the_reproduction_index():
    """The page renumbers figures relative to the manuscript; shipping without
    the disclosure note would leave that unexplained, so its absence must fail
    the build rather than pass silently."""
    module = _load()
    body = '<table id="tab:repro_index"><caption>Reproduction index.</caption></table>'

    out, _ = module.fold_back_matter(body)

    note_at = out.index("Figure numbers on this page")
    assert note_at < out.index('<table id="tab:repro_index"')


def test_a_missing_reproduction_index_fails_the_build():
    import pytest

    module = _load()
    with pytest.raises(SystemExit):
        module.fold_back_matter("<p>no table here</p>")
