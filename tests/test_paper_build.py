"""Checks on the page assembly in ``scripts/build_paper.py``.

These exist because a whole class of defect here is invisible: the build
succeeds, the page renders, every link resolves -- and the text still names the
wrong figure. 52 references pointed at a figure one place off from the one they
named, and survived every build, because nothing compared the number in the
sentence against the number in the caption.
"""

import importlib.util
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
