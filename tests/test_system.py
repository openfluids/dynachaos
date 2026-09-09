"""Coverage test for dynachaos.utils.system.

The `resource` module is POSIX-only; the None fallback (system.py:14) is
normally only reached on Windows, so it is forced here directly.
"""

from dynachaos.utils import system


def test_get_rss_mb_returns_zero_when_resource_module_unavailable(monkeypatch):
    monkeypatch.setattr(system, "resource", None)

    assert system.get_rss_mb() == 0.0


def test_get_rss_mb_returns_a_positive_finite_value_on_this_platform():
    rss = system.get_rss_mb()

    assert rss > 0.0
