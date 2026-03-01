import dynachaos
from dynachaos.io.paths import output_root, section_dir


def test_package_imports():
    assert hasattr(dynachaos, "__all__")


def test_submodule_imports():
    from dynachaos.maps.circle_map import circle_map
    from dynachaos.diagnostics.permutation import permutation_entropy

    assert callable(circle_map)
    assert callable(permutation_entropy)


def test_output_path_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert output_root() == tmp_path / "figures"
    assert section_dir("sec02_circle_map") == tmp_path / "figures" / "sec02_circle_map"


# ---------------------------------------------------------------------------
# Re-export smoke tests
# ---------------------------------------------------------------------------

def test_maps_reexports():
    from dynachaos.maps import logistic, delayed_logistic, circle_map
    assert all(callable(f) for f in [logistic, delayed_logistic, circle_map])


def test_diagnostics_reexports():
    from dynachaos.diagnostics import correlation_integral, lyapunov_exponent_1d
    assert all(callable(f) for f in [correlation_integral, lyapunov_exponent_1d])


def test_cml_reexports():
    from dynachaos.cml import cml_step, gcm_step
    assert all(callable(f) for f in [cml_step, gcm_step])


def test_safe_load_reexport():
    from dynachaos.io import safe_load
    assert callable(safe_load)
