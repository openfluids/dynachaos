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
