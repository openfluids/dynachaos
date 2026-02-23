"""Section registry for dynachaos figure/data pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dynachaos.io.paths import section_dir


@dataclass(frozen=True)
class SectionSpec:
    """Pipeline specification for one paper section."""

    section_id: str
    modules: tuple[str, ...]
    cache_files: tuple[str, ...]
    output_files: tuple[str, ...]

    def cache_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.cache_files)

    def output_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.output_files)


SECTION_ORDER = (
    "sec02_circle_map",
    "sec03_transition",
    "sec04_doubling",
    "sec05_oscillation",
    "sec06_three_torus",
    "sec07_fractalization",
    "sec08_sti",
    "sec09_pattern",
    "sec10_gcm",
    "sec11_diagnostics",
)


SECTION_SPECS = {
    "sec02_circle_map": SectionSpec(
        section_id="sec02_circle_map",
        modules=("dynachaos.maps.circle_map", "dynachaos.maps.arnold_tongues"),
        cache_files=("devils_staircase.npz", "arnold_tongues.npz", "staircase_zoom.npz"),
        output_files=(
            "devils_staircase.npz",
            "devils_staircase.png",
            "arnold_tongues.npz",
            "arnold_tongues.png",
            "staircase_zoom.npz",
            "staircase_zoom.png",
        ),
    ),
    "sec03_transition": SectionSpec(
        section_id="sec03_transition",
        modules=("dynachaos.maps.coupled_logistic",),
        cache_files=("phase_diagram.npz", "attractors.npz", "basins.npz"),
        output_files=(
            "phase_diagram.npz",
            "attractors.npz",
            "basins.npz",
            "phase_diagram.png",
            "attractors.png",
            "basins.png",
        ),
    ),
    "sec04_doubling": SectionSpec(
        section_id="sec04_doubling",
        modules=("dynachaos.maps.torus_doubling",),
        cache_files=("map_I_attractors.npz", "map_IV_attractors.npz", "map_IV_lyapunov.npz"),
        output_files=(
            "map_I_attractors.npz",
            "map_IV_attractors.npz",
            "map_IV_lyapunov.npz",
            "map_I_attractors.png",
            "map_IV_attractors.png",
            "map_IV_lyapunov.png",
        ),
    ),
    "sec05_oscillation": SectionSpec(
        section_id="sec05_oscillation",
        modules=("dynachaos.maps.delayed_logistic",),
        cache_files=("attractors.npz", "lyapunov_vs_D.npz", "locking_sequence.npz"),
        output_files=(
            "attractors.npz",
            "lyapunov_vs_D.npz",
            "locking_sequence.npz",
            "attractors.png",
            "lyapunov_vs_D.png",
            "locking_sequence.png",
        ),
    ),
    "sec06_three_torus": SectionSpec(
        section_id="sec06_three_torus",
        modules=("dynachaos.maps.coupled_delayed", "dynachaos.maps.modulated_circle"),
        cache_files=("lyapunov_vs_DB.npz", "xz_projections.npz", "double_staircase.npz"),
        output_files=(
            "lyapunov_vs_DB.npz",
            "xz_projections.npz",
            "double_staircase.npz",
            "lyapunov_vs_DB.png",
            "xz_projections.png",
            "double_staircase.png",
            "double_staircase_zoom.png",
        ),
    ),
    "sec07_fractalization": SectionSpec(
        section_id="sec07_fractalization",
        modules=("dynachaos.maps.fractalization",),
        cache_files=("fractal_attractors.npz", "correlation_dimension.npz"),
        output_files=(
            "fractal_attractors.npz",
            "correlation_dimension.npz",
            "fractal_attractors.png",
            "correlation_dimension.png",
        ),
    ),
    "sec08_sti": SectionSpec(
        section_id="sec08_sti",
        modules=(
            "dynachaos.cml.spatiotemporal",
            "dynachaos.cml.comoving_figure",
            "dynachaos.cml.correlation_figure",
        ),
        cache_files=(
            "spacetime_diagrams.npz",
            "comoving_lyapunov.npz",
            "correlation_decay.npz",
        ),
        output_files=(
            "spacetime_diagrams.npz",
            "spacetime_diagrams.png",
            "comoving_lyapunov.npz",
            "comoving_lyapunov.png",
            "correlation_decay.npz",
            "correlation_decay.png",
        ),
    ),
    "sec09_pattern": SectionSpec(
        section_id="sec09_pattern",
        modules=("dynachaos.cml.pattern_dynamics",),
        cache_files=("phase_diagram.npz", "space_amplitude.npz"),
        output_files=(
            "phase_diagram.npz",
            "space_amplitude.npz",
            "phase_diagram.png",
            "space_amplitude.png",
        ),
    ),
    "sec10_gcm": SectionSpec(
        section_id="sec10_gcm",
        modules=(
            "dynachaos.cml.globally_coupled",
            "dynachaos.cml.gcm_clusters",
        ),
        cache_files=(
            "gcm_results.npz",
            "gcm_clusters.npz",
            "collective_lyapunov.npz",
        ),
        output_files=(
            "gcm_results.npz",
            "gcm_msd.png",
            "gcm_distribution.png",
            "gcm_clusters.npz",
            "gcm_clusters.png",
            "collective_lyapunov.npz",
            "collective_lyapunov.png",
        ),
    ),
    "sec11_diagnostics": SectionSpec(
        section_id="sec11_diagnostics",
        modules=("dynachaos.diagnostics.compare_all",),
        cache_files=(
            "test01_sweep.npz",
            "sali_comparison.npz",
            "permutation_entropy.npz",
            "complexity_entropy_plane.npz",
            "rqa_measures.npz",
        ),
        output_files=(
            "test01_sweep.npz",
            "sali_comparison.npz",
            "permutation_entropy.npz",
            "complexity_entropy_plane.npz",
            "rqa_measures.npz",
            "test01_sweep.png",
            "sali_comparison.png",
            "permutation_entropy.png",
            "complexity_entropy_plane.png",
            "rqa_measures.png",
        ),
    ),
}


def list_sections() -> tuple[str, ...]:
    """Return section IDs in paper order."""
    return SECTION_ORDER


def get_section(section_id: str) -> SectionSpec:
    """Return a section spec or raise a KeyError."""
    return SECTION_SPECS[section_id]
