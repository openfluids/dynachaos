"""Section registry for dynachaos figure/data pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dynachaos.io.paths import section_dir


@dataclass(frozen=True)
class NpzContract:
    """Required keys for one generated ``.npz`` artifact."""

    file_name: str
    required_keys: tuple[str, ...]


@dataclass(frozen=True)
class SectionSpec:
    """Pipeline specification for one paper section."""

    section_id: str
    modules: tuple[str, ...]
    cache_files: tuple[str, ...]
    output_files: tuple[str, ...]
    npz_contracts: tuple[NpzContract, ...] = ()

    def cache_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.cache_files)

    def output_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.output_files)

    def required_npz_keys(self, file_name: str) -> tuple[str, ...]:
        """Return required keys for an artifact, or an empty tuple if unconstrained."""
        for contract in self.npz_contracts:
            if contract.file_name == file_name:
                return contract.required_keys
        return ()


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
    "sec12_intermittency",
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
        npz_contracts=(
            NpzContract("devils_staircase.npz", ("A", "rho", "lam")),
            NpzContract("arnold_tongues.npz", ("Omega", "K", "rho")),
            NpzContract("staircase_zoom.npz", ("A", "rho")),
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
        npz_contracts=(
            NpzContract("phase_diagram.npz", ("A", "D", "asym", "lyap", "schema_version")),
            NpzContract(
                "attractors.npz",
                ("A_values", "labels", "initial_states", "x_limits", "y_limits", "D", "schema_version"),
            ),
            NpzContract("basins.npz", ("x", "y", "basin", "A", "D")),
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
        npz_contracts=(
            NpzContract("map_I_attractors.npz", ("D_values",)),
            NpzContract("map_IV_attractors.npz", ("D_values",)),
            NpzContract("map_IV_lyapunov.npz", ("D", "spectra")),
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
        npz_contracts=(
            NpzContract("attractors.npz", ("D_values", "A")),
            NpzContract("lyapunov_vs_D.npz", ("D", "spectra")),
            NpzContract("locking_sequence.npz", ("D_values", "A")),
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
        npz_contracts=(
            NpzContract("lyapunov_vs_DB.npz", ("DB", "eps_values")),
            NpzContract("xz_projections.npz", ("DB_values", "labels", "render_modes", "schema_version")),
            NpzContract("double_staircase.npz", ("D", "rho_theta", "rho_phi", "A", "C", "eps")),
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
        npz_contracts=(
            NpzContract("fractal_attractors.npz", ("D_values", "A")),
            NpzContract("correlation_dimension.npz", ("D", "D2", "A")),
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
        npz_contracts=(
            NpzContract("spacetime_diagrams.npz", ("A_eps_0.06", "B_eps_0.02", "C_eps_0.16")),
            NpzContract("comoving_lyapunov.npz", ("v_values", "a_values", "eps", "N")),
            NpzContract("correlation_decay.npz", ("a_corr", "r_vals", "all_corr", "xi_values")),
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
        npz_contracts=(
            NpzContract("phase_diagram.npz", ("a", "eps", "lam")),
            NpzContract("space_amplitude.npz", ("params", "schema_version")),
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
        npz_contracts=(
            NpzContract("gcm_results.npz", ("N_values", "a", "eps", "N_grid", "msd_grid")),
            NpzContract("gcm_clusters.npz", ("cluster_labels", "x_record", "a", "eps", "N")),
            NpzContract("collective_lyapunov.npz", ("a_values", "lyap_c", "eps", "N")),
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
        npz_contracts=(
            NpzContract("test01_sweep.npz", ("a", "K")),
            NpzContract("sali_comparison.npz", ("DB_values",)),
            NpzContract("permutation_entropy.npz", ("a", "H_logistic", "D", "H_delayed")),
            NpzContract(
                "complexity_entropy_plane.npz",
                ("a", "H_logistic", "C_logistic", "D", "H_delayed", "C_delayed"),
            ),
            NpzContract("rqa_measures.npz", ("D", "RR", "DET", "LAM", "ENTR")),
        ),
    ),
    "sec12_intermittency": SectionSpec(
        section_id="sec12_intermittency",
        modules=("dynachaos.diagnostics.intermittency_figure",),
        cache_files=("intermittency_diagnostics.npz",),
        output_files=(
            "intermittency_diagnostics.npz",
            "intermittency_diagnostics.png",
        ),
        npz_contracts=(
            NpzContract(
                "intermittency_diagnostics.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "type_i_laminar_lengths",
                    "type_i_tail_alpha",
                    "type_i_vuong_z",
                    "on_off_burst_alpha",
                    "on_off_symmetry_p",
                    "lorenz_section_points",
                    "lorenz_channel_slope",
                ),
            ),
        ),
    ),
}


def list_sections() -> tuple[str, ...]:
    """Return section IDs in paper order."""
    return SECTION_ORDER


def get_section(section_id: str) -> SectionSpec:
    """Return a section spec or raise a KeyError."""
    return SECTION_SPECS[section_id]
