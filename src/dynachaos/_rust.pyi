"""Type stubs for the optional Rust extension module."""

import numpy as np
import numpy.typing as npt

def diagonal_lines(
    r: npt.NDArray[np.bool_],
    l_min: int = 2,
) -> npt.NDArray[np.int64]: ...
def vertical_lines(
    r: npt.NDArray[np.bool_],
    v_min: int = 2,
) -> npt.NDArray[np.int64]: ...
def ordinal_distribution(
    x: npt.NDArray[np.float64],
    d: int = 5,
    tau: int = 1,
) -> tuple[npt.NDArray[np.int64], int]: ...
def ami_histogram(
    x: npt.NDArray[np.float64],
    tau_max: int,
    n_bins: int = 64,
) -> npt.NDArray[np.float64]: ...
def select_dimension_cao(
    e1: npt.NDArray[np.float64],
    near_one_lower: float = 0.95,
    near_one_upper: float = 1.05,
    saturation_tol: float = 0.02,
    plateau_span: int = 3,
    smoothing_window: int = 1,
    min_dim: int = 2,
    max_dim: int | None = None,
) -> int: ...
def correlation_counts(
    traj: npt.NDArray[np.float64],
    r_values: npt.NDArray[np.float64],
    theiler_window: int = 0,
    use_chebyshev: bool = True,
) -> npt.NDArray[np.int64]: ...
def apen_counts(
    traj: npt.NDArray[np.float64],
    r: float,
) -> npt.NDArray[np.int64]: ...
def fuzzy_entropy_sum(
    traj: npt.NDArray[np.float64],
    r: float,
    n: int,
    theiler_window: int = 0,
) -> float: ...
def multifractal_moments(
    field: npt.NDArray[np.float64],
    box_sizes: npt.NDArray[np.int64],
    q_values: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
