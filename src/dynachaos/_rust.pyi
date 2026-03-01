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

def cao_statistic(
    x: npt.NDArray[np.float64],
    tau: int,
    d_max: int,
    theiler_window: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def fnn_statistic(
    x: npt.NDArray[np.float64],
    tau: int,
    d_max: int,
    r_tol: float = 15.0,
    a_tol: float = 2.0,
    theiler_window: int = 0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...

def correlation_counts(
    traj: npt.NDArray[np.float64],
    r_values: npt.NDArray[np.float64],
    theiler_window: int = 0,
    use_chebyshev: bool = True,
) -> npt.NDArray[np.int64]: ...
