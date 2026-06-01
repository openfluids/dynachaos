"""Modern diagnostic tools for dynamical systems."""

from dynachaos.diagnostics.correlation import (
    correlation_dimension,
    correlation_integral,
    fit_power_law_loglog,
)
from dynachaos.diagnostics.embedding import (
    average_mutual_information,
    cao_method,
    false_nearest_neighbors,
    optimal_delay,
    optimal_dimension,
    select_dimension_cao,
)
from dynachaos.diagnostics.entropy import (
    approximate_entropy,
    fuzzy_entropy,
    multiscale_entropy,
    sample_entropy,
)
from dynachaos.diagnostics.intermittency import (
    LaminarLengthDistribution,
    detect_laminar_phases,
    laminar_length_distribution,
)
from dynachaos.diagnostics.lyapunov import (
    flow_lyapunov_spectrum,
    lyapunov_exponent_1d,
    lyapunov_spectrum,
)
from dynachaos.diagnostics.multifractal import local_multifractality, multifractal_spectrum
from dynachaos.diagnostics.permutation import complexity_entropy, permutation_entropy
from dynachaos.diagnostics.poincare import poincare_section
from dynachaos.diagnostics.recurrence import (
    LaminarLengthsResult,
    embed_time_delay,
    laminar_lengths,
    recurrence_matrix,
    rqa,
    rqa_from_trajectory,
)
from dynachaos.diagnostics.sali_gali import gali, sali
from dynachaos.diagnostics.zero_one_test import zero_one_statistic

__all__ = [
    "approximate_entropy",
    "average_mutual_information",
    "cao_method",
    "complexity_entropy",
    "correlation_dimension",
    "correlation_integral",
    "embed_time_delay",
    "false_nearest_neighbors",
    "fit_power_law_loglog",
    "flow_lyapunov_spectrum",
    "fuzzy_entropy",
    "gali",
    "LaminarLengthsResult",
    "LaminarLengthDistribution",
    "detect_laminar_phases",
    "laminar_length_distribution",
    "laminar_lengths",
    "lyapunov_exponent_1d",
    "lyapunov_spectrum",
    "local_multifractality",
    "multiscale_entropy",
    "multifractal_spectrum",
    "optimal_delay",
    "optimal_dimension",
    "permutation_entropy",
    "poincare_section",
    "recurrence_matrix",
    "rqa",
    "rqa_from_trajectory",
    "sali",
    "sample_entropy",
    "select_dimension_cao",
    "zero_one_statistic",
]
