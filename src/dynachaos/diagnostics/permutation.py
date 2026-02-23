"""Permutation entropy and complexity–entropy planes.

Permutation entropy (Bandt & Pompe 2002) quantifies the complexity of a
time series by examining the relative ordering of consecutive values.
For embedding dimension d, each window of d consecutive values defines
an ordinal pattern (permutation of {0, 1, ..., d-1}).

    H_PE = -Σ p_π log(p_π) / log(d!)

where the sum is over all d! possible permutations and p_π is the
relative frequency of each pattern.

    H_PE = 0    →  completely deterministic (single pattern)
    H_PE = 1    →  completely random (uniform distribution)

The complexity–entropy (CH) plane combines H_PE with the Jensen-Shannon
statistical complexity C_JS (Rosso et al. 2007), placing different
dynamics in characteristic regions of the plane.

References
----------
Bandt, C. & Pompe, B. (2002) "Permutation entropy: A natural complexity
  measure for time series", Phys. Rev. Lett., 88(17), 174102.
Rosso, O.A. et al. (2007) "Distinguishing noise from chaos", Phys. Rev.
  Lett., 99(15), 154102.

Usage
-----
    from dynachaos.diagnostics.permutation import (
        permutation_entropy, complexity_entropy
    )

    H = permutation_entropy(time_series, d=5)
    H, C = complexity_entropy(time_series, d=5)
"""

import numpy as np
from math import factorial


def _ordinal_pattern(window):
    """Return the ordinal pattern (rank permutation) of a window."""
    return tuple(np.argsort(window))


def ordinal_distribution(x, d=5, tau=1):
    """Compute the probability distribution of ordinal patterns.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    d : int
        Embedding dimension (pattern length). Typically 3-7.
        d! must not exceed N - (d-1)*tau.
    tau : int
        Time delay between successive elements.

    Returns
    -------
    patterns : dict
        Mapping from ordinal pattern (tuple) to relative frequency.
    n_patterns : int
        Total number of windows analysed.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    n_windows = N - (d - 1) * tau

    counts = {}
    for i in range(n_windows):
        indices = [i + j * tau for j in range(d)]
        window = x[indices]
        pattern = _ordinal_pattern(window)
        counts[pattern] = counts.get(pattern, 0) + 1

    # Normalise
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    return probs, total


def permutation_entropy(x, d=5, tau=1, normalise=True):
    """Compute the permutation entropy of a time series.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    d : int
        Embedding dimension.
    tau : int
        Time delay.
    normalise : bool
        If True, normalise by log(d!) so that H ∈ [0, 1].

    Returns
    -------
    H : float
        The permutation entropy.
    """
    probs, _ = ordinal_distribution(x, d, tau)

    # Shannon entropy
    H = 0.0
    for p in probs.values():
        if p > 0:
            H -= p * np.log(p)

    if normalise:
        H_max = np.log(factorial(d))
        if H_max > 0:
            H /= H_max

    return float(H)


def _jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence between two probability distributions.

    Parameters are arrays (aligned to the same set of outcomes).
    """
    m = 0.5 * (p + q)
    # KL(p || m) + KL(q || m), masking zeros to avoid log(0)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / m[mask_p]))
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / m[mask_q]))
    return 0.5 * (kl_pm + kl_qm)


def complexity_entropy(x, d=5, tau=1):
    """Compute the permutation entropy and Jensen-Shannon complexity.

    The pair (H, C) locates the dynamics in the complexity-entropy plane
    (Rosso et al. 2007).

    Parameters
    ----------
    x : array_like
        Scalar time series.
    d : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    H : float
        Normalised permutation entropy, ∈ [0, 1].
    C : float
        Jensen-Shannon statistical complexity, C = Q_JS · H_S.
    """
    probs_dict, _ = ordinal_distribution(x, d, tau)

    n_perm = factorial(d)

    # Build probability vector aligned to all possible permutations
    # (include zero-probability patterns)
    p = np.zeros(n_perm)
    # Map each observed pattern to an index
    from itertools import permutations as iter_perms
    all_perms = list(iter_perms(range(d)))
    perm_to_idx = {perm: i for i, perm in enumerate(all_perms)}

    for pattern, prob in probs_dict.items():
        idx = perm_to_idx[pattern]
        p[idx] = prob

    # Uniform distribution
    q = np.ones(n_perm) / n_perm

    # Normalised Shannon entropy
    H_max = np.log(n_perm)
    H_S = 0.0
    for pi in p:
        if pi > 0:
            H_S -= pi * np.log(pi)
    H = H_S / H_max if H_max > 0 else 0.0

    # Jensen-Shannon divergence
    JS = _jensen_shannon_divergence(p, q)

    # Normalisation factor Q_0 for the JSD (maximum possible value)
    # Q_0 = -0.5 * [(N+1)/N * ln(N+1) - 2*ln(2N) + ln(N)]
    # where N = number of possible states = d!
    N_states = n_perm
    Q_0 = (-0.5 * ((N_states + 1.0) / N_states * np.log(N_states + 1.0)
                    - 2.0 * np.log(2.0 * N_states) + np.log(N_states)))

    # Statistical complexity
    Q_JS = JS / Q_0 if Q_0 > 0 else 0.0
    C = Q_JS * H

    return float(H), float(C)


def permutation_entropy_sweep(x, d_values=None, tau=1):
    """Compute permutation entropy for multiple embedding dimensions.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    d_values : list of int or None
        Embedding dimensions to test. Default: [3, 4, 5, 6, 7].
    tau : int
        Time delay.

    Returns
    -------
    d_values : list of int
    H_values : list of float
    """
    if d_values is None:
        d_values = [3, 4, 5, 6, 7]

    H_values = [permutation_entropy(x, d=d, tau=tau) for d in d_values]
    return d_values, H_values
