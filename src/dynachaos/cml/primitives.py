"""Primitive CML/GCM step functions shared across dynachaos.cml modules."""

import numpy as np

from dynachaos.cml.globally_coupled import gcm_step
from dynachaos.cml.spatiotemporal import cml_step
from dynachaos.maps.primitives import logistic


def cml_step_logistic(x, a, eps):
    """One CML step with f=g=logistic (Model C shorthand)."""
    fx = logistic(x, a)
    coupling = eps / 2.0 * (np.roll(fx, -1) + np.roll(fx, 1) - 2.0 * fx)
    return fx + coupling


__all__ = ["cml_step", "cml_step_logistic", "gcm_step"]
