# -*- coding: utf-8 -*-
"""
action_projection.py — the MILP-to-action-grid projection operator.

WHY THIS IS ITS OWN MODULE
--------------------------
Reviewer 2, point 7: "The projection of the MILP schedule onto the action grid
is asserted to be conservative but is never described, and post-projection
feasibility (SOC bounds, sustain constraints) is not established. Specify the
operator and verify feasibility, or solve the MILP directly on the grid."

The operator lived inside marl_bc as two private helpers, which made it look
like an implementation detail of behavioural cloning. It is not: it is the
bridge between the benchmark and every learned policy, and if it is not
faithful then the reported gap between them measures the bridge rather than
the methods. Pulling it out gives it a name, a specification, and a test file.

It also removes an incidental torch dependency: the operator is pure numpy,
but importing it from marl_bc dragged in torch and made it untestable
wherever torch is absent — including inside the DAgger expert re-query.

THE OPERATOR
------------
Each power axis is expressed as a fraction of the unit's rated power and
rounded to the nearest of N_ACTION_BINS levels spaced evenly on [0, 1]:

    bin(x) = round( clip(x / P_max, 0, 1) * (N_ACTION_BINS - 1) )

Two properties matter for the comparison to be fair.

FIDELITY. With milp_mode="discrete" the optimizer already pins every power
axis to a grid level, so the projection is the IDENTITY: it recovers exactly
the bin the MILP chose and loses nothing. This is verifiable and is asserted
in test_action_projection.py. With milp_mode="continuous" the projection is a
genuine rounding and can move a value by at most half a bin, which is
0.5 / (N_ACTION_BINS - 1) = 5% of rated power at 11 bins.

FEASIBILITY. Rounding is to the NEAREST level, so it can round UP, and a
value rounded up may exceed what the state of charge sustains. The
environment's decode then clips it back. That is why the operator alone is
not a feasibility guarantee and the claim to check is the weaker, true one:
a MILP-feasible schedule passes the environment's decode unchanged in the
discrete mode, because there the projection changes nothing at all. In
continuous mode a rounded-up bid may be clipped, and the resulting handicap
belongs in the paper as a stated property of the projection rather than as an
unexplained part of the MILP-to-policy gap.
"""

from __future__ import annotations

import numpy as np

from marl_env import N_ACTION_BINS, ACTION_AXES, ACTION_AXES_DIRECTIONAL


def bin_fraction(x: float, p_max: float,
                 n_bins: int = N_ACTION_BINS) -> int:
    """Project one power value onto the action grid.

    Clipping to [0, 1] before rounding is deliberate: a solver can return a
    value a few ulps outside its own bound, and an unclipped fraction would
    produce an out-of-range bin that the action space rejects.
    """
    if p_max <= 0:
        return 0
    frac = max(0.0, min(1.0, x / p_max))
    return int(round(frac * (n_bins - 1)))


def bin_width_mw(p_max: float, n_bins: int = N_ACTION_BINS) -> float:
    """Width of one grid step in MW. Half of this bounds the rounding error."""
    return p_max / max(n_bins - 1, 1)


def discretise_milp_action(P_ch: float, P_dis: float,
                           R_fcr: float, R_afrr: float, R_mfrr: float,
                           P_max: float) -> np.ndarray:
    """Legacy 5-axis: [charge, discharge, FCR, aFRR, mFRR]."""
    if P_max <= 0:
        return np.zeros(ACTION_AXES, dtype=np.int64)
    return np.array(
        [bin_fraction(v, P_max)
         for v in (P_ch, P_dis, R_fcr, R_afrr, R_mfrr)],
        dtype=np.int64)


def discretise_milp_action_directional(
    P_ch: float, P_dis: float, R_fcr: float,
    R_afrr_up: float, R_afrr_dn: float,
    R_mfrr_up: float, R_mfrr_dn: float,
    P_max: float,
) -> np.ndarray:
    """Directional 7-axis: [charge, discharge, FCR, aFRR_up, aFRR_dn,
    mFRR_up, mFRR_dn]."""
    if P_max <= 0:
        return np.zeros(ACTION_AXES_DIRECTIONAL, dtype=np.int64)
    return np.array(
        [bin_fraction(v, P_max)
         for v in (P_ch, P_dis, R_fcr, R_afrr_up, R_afrr_dn,
                   R_mfrr_up, R_mfrr_dn)],
        dtype=np.int64)


# Backward-compatible aliases: marl_bc exported these names privately and
# marl_pipeline imports them by those names.
_discretise_milp_action = discretise_milp_action
_discretise_milp_action_directional = discretise_milp_action_directional


__all__ = [
    "bin_fraction", "bin_width_mw",
    "discretise_milp_action", "discretise_milp_action_directional",
    "_discretise_milp_action", "_discretise_milp_action_directional",
]
