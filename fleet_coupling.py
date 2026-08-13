# -*- coding: utf-8 -*-
"""
fleet_coupling.py — Constraints that bind the units to EACH OTHER.

WHY THIS MODULE EXISTS
----------------------
All three EI.A reviewers rejected the multi-agent framing on the same
ground. Reviewer 2 put it most bluntly: with one shared policy, clusters
differing only in scale, and a 1 MW participation threshold that rarely
binds against a 55 MW fleet, the formulation is "effectively fifty
independent single-agent problems".

They are right, and no amount of architecture fixes it: without a constraint
that makes one unit's decision depend on what the others do, there is
nothing to coordinate. Coordination has to be created in the PROBLEM before
it can be studied in the SOLUTION.

The coupling implemented here is a shared grid connection limit: the units
sit behind one point of delivery whose rated power is smaller than the sum
of the individual ratings, so at every hour the fleet competes for a scarce
shared resource. It is the most defensible option physically (a real
aggregator's assets very often share a connection), it is trivially
representable in the MILP so the benchmark stays matched, and it binds
continuously rather than at a threshold.

THE ENVELOPE CONVENTION
-----------------------
A connection limit constrains instantaneous power flow. Scheduled arbitrage
power flows for certain; reserved capacity flows only if activated. The
convention used here is the FIRM one: the connection must be able to carry
the worst case in each direction, so per hour

    upward   :  sum_i ( P_dis_i + R_fcr_i + R_afrr_up_i + R_mfrr_up_i ) <= P_conn
    downward :  sum_i ( P_ch_i  + R_fcr_i + R_afrr_dn_i + R_mfrr_dn_i ) <= P_conn

FCR is symmetric and therefore counts on both sides, exactly as it does in
the per-unit sustain budget (marl_env FIX A). Reserving capacity that the
connection could not physically carry would be a paper commitment, which is
the same failure mode the reviewers flagged in the reward model.

MILP AND ENVIRONMENT MUST AGREE
-------------------------------
This project has twice been bitten by a constraint that the optimizer and
the environment interpreted differently (tau_FCR defined in seven places
with three live values; the per-service sustain caps that summed to three
times the SoC budget). Both times the benchmark was planning against one
world and the policy was evaluated in another.

So this module is the single source of truth for the coupling, exactly as
market_constants is for the sustain windows. The MILP adds the two
inequalities above; the environment projects the decoded fleet bid onto the
same envelope. `test_fleet_coupling.py` asserts that a MILP-feasible
schedule passes the environment projection unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Service keys contributing to each direction of the connection envelope.
# FCR is symmetric and appears in both.
UPWARD_RESERVE_KEYS = ("R_fcr", "R_afrr_up", "R_mfrr_up")
DOWNWARD_RESERVE_KEYS = ("R_fcr", "R_afrr_dn", "R_mfrr_dn")

# Tolerance on the envelope check.
#
# The MILP drives the connection constraint to equality whenever it is
# profitable, so a replayed MILP schedule sums to the limit plus whatever
# floating-point error the summation carries — measured at 4.4e-16 MW on a
# 2.5 MW limit, i.e. machine epsilon. Comparing exactly would make the
# projection fire on EVERY hour of every MILP replay, scaling the schedule by
# 1 - 1e-16 and reporting the constraint as binding when it is not. That is a
# silent MILP/environment divergence of exactly the kind this project has
# already been bitten by twice, so the comparison is made with a relative and
# an absolute slack. When the envelope genuinely binds, the projection still
# scales to the TRUE limit, not the tolerant one, so no drift accumulates.
# Calibrated against the SOLVER's feasibility tolerance, not against
# floating-point summation error. A MILP reports a solution feasible when its
# constraint violations sit inside its own primal tolerance, typically around
# 1e-6 relative; measured here, a 50-unit fleet against a 33 MW limit came
# back 3.3e-7 MW over, i.e. 1e-8 relative. An envelope tolerance TIGHTER than
# the solver's own therefore rejects the solver's own optimum, and every hour
# of every MILP replay would be silently re-scaled while the code claimed
# parity.
#
# An earlier value of 1e-9 passed on a 6-unit test fleet and failed on the
# 50-unit production one, which is why the preflight builds the real fleet.
# At 1e-6 the slack is 33 W on a 33 MW connection: far below anything
# physically meaningful, and still a hundred times tighter than the smallest
# violation the tests require to be caught.
ENVELOPE_TOL_REL = 1e-6
ENVELOPE_TOL_ABS = 1e-6   # MW


@dataclass(frozen=True)
class FleetCoupling:
    """Fleet-level constraints shared by the optimizer and the environment.

    connection_limit_mw
        Rated power of the shared point of delivery, in MW. None disables
        the coupling and reproduces the uncoupled behaviour of the submitted
        paper exactly, which is what makes the coupled and uncoupled runs a
        controlled pair.

    Frozen so that a scenario cannot be mutated halfway through a run.
    """

    connection_limit_mw: Optional[float] = None
    label: str = "uncoupled"

    def __post_init__(self) -> None:
        if self.connection_limit_mw is not None:
            if not (self.connection_limit_mw > 0.0):
                raise ValueError(
                    "connection_limit_mw must be strictly positive or None, "
                    f"got {self.connection_limit_mw!r}")

    # -- constructors --------------------------------------------------

    @classmethod
    def uncoupled(cls) -> "FleetCoupling":
        """No fleet coupling. Reproduces the submitted paper."""
        return cls(connection_limit_mw=None, label="uncoupled")

    @classmethod
    def connection(cls, limit_mw: float,
                   label: Optional[str] = None) -> "FleetCoupling":
        return cls(connection_limit_mw=float(limit_mw),
                   label=label or f"conn_{limit_mw:g}MW")

    @classmethod
    def as_fraction_of_fleet(cls, fleet, fraction: float) -> "FleetCoupling":
        """Connection sized as a fraction of the summed unit ratings.

        A fraction of 1.0 or more never binds and is equivalent to no
        coupling; the interesting range is roughly 0.4 to 0.8, where the
        fleet must choose which units get the connection in the hours that
        matter. Expressing the limit relative to the fleet keeps the
        experiment meaningful when the fleet size changes.
        """
        total = sum(b.max_power_mw for b in fleet.batteries)
        return cls.connection(total * float(fraction),
                              label=f"conn_{fraction:g}x_fleet")

    # -- accessors -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.connection_limit_mw is not None

    def describe(self) -> str:
        if not self.is_active:
            return "no fleet coupling (units are independent)"
        return (f"shared connection {self.connection_limit_mw:g} MW "
                f"({self.label})")


# ---------------------------------------------------------------------------
# Envelope arithmetic, shared by the environment and the tests
# ---------------------------------------------------------------------------

def direction_envelopes(decoded: Dict[str, Dict[str, float]]
                        ) -> Tuple[float, float]:
    """Total upward and downward power the fleet bid would put on the wire.

    `decoded` is the environment's per-agent decoded bid dictionary. Missing
    keys count as zero so this works for both the 7-axis directional mode
    and the legacy 5-axis one.
    """
    up = 0.0
    dn = 0.0
    for bid in decoded.values():
        up += float(bid.get("P_discharge", 0.0))
        dn += float(bid.get("P_charge", 0.0))
        for k in UPWARD_RESERVE_KEYS:
            up += float(bid.get(k, 0.0))
        for k in DOWNWARD_RESERVE_KEYS:
            dn += float(bid.get(k, 0.0))
        # legacy 5-axis: aFRR and mFRR are symmetric, so they load both sides
        for k in ("R_afrr", "R_mfrr"):
            sym = float(bid.get(k, 0.0))
            up += sym
            dn += sym
    return up, dn


def envelope_scales(up: float, dn: float, limit: float) -> Tuple[float, float]:
    """Scale factors that bring both envelopes within `limit`.

    Returned as (s_up, s_dn). Quantities that load only one direction are
    scaled by that direction's factor; FCR, which loads both, must be scaled
    by min(s_up, s_dn) so that neither envelope is left violated.

    Scaling DOWN is always safe for the per-unit constraints: less power
    needs less sustaining energy and less headroom, so a bid that satisfied
    the per-unit caps still satisfies them after the projection. That is why
    the fleet pass can run after the per-unit clip without redoing it.
    """
    slack = limit * ENVELOPE_TOL_REL + ENVELOPE_TOL_ABS
    s_up = 1.0 if up <= limit + slack or up <= 1e-12 else limit / up
    s_dn = 1.0 if dn <= limit + slack or dn <= 1e-12 else limit / dn
    return s_up, s_dn


def project_to_connection(decoded: Dict[str, Dict[str, float]],
                          limit: Optional[float]) -> Dict[str, float]:
    """Project a decoded fleet bid onto the connection envelope, in place.

    Proportional scaling is used rather than any priority rule: it preserves
    the relative mix the policy chose, so the projection does not silently
    impose a service or unit ordering the agent never expressed. Returns a
    small diagnostics dict.
    """
    if limit is None:
        return {"binding": 0.0, "s_up": 1.0, "s_dn": 1.0,
                "up_before": 0.0, "dn_before": 0.0}

    up, dn = direction_envelopes(decoded)
    s_up, s_dn = envelope_scales(up, dn, limit)
    if s_up >= 1.0 and s_dn >= 1.0:
        return {"binding": 0.0, "s_up": 1.0, "s_dn": 1.0,
                "up_before": up, "dn_before": dn}

    s_both = min(s_up, s_dn)
    for bid in decoded.values():
        if "P_discharge" in bid:
            bid["P_discharge"] *= s_up
        if "P_charge" in bid:
            bid["P_charge"] *= s_dn
        if "R_afrr_up" in bid:
            bid["R_afrr_up"] *= s_up
        if "R_mfrr_up" in bid:
            bid["R_mfrr_up"] *= s_up
        if "R_afrr_dn" in bid:
            bid["R_afrr_dn"] *= s_dn
        if "R_mfrr_dn" in bid:
            bid["R_mfrr_dn"] *= s_dn
        # symmetric products load both directions
        if "R_fcr" in bid:
            bid["R_fcr"] *= s_both
        for k in ("R_afrr", "R_mfrr"):
            if k in bid:
                bid[k] *= s_both

    return {"binding": 1.0, "s_up": s_up, "s_dn": s_dn,
            "up_before": up, "dn_before": dn}


__all__ = [
    "FleetCoupling",
    "direction_envelopes",
    "envelope_scales",
    "project_to_connection",
    "UPWARD_RESERVE_KEYS",
    "DOWNWARD_RESERVE_KEYS",
]
