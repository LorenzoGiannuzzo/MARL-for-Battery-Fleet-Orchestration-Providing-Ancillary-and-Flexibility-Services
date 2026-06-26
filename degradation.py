"""
Non-linear LFP battery degradation model.

Step 3 of the multi-agent track. Implements a physically-realistic
degradation model for LFP cells, suitable for hourly BESS scheduling
simulation. The model is the canonical decomposition of total capacity
fade into calendar aging (time-driven, independent of operation) and
cycle aging (operation-driven, dependent on throughput and DoD):

    Q_loss(t)  =  Q_loss_calendar(t)  +  Q_loss_cycle(t)

This module is intentionally STANDALONE: it does not import from any other
project file. It can be unit-tested in isolation and validated against
the reference numbers published in:

  Schmalstieg, J. et al. (2014). "A holistic aging model for
    Li(NiMnCo)O2 based 18650 lithium-ion batteries". J. Power Sources,
    257, 325-334. [cycle aging structural form, sqrt-throughput].

  Naumann, M. et al. (2018). "Analysis and modeling of calendar aging of
    LiFePO4 batteries". J. Energy Storage, 17, 153-169. [LFP calendar
    aging coefficients, SOC-Arrhenius stress factors].

  Cui, Y. et al. (2022). "Multi-stress aging modeling for LFP batteries
    in BESS applications" [LFP cycle parameter recalibration].

The model exposes:

  - `LFPDegradationParameters`: dataclass holding all coefficients with
    paper references. Default values are the consensus LFP estimates at
    25C, SOC reference 0.5.
  - `calendar_aging_loss(t_days, soc_avg, T_kelvin, params)`: closed-form
    expression for the calendar capacity loss after `t_days` at constant
    average SOC and temperature.
  - `cycle_aging_loss(Q_throughput_mwh, dod_avg, params)`: closed-form
    expression for the cycle capacity loss after `Q_throughput_mwh` of
    cumulative throughput at average DoD `dod_avg`.
  - `LFPBatteryState`: a stateful integrator that tracks SOH evolution
    step by step as the simulation advances, accumulating both aging
    streams with their respective stress factors evaluated at the
    operating point of each hour.

For Step 4 we will piecewise-linearise these closed-form expressions
into MILP-compatible constraints. For now this module produces the
ground-truth non-linear cost that the MILP will be linearising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================================
# Parameter dataclass
# ============================================================================

@dataclass
class LFPDegradationParameters:
    """LFP capacity-fade model coefficients.

    Defaults are the consensus values for prismatic LFP cells in BESS
    applications at 25 C and SOC reference 0.5, sourced from Naumann
    et al. (2018) for calendar and Schmalstieg et al. (2014) recalibrated
    by Cui et al. (2022) for LFP cycle aging.
    """
    # ----- Calendar aging (Naumann 2018, LFP) -----
    # Form: Q_loss_cal = k_cal * f_SOC(soc) * f_T(T) * sqrt(t_days)
    # Calibrated so that 1 year at SOC=0.5, 25C gives ~3.5% capacity loss,
    # matching the consensus LFP calendar aging rate at ambient conditions.
    k_cal: float = 0.00183         # fraction per day^0.5, ref 25C SOC=0.5
    c_soc_a: float = 1.39          # SOC-stress coefficient a (Naumann eq. 7)
    c_soc_b: float = 0.39          # SOC-stress coefficient b
    soc_ref: float = 0.5           # SOC at which k_cal is calibrated
    E_a_cal: float = 24500.0       # J/mol activation energy
    T_ref_K: float = 298.15        # K reference (25 C)

    # ----- Cycle aging (Schmalstieg 2014 form, LFP recalibration) -----
    # Form: Q_loss_cyc = k_cyc * f_DoD(dod) * Q^z
    # where Q is the cumulative full-cycle equivalents (FCE), i.e. cumulative
    # throughput MWh divided by (2 * capacity_mwh). Calibrated so that 8000
    # FCE at DoD=0.80 gives 20% capacity loss, matching the LFP datasheet
    # nameplate convention (commercial prismatic LFP: CATL, BYD, EVE typically
    # rate 6000-8000 cycles to 80% SOH at DoD=80%).
    # DoD exponent set to 2.0 following Cui et al. (2022) recalibration for
    # LFP chemistry — earlier Schmalstieg used 4.0 for NMC, which is too
    # aggressive for LFP and collapses the shallow-cycle damage to zero.
    k_cyc: float = 0.00224         # fraction per FCE^z at reference DoD
    z: float = 0.5                 # throughput exponent (sqrt; concave fade)
    c_dod_a: float = 2.0           # DoD stress exponent (LFP, Cui 2022)
    c_dod_b: float = 1.5           # reserved for extensions
    dod_ref: float = 0.80          # DoD at which k_cyc is calibrated

    # ----- Operational thresholds -----
    eol_capacity_fraction: float = 0.80  # 20% loss = end of life

    R_gas: float = 8.314           # J/(mol K), universal gas constant


# ============================================================================
# Closed-form aging expressions
# ============================================================================

def _soc_stress(soc: float, p: LFPDegradationParameters) -> float:
    """Calendar SOC-stress factor, normalised so f_SOC(soc_ref) = 1.

    Form: exp(c_soc_a * (soc - soc_ref)) * exp(c_soc_b * (soc**2 - soc_ref**2))
    Captures the well-documented fact that high SOC accelerates SEI growth
    superlinearly. Slightly above 1 at SOC > 0.5, slightly below at SOC < 0.5.
    """
    return (math.exp(p.c_soc_a * (soc - p.soc_ref))
            * math.exp(p.c_soc_b * (soc * soc - p.soc_ref * p.soc_ref)))


def _temperature_stress(T_kelvin: float, p: LFPDegradationParameters) -> float:
    """Arrhenius temperature factor, normalised so f_T(T_ref) = 1.

    Form: exp(-E_a/R * (1/T - 1/T_ref))
    """
    return math.exp(-(p.E_a_cal / p.R_gas) * (1.0 / T_kelvin - 1.0 / p.T_ref_K))


def _dod_stress(dod: float, p: LFPDegradationParameters) -> float:
    """Cycle DoD-stress factor, normalised so f_DoD(dod_ref) = 1.

    Form: (dod/dod_ref)^c_dod_a, capturing the well-known fact that deep
    cycles damage much more than shallow cycles. At dod=1 (full cycle)
    the factor is ~2x the reference, at dod=0.2 it is ~0.05x.
    """
    if dod <= 1e-6:
        return 0.0
    return (dod / p.dod_ref) ** p.c_dod_a


def calendar_aging_loss(
    t_days: float,
    soc_avg: float,
    T_kelvin: float = 298.15,
    params: Optional[LFPDegradationParameters] = None,
) -> float:
    """Fraction of nominal capacity lost to calendar aging after `t_days`.

    Returns a number in [0, 1]. Typical LFP values: ~3-4% per year at
    25 C and SOC 0.5, growing more at high SOC and high temperature.
    """
    p = params or LFPDegradationParameters()
    if t_days <= 0.0:
        return 0.0
    return float(
        p.k_cal
        * _soc_stress(soc_avg, p)
        * _temperature_stress(T_kelvin, p)
        * math.sqrt(t_days)
    )


def cycle_aging_loss(
    Q_throughput_fce: float,
    dod_avg: float,
    params: Optional[LFPDegradationParameters] = None,
) -> float:
    """Fraction of nominal capacity lost to cycle aging after `Q_throughput_fce`
    full-cycle-equivalents at average depth-of-discharge `dod_avg`.

    A "full-cycle equivalent" (FCE) is defined as the cumulative MWh of
    throughput divided by the nominal capacity in MWh. So a 4 MWh battery
    that has charged-and-discharged 8 MWh cumulatively has Q_throughput_fce
    = 2 FCE (one full cycle is 2 * capacity = charge + discharge).

    Returns a number in [0, 1].
    """
    p = params or LFPDegradationParameters()
    if Q_throughput_fce <= 0.0:
        return 0.0
    return float(
        p.k_cyc
        * _dod_stress(dod_avg, p)
        * (Q_throughput_fce ** p.z)
    )


# ============================================================================
# Stateful integrator
# ============================================================================

@dataclass
class LFPBatteryState:
    """Step-by-step capacity-fade integrator for an LFP cell.

    Holds the current state of health and accumulates calendar+cycle aging
    incrementally as the simulation advances. Designed to be called once
    per simulation hour by the env / MILP-shadow simulator.

    Usage pattern:

        state = LFPBatteryState(capacity_mwh=4.0)
        for hour in range(8760):
            # ... env step produces charge/discharge ...
            state.advance(
                hours=1.0, soc_avg=soc_this_hour,
                charge_mwh=ch_mwh, discharge_mwh=dis_mwh,
                dod=instantaneous_dod, T_kelvin=298.15,
            )
        print(state.soh, state.capacity_lost_fraction)

    The total capacity lost at any point is the sum of calendar and cycle
    contributions evaluated at the CURRENT cumulative time and throughput.
    The integrator does not assume the two streams are independent at the
    instant of update; instead, it integrates each stream's marginal rate
    over the step.
    """
    capacity_mwh: float
    params: LFPDegradationParameters = None

    # Cumulative state
    t_days: float = 0.0
    fce_cumulative: float = 0.0   # full-cycle equivalents

    # Running averages for the stress factors (time- and throughput-weighted)
    _soc_time_integral: float = 0.0       # SOC * days
    _dod_throughput_integral: float = 0.0  # DoD * FCE
    _T_time_integral: float = 0.0          # T_K * days

    # Cumulative losses (for fast access)
    _Q_loss_cal: float = 0.0
    _Q_loss_cyc: float = 0.0

    def __post_init__(self):
        if self.params is None:
            self.params = LFPDegradationParameters()

    @property
    def soh(self) -> float:
        """State of Health = 1 - total_capacity_lost_fraction, in [0, 1]."""
        return max(0.0, 1.0 - self.capacity_lost_fraction)

    @property
    def capacity_lost_fraction(self) -> float:
        """Total fractional capacity loss (calendar + cycle)."""
        return self._Q_loss_cal + self._Q_loss_cyc

    @property
    def is_end_of_life(self) -> bool:
        """True when SOH drops below the EoL threshold (default 80%)."""
        return self.soh < self.params.eol_capacity_fraction

    def advance(
        self,
        hours: float,
        soc_avg: float,
        charge_mwh: float,
        discharge_mwh: float,
        dod: float = 0.8,
        T_kelvin: float = 298.15,
    ) -> Tuple[float, float]:
        """Integrate one simulation step (typically 1 hour).

        Parameters
        ----------
        hours : float
            Step length in hours (usually 1.0).
        soc_avg : float
            Average SOC during the step, used for calendar SOC-stress.
        charge_mwh : float
            Charge energy delivered to the battery during the step.
        discharge_mwh : float
            Discharge energy taken from the battery during the step.
        dod : float
            Instantaneous DoD characterising the cycling stress during the
            step. For most BESS operation a sensible proxy is the
            running-window DoD over the past 24h; for a quick approximation
            the caller can supply a fixed nominal DoD (e.g. 0.8) and rely
            on the throughput-only term.
        T_kelvin : float
            Cell temperature during the step. Defaults to 298.15 K (25 C).

        Returns
        -------
        (delta_cal, delta_cyc) : tuple of floats
            The incremental fractional capacity loss added in this step
            from each aging stream.
        """
        dt_days = hours / 24.0
        new_t_days = self.t_days + dt_days

        # Calendar aging: marginal increment from t_days to new_t_days at
        # the current stress factors.
        cal_before = calendar_aging_loss(self.t_days, soc_avg, T_kelvin, self.params)
        cal_after  = calendar_aging_loss(new_t_days, soc_avg, T_kelvin, self.params)
        delta_cal = max(0.0, cal_after - cal_before)

        # Cycle aging: convert throughput to FCE (one FCE = one full cycle's
        # worth of energy, i.e. 2 * capacity_mwh because a cycle includes
        # one charge and one discharge of the nominal capacity).
        throughput_mwh = charge_mwh + discharge_mwh
        d_fce = throughput_mwh / (2.0 * self.capacity_mwh) if self.capacity_mwh > 0 else 0.0
        new_fce = self.fce_cumulative + d_fce

        cyc_before = cycle_aging_loss(self.fce_cumulative, dod, self.params)
        cyc_after  = cycle_aging_loss(new_fce, dod, self.params)
        delta_cyc = max(0.0, cyc_after - cyc_before)

        # Commit state
        self.t_days = new_t_days
        self.fce_cumulative = new_fce
        self._soc_time_integral += soc_avg * dt_days
        self._dod_throughput_integral += dod * d_fce
        self._T_time_integral += T_kelvin * dt_days
        self._Q_loss_cal += delta_cal
        self._Q_loss_cyc += delta_cyc

        return delta_cal, delta_cyc

    # ------------------------------------------------------------------
    # Marginal cost helpers
    # ------------------------------------------------------------------

    def marginal_cycle_cost_per_mwh(
        self, dod: float, replacement_cost: float = 200000.0,
    ) -> float:
        """POINT marginal degradation cost per MWh at the current state.

        Note: at FCE_cumulative = 0 (brand new cell), the derivative of
        the sqrt curve is infinite. This function uses a small FCE floor
        for that case. For MILP integration, prefer
        `avg_marginal_cycle_cost_per_mwh` which integrates over the
        expected daily throughput range and is numerically well-behaved
        for fresh cells.
        """
        p = self.params
        if self.fce_cumulative <= 1e-9:
            fce_eval = 1e-3
        else:
            fce_eval = self.fce_cumulative
        d_loss_d_fce = (p.k_cyc * _dod_stress(dod, p) * p.z
                        * (fce_eval ** (p.z - 1.0)))
        d_loss_d_mwh = d_loss_d_fce / (2.0 * self.capacity_mwh)
        return d_loss_d_mwh * replacement_cost

    def avg_marginal_cycle_cost_per_mwh(
        self, dod: float, Q_estimated_daily_mwh: float,
        replacement_cost: float = 200000.0,
    ) -> float:
        """AVERAGE cost per MWh of throughput across the expected daily window.

        Computes the chord slope:
          (C(Q0 + dQ) - C(Q0)) / dQ
        where dQ = Q_estimated_daily_mwh is a representative daily
        throughput (typically 1-2 full cycles for an active BESS). This
        is the right LINEAR cost coefficient to give a MILP that operates
        on a daily horizon: it is the average marginal damage the BESS
        will experience across the day. Unlike the point derivative, it
        is finite for fresh cells.

        This is the MILP-friendly cost rate used in Step 4. The MILP
        approximates the cycle aging cost as a linear function of daily
        throughput with this slope; the DRL is not constrained by this
        approximation and can exploit the within-horizon non-linearity
        AND the DoD-dependence AND the SOC-calendar coupling, none of
        which the MILP captures at all.
        """
        p = self.params
        cap = self.capacity_mwh
        if cap <= 0 or Q_estimated_daily_mwh <= 0:
            return 0.0
        Q0_fce = self.fce_cumulative
        dQ_fce = Q_estimated_daily_mwh / (2.0 * cap)
        f0 = cycle_aging_loss(Q0_fce, dod, p)
        f1 = cycle_aging_loss(Q0_fce + dQ_fce, dod, p)
        d_loss_per_mwh = (f1 - f0) / Q_estimated_daily_mwh
        return float(d_loss_per_mwh * replacement_cost)


# ============================================================================
# Piecewise linearisation for MILP (Step 4)
# ============================================================================

@dataclass
class PiecewiseLinearDegradation:
    """Piecewise-linear over-approximation of the cycle aging loss.

    The cycle aging cost as a function of cumulative throughput Q (in FCE)
    is concave: f(Q) = k_cyc * f_DoD(dod) * Q^0.5. We approximate it on
    [0, Q_max] with K linear segments, each defined by a tangent line at a
    breakpoint:

        L_j(Q) = slope_j * Q + intercept_j

    Because the curve is concave, every tangent line lies ABOVE the curve
    everywhere except at the tangent point. The piecewise-linear
    OVER-approximation is therefore

        f_pwl(Q) = min_j L_j(Q)   (the lowest tangent at Q)

    In a MILP that MAXIMISES profit (= revenue - cost), the cost being a
    `min` of linear functions becomes a `max` of linear lower bounds on a
    free variable Q_loss_var, with no integer variables needed:

        Q_loss_var >= L_j(Q)   for j = 1..K
        objective: ... - replacement_cost * Q_loss_var

    The solver automatically picks the binding constraint (the lowest tangent
    at the realised Q), which IS the piecewise-linear approximation. Tight,
    LP-friendly, no auxiliary binaries.

    The approximation error is bounded by the curvature between adjacent
    breakpoints. We position the breakpoints logarithmically so the error
    is roughly uniform in relative terms across the throughput range.
    """
    capacity_mwh: float
    dod_assumed: float = 0.6
    replacement_cost: float = 200000.0
    fce_breakpoints: List[float] = None  # FCE values at which to linearise
    params: LFPDegradationParameters = None

    # Computed coefficients
    slopes_eur_per_mwh: List[float] = None
    intercepts_eur: List[float] = None
    fce_max: float = None

    def __post_init__(self):
        if self.params is None:
            self.params = LFPDegradationParameters()
        if self.fce_breakpoints is None:
            # Logarithmic placement: dense at small Q (where curvature is
            # high) and sparser at large Q (where the curve flattens).
            self.fce_breakpoints = [50.0, 200.0, 600.0, 1500.0, 3500.0, 8000.0]
        self._compute_tangents()

    def _compute_tangents(self):
        """Compute slope and intercept for each tangent line in EUR units.

        At each breakpoint Q*, the curve value is f(Q*) and the slope is
        f'(Q*) = k_cyc * f_DoD * z * Q*^(z-1). The tangent line is

            L(Q) = f(Q*) + f'(Q*) * (Q - Q*)
                 = f'(Q*) * Q + (f(Q*) - f'(Q*) * Q*)

        Converted to EUR cost per MWh of throughput:
          - fraction-of-capacity-lost is multiplied by replacement_cost to
            get EUR;
          - Q (in FCE) is divided by 2*capacity_mwh to convert to MWh;
          - therefore slope in EUR/MWh = f'(Q*) * replacement_cost / (2*cap)
          - and intercept in EUR = (f(Q*) - f'(Q*) * Q*) * replacement_cost.
        """
        p = self.params
        dod_factor = _dod_stress(self.dod_assumed, p)
        slopes_eur_per_mwh = []
        intercepts_eur = []
        for fce_star in self.fce_breakpoints:
            f_val = p.k_cyc * dod_factor * (fce_star ** p.z)
            f_prime = p.k_cyc * dod_factor * p.z * (fce_star ** (p.z - 1.0))
            # Tangent intercept in FCE units, in fraction-of-capacity
            intercept_frac = f_val - f_prime * fce_star
            # Convert to EUR units
            slope_eur_per_fce = f_prime * self.replacement_cost
            slope_eur_per_mwh = slope_eur_per_fce / (2.0 * self.capacity_mwh)
            intercept_eur = intercept_frac * self.replacement_cost
            slopes_eur_per_mwh.append(slope_eur_per_mwh)
            intercepts_eur.append(intercept_eur)
        self.slopes_eur_per_mwh = slopes_eur_per_mwh
        self.intercepts_eur = intercepts_eur
        self.fce_max = max(self.fce_breakpoints)

    def cost_eur_for_throughput(self, cumulative_throughput_mwh: float) -> float:
        """Evaluate the piecewise-linear cost in EUR at a given throughput.

        Returns the OVER-approximation (max of tangents below the curve =
        min of tangents above the curve for the cost function). Use to
        verify approximation quality against the true non-linear cost.
        """
        # Lowest tangent at this Q gives the tightest upper bound on the
        # true (concave) cost curve.
        costs = [s * cumulative_throughput_mwh + i
                 for s, i in zip(self.slopes_eur_per_mwh, self.intercepts_eur)]
        # Tangents to a concave function are above the curve; the tightest
        # one is the minimum (= the closest from above).
        return float(min(costs))

    def true_cost_eur_for_throughput(self, cumulative_throughput_mwh: float) -> float:
        """Evaluate the TRUE non-linear cost at a given throughput, EUR.

        Used for benchmarking the piecewise approximation quality.
        """
        if self.capacity_mwh <= 0:
            return 0.0
        fce = cumulative_throughput_mwh / (2.0 * self.capacity_mwh)
        loss_frac = cycle_aging_loss(fce, self.dod_assumed, self.params)
        return float(loss_frac * self.replacement_cost)

    def approximation_error(self, cumulative_throughput_mwh: float) -> float:
        """Relative approximation error |pwl - true| / true at a given Q.

        Returns a non-negative float. The pwl is always >= the true cost
        (over-approximation), so this is also the relative overestimate.
        """
        true_val = self.true_cost_eur_for_throughput(cumulative_throughput_mwh)
        if true_val <= 1e-9:
            return 0.0
        pwl_val = self.cost_eur_for_throughput(cumulative_throughput_mwh)
        return abs(pwl_val - true_val) / true_val