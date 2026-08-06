# -*- coding: utf-8 -*-
"""
market_constants.py — Single source of truth for reserve-product sustain
durations and their regulatory basis.

WHY THIS MODULE EXISTS
----------------------
Before this module, tau_FCR was defined in SEVEN places across the codebase
with THREE different values simultaneously active in a single run:

    marl_env.py:1065           4.00 h   bid decode/clip
    marl_env.py:699            0.50 h   FIX 6 availability check
    marl_milp_continuous.py    0.25 h   MILP constructor default
    marl_pipeline.py:242,257   4.00 h   MILP override (demos)
    marl_bc.py:211             4.00 h   MILP override (BC aggregate)
    italian_market_config.py   4    h   TechnicalRequirements
    flexibility_market.py:146  4    h   tariff table

tau_aFRR was 1 h in the env and MILP but 4 h in italian_market_config.
A reviewer with access to the code would (correctly) call this
uninterpretable. Every module now imports from here.

REGULATORY BASIS
----------------
FCR. Commission Regulation (EU) 2017/1485 (SO GL), Article 156(10):
    "all TSOs shall develop a proposal concerning the minimum activation
     period to be ensured by FCR providers. The period determined shall
     not be greater than 30 or smaller than 15 minutes."
Article 156(9) applies the requirement to providing units or groups with
limited energy reservoirs (LER), which is what a BESS is. Where no period
has been determined, the fallback in Art. 156(9) is 15 minutes continuous
full activation. Terna's energy-margin requirement for electrochemical
storage in primary regulation is in Allegato A.15 to the Codice di Rete,
section 6.3.2.

    => SO_GL_DEFAULT uses tau_FCR = 0.25 h (15 min).

The 4.0 h previously hard-coded in the environment is off by roughly one
order of magnitude. Its practical effect was to make FCR unofferable at
rated power for a 2 h fleet, which mechanically forced the policies onto
the restoration products. Keep LEGACY_PAPER only to reproduce the
pre-revision runs; it is not a defensible modelling assumption.

FCR remuneration. FCR in Italy has historically been a mandatory,
non-remunerated obligation under Chapter 4 of the Codice di Rete. It is now
procured through a market: Allegato A.83 to the Codice di Rete ("Meccanismo
di approvvigionamento della FCR"), approved by ARERA with Deliberazione
466/2025/R/eel under Section 4-15 of the TIDE, remunerates the awarded band
irrespective of activation, with separate upward and downward reserve
premia. Start of the consolidation phase was moved to 3 June 2026 by ARERA
Deliberazione 51/2026/R/eel. The capacity payment modelled here is
therefore prospective rather than counterfactual, and should be cited as
such in the paper.

aFRR / mFRR. The 1 h and 2 h figures are modelling assumptions for the
energy-adequacy constraint, not prequalification durations transcribed from
a Terna document. They are exposed here so that they enter the sensitivity
analysis explicitly rather than sitting as magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Regulatory bounds (SO GL Art. 156(10)): 15 min <= tau_FCR <= 30 min
# ---------------------------------------------------------------------------
SOGL_TAU_FCR_MIN_H: float = 0.25
SOGL_TAU_FCR_MAX_H: float = 0.50


@dataclass(frozen=True)
class SustainDurations:
    """Minimum sustain window per reserve product, in hours.

    A unit reserving power r for service s must be able to hold r for
    tau_s, which is what couples the reservation to the SOC in MILP
    Eq. (4)-(5) and in the environment's bid clip.

    Frozen so that a scenario cannot be mutated halfway through a run,
    which is how the three-way inconsistency arose in the first place.
    """

    fcr: float = SOGL_TAU_FCR_MIN_H
    afrr: float = 1.0
    mfrr: float = 2.0
    label: str = "sogl_default"

    def __post_init__(self) -> None:
        for name in ("fcr", "afrr", "mfrr"):
            v = getattr(self, name)
            if not (v > 0.0):
                raise ValueError(
                    f"sustain duration {name}={v!r} must be strictly positive"
                )

    # -- constructors --------------------------------------------------

    @classmethod
    def sogl_default(cls) -> "SustainDurations":
        """SO GL Art. 156(9)-(10) compliant baseline. Use this for the
        headline results of the revised paper."""
        return cls(fcr=SOGL_TAU_FCR_MIN_H, afrr=1.0, mfrr=2.0,
                   label="sogl_default")

    @classmethod
    def legacy_paper(cls) -> "SustainDurations":
        """The pre-revision configuration (tau_FCR = 4 h). Reproduction
        only. Reviewer 2 point 2 identifies this as an error, not a
        conservative assumption."""
        return cls(fcr=4.0, afrr=1.0, mfrr=2.0, label="legacy_paper")

    @classmethod
    def with_tau_fcr(cls, tau_fcr: float,
                     label: str | None = None) -> "SustainDurations":
        """One point of the tau_FCR sensitivity sweep, other products held
        fixed."""
        return cls(fcr=float(tau_fcr), afrr=1.0, mfrr=2.0,
                   label=label or f"tau_fcr_{tau_fcr:g}h")

    def replace(self, **kw) -> "SustainDurations":
        return replace(self, **kw)

    # -- accessors -----------------------------------------------------

    def as_milp_kwargs(self) -> Dict[str, float]:
        """Keyword arguments for the MILP constructors
        (marl_milp_continuous / marl_milp_discrete)."""
        return {
            "fcr_sustain_hours": self.fcr,
            "afrr_sustain_hours": self.afrr,
            "mfrr_sustain_hours": self.mfrr,
        }

    def as_dict(self) -> Dict[str, float]:
        return {"fcr": self.fcr, "afrr": self.afrr, "mfrr": self.mfrr}

    def for_service_key(self, key: str) -> float:
        """Sustain window for a service key as used by the commitment table
        ('fcr', 'afrr', 'afrr_up', 'afrr_dn', 'mfrr', 'mfrr_up', ...)."""
        k = str(key).lower()
        if k.startswith("fcr"):
            return self.fcr
        if k.startswith("afrr"):
            return self.afrr
        if k.startswith("mfrr"):
            return self.mfrr
        return 1.0

    def is_sogl_compliant(self) -> bool:
        return SOGL_TAU_FCR_MIN_H <= self.fcr <= SOGL_TAU_FCR_MAX_H

    def describe(self) -> str:
        flag = "" if self.is_sogl_compliant() else "  [OUTSIDE SO GL 156(10)]"
        return (f"tau_FCR={self.fcr:g}h, tau_aFRR={self.afrr:g}h, "
                f"tau_mFRR={self.mfrr:g}h ({self.label}){flag}")


# ---------------------------------------------------------------------------
# Module-level default. Every module that does not receive an explicit
# SustainDurations falls back to this, so there is exactly one place to
# change the baseline.
# ---------------------------------------------------------------------------
DEFAULT_SUSTAIN = SustainDurations.sogl_default()


# ---------------------------------------------------------------------------
# tau_FCR sensitivity grid (Reviewer 2 point 2: "Correct it, or add a
# sensitivity over tau_FCR" -- the revision does both).
#
# 0.25 and 0.50 bracket the SO GL admissible range; 1, 2 and 4 h are
# out-of-range points retained so that the paper can show where the
# original result came from and how it moves.
# ---------------------------------------------------------------------------
TAU_FCR_SWEEP: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def tau_fcr_sweep_scenarios() -> Tuple[SustainDurations, ...]:
    return tuple(SustainDurations.with_tau_fcr(t) for t in TAU_FCR_SWEEP)


__all__ = [
    "SustainDurations",
    "DEFAULT_SUSTAIN",
    "TAU_FCR_SWEEP",
    "tau_fcr_sweep_scenarios",
    "SOGL_TAU_FCR_MIN_H",
    "SOGL_TAU_FCR_MAX_H",
]
