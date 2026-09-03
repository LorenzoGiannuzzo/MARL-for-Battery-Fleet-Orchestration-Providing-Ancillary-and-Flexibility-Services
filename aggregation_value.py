# -*- coding: utf-8 -*-
"""
aggregation_value.py — what is aggregation worth, and to whom?

THE QUESTION
------------
Italian ancillary-service products carry a minimum accepted bid of 1 MW per
service. A 0.5 MW unit therefore cannot reach those markets ALONE, whatever
its state of charge: it is confined to day-ahead arbitrage. A 2.5 MW unit
clears the threshold by itself and needs nobody.

So the threshold does not merely reduce revenue: it decides WHO IS ALLOWED IN.
This script measures the consequence by running the same portfolio twice.

    standalone   each unit optimised on its own (N=1), threshold applied to
                 its own bid
    aggregated   the whole portfolio optimised jointly (N=20), threshold
                 applied to the SUM across units

The difference is the value of aggregation.

WHY THE THRESHOLD STAYS ON IN BOTH ARMS
---------------------------------------
Removing it from the comparison arm would measure something else entirely —
the cost of the RULE. The question here is what it is worth to be allowed to
form a group GIVEN the rule, which is the decision an aggregator actually
faces. So the constraint is enforced in both arms and only its SCOPE changes,
from the single unit to the sum.

WHAT MAKES THIS DIFFERENT FROM THE STACKING LITERATURE
------------------------------------------------------
Published revenue-stacking studies optimise a SINGLE asset, where the
threshold is trivial: you clear it or you do not. Across a heterogeneous
portfolio it becomes combinatorial — which subset aggregates onto which
product in which hour, given that the units share an energy budget and have
different storage durations. That decision is what the binaries in
_add_aggregate_market_constraints represent, and it has no counterpart in a
single-unit model.

COST
----
The standalone arm is cheap despite having twenty times the solves: each is a
one-battery problem. The aggregated arm is the expensive one. No DRL, no
behavioural clone, no RLlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from market_constants import SustainDurations, DEFAULT_SUSTAIN
from fleet_coupling import FleetCoupling
from milp_optimizer import BatteryParameters
from marl_milp_continuous import MultiBatteryParameters, MultiBESSMILPOptimizer


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

# (count, rated power MW, duration h, label)
#
# Chosen so the threshold BITES asymmetrically, which is the whole point:
# the 0.5 MW units cannot reach any service alone, the 1.0 MW units reach
# exactly one service at full power and nothing else, and the 2.5 MW units are
# unconstrained. A portfolio of uniformly large units would show no effect and
# a portfolio of uniformly small ones would show no aggregation either, since
# there would be nobody to aggregate WITH.
DEFAULT_PORTFOLIO = (
    (8, 0.5, 4.0, "small_4h"),
    (8, 1.0, 2.0, "medium_2h"),
    (4, 2.5, 1.0, "large_1h"),
)


def build_portfolio(spec=DEFAULT_PORTFOLIO, degradation_cost_per_mwh=25000.0,
                    cycle_life=8000):
    """Return (fleet, per-unit metadata)."""
    batteries, meta = [], []
    for count, p_mw, dur_h, label in spec:
        for k in range(count):
            batteries.append(BatteryParameters(
                capacity_mwh=p_mw * dur_h, max_power_mw=p_mw,
                degradation_cost_per_mwh=degradation_cost_per_mwh,
                cycle_life=cycle_life))
            meta.append({"label": label, "power_mw": p_mw,
                         "duration_h": dur_h, "capacity_mwh": p_mw * dur_h,
                         "index": len(batteries) - 1,
                         "clears_threshold_alone": p_mw >= 1.0})
    return MultiBatteryParameters(batteries), meta


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class UnitResult:
    index: int
    label: str
    power_mw: float
    duration_h: float
    standalone_eur: float = 0.0
    standalone_arbitrage_eur: float = 0.0
    standalone_flex_eur: float = 0.0
    standalone_degradation_eur: float = 0.0
    aggregated_eur: float = float("nan")
    aggregated_arbitrage_eur: float = float("nan")
    aggregated_degradation_eur: float = float("nan")
    # What the unit actually contributed to the joint bid, in MW-hours of
    # reserved band, and its share of the fleet total.
    reserved_band_mwh: float = 0.0
    band_share: float = 0.0
    power_share: float = 0.0
    # The same aggregated result under the three allocation conventions.
    aggregated_by_power_eur: float = float("nan")
    aggregated_by_band_eur: float = float("nan")
    aggregated_floor_eur: float = float("nan")

    def gain_pct(self, rule: str = "band") -> float:
        y = {"power": self.aggregated_by_power_eur,
             "band": self.aggregated_by_band_eur,
             "floor": self.aggregated_floor_eur}[rule]
        if abs(self.standalone_eur) < 1e-9:
            return float("inf") if y > 0 else 0.0
        return 100.0 * (y / self.standalone_eur - 1.0)

    @property
    def aggregation_gain_eur(self) -> float:
        return self.aggregated_eur - self.standalone_eur

    @property
    def aggregation_gain_pct(self) -> float:
        if abs(self.standalone_eur) < 1e-9:
            return float("inf") if self.aggregated_eur > 0 else 0.0
        return 100.0 * (self.aggregated_eur / self.standalone_eur - 1.0)


@dataclass
class ArmResult:
    label: str
    total_eur: float = 0.0
    arbitrage_eur: float = 0.0
    flexibility_eur: float = 0.0
    degradation_eur: float = 0.0
    revenue_by_service: Dict[str, float] = field(default_factory=dict)
    n_days: int = 0
    n_optimal: int = 0
    n_solves: int = 0
    seconds: float = 0.0


@dataclass
class AggregationStudy:
    # Per-unit, per-service detail from both arms. Saved so that a question
    # about allocation raised while writing does not cost another day of
    # solving.
    detail: Dict[str, Any] = field(default_factory=dict)
    year_label: str = ""
    sustain_label: str = ""
    portfolio: List[Dict[str, Any]] = field(default_factory=list)
    standalone: Optional[ArmResult] = None
    aggregated: Optional[ArmResult] = None
    units: List[UnitResult] = field(default_factory=list)

    @property
    def aggregation_value_eur(self) -> float:
        return self.aggregated.total_eur - self.standalone.total_eur

    @property
    def aggregation_value_pct(self) -> float:
        if abs(self.standalone.total_eur) < 1e-9:
            return float("nan")
        return 100.0 * (self.aggregated.total_eur / self.standalone.total_eur
                        - 1.0)


# ---------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------

def _solve_window(fleet, window, n_days: int, sustain: SustainDurations,
                  tag: str, verbose: bool = True) -> Dict[str, Any]:
    """Roll a MILP day by day, carrying the state of charge across days.

    Carrying the SoC matters: an operator does not wake up at 50% every
    morning, and a model that resets it manufactures arbitrage opportunities
    that do not exist. Both arms carry it, so neither is advantaged.
    """
    N = fleet.n_batteries
    opt = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=False,
        directional_services=True, coupling=FleetCoupling.uncoupled(),
        **sustain.as_milp_kwargs())

    soc = [b.initial_soc for b in fleet.batteries]
    totals = dict(net=0.0, arb=0.0, flex=0.0, deg=0.0)
    by_service: Dict[str, float] = {}
    per_unit_arb = np.zeros(N)
    # Reserved band per unit, summed over hours and products, in MW-hours.
    # This is what each unit actually CONTRIBUTED to the joint bid, and the
    # only contribution measure the optimizer itself produces. Rated power is
    # a property of the asset, not of what it did.
    per_unit_band = np.zeros(N)
    # PER-UNIT, PER-SERVICE detail, kept so that any later question about who
    # contributed what can be answered from the saved files instead of by
    # re-solving a year of MILPs. Each solve costs about a minute at fleet
    # scale, so an unrecorded quantity costs a full day to recover; recording
    # a few arrays costs nothing.
    #
    #   band[service][unit]  reserved capacity, MW-hours over the window
    #   act[service][unit]   activated energy, expected MWh
    #   hours[service][unit] hours in which the unit held a non-zero bid,
    #                        which measures ACCESS rather than volume: a unit
    #                        that participates rarely but heavily is a
    #                        different proposition from one that is always in.
    band_by_service = {}
    act_by_service = {}
    hours_by_service = {}
    per_unit_flex_throughput = np.zeros(N)
    per_unit_deg = np.zeros(N)
    n_optimal = 0
    t0 = time.time()

    for day in range(n_days):
        prices, services = window.slice_day(day)
        opt.current_soc = list(soc)
        res = opt.optimize(24, list(prices), list(services))

        if res.solver_status == "Optimal":
            n_optimal += 1
        totals["net"] += float(res.net_profit)
        totals["arb"] += float(res.arbitrage_profit)
        totals["flex"] += float(res.flexibility_revenue)
        totals["deg"] += float(res.degradation_cost)
        for k, v in (res.flexibility_revenue_by_service or {}).items():
            by_service[str(k)] = by_service.get(str(k), 0.0) + float(v)
        per_unit_arb += np.asarray(res.per_battery_arbitrage_profit,
                                   dtype=float)
        per_unit_deg += np.asarray(res.per_battery_degradation_cost,
                                   dtype=float)
        for _svc, mat in (res.flexibility_reservations or {}).items():
            arr = np.asarray(mat, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == N:
                per_unit_band += arr.sum(axis=1)
                k = str(_svc)
                band_by_service[k] = (band_by_service.get(k, np.zeros(N))
                                      + arr.sum(axis=1))
                hours_by_service[k] = (hours_by_service.get(k, np.zeros(N))
                                       + (arr > 1e-9).sum(axis=1))
        for _svc, mat in (res.flexibility_activations or {}).items():
            arr = np.asarray(mat, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == N:
                k = str(_svc)
                act_by_service[k] = (act_by_service.get(k, np.zeros(N))
                                     + arr.sum(axis=1))
                per_unit_flex_throughput += arr.sum(axis=1)
        soc = [float(res.soc_trajectories[i][-1]) for i in range(N)]

        if verbose and (day + 1) % 30 == 0:
            el = time.time() - t0
            print(f"  [{tag}] day {day+1}/{n_days} | {el:.0f}s "
                  f"({el/(day+1):.2f}s/day) | net {totals['net']:,.0f} EUR",
                  flush=True)

    return dict(totals=totals, by_service=by_service,
                per_unit_arb=per_unit_arb, per_unit_deg=per_unit_deg,
                per_unit_band=per_unit_band,
                band_by_service={k: v.tolist()
                                 for k, v in band_by_service.items()},
                act_by_service={k: v.tolist()
                                for k, v in act_by_service.items()},
                hours_by_service={k: v.tolist()
                                  for k, v in hours_by_service.items()},
                per_unit_flex_throughput=per_unit_flex_throughput,
                n_optimal=n_optimal, n_solves=n_days,
                seconds=time.time() - t0)


def run_standalone_arm(fleet, meta, window, n_days, sustain, verbose=True):
    """Every unit on its own: the threshold applies to its own bid.

    A separate one-battery MILP per unit. Twenty times the solves, but each is
    tiny, so this arm is the cheap one.
    """
    arm = ArmResult(label="standalone")
    units: List[UnitResult] = []
    detail = {"band_by_service": {}, "act_by_service": {},
              "hours_by_service": {}}
    t0 = time.time()

    for i, m in enumerate(meta):
        solo = MultiBatteryParameters([fleet.batteries[i]])
        if verbose:
            print(f"[standalone] unit {i+1}/{len(meta)} "
                  f"({m['label']}, {m['power_mw']} MW / "
                  f"{m['capacity_mwh']} MWh)", flush=True)
        out = _solve_window(solo, window, n_days, sustain,
                            tag=f"solo {i+1}", verbose=False)

        u = UnitResult(index=i, label=m["label"], power_mw=m["power_mw"],
                       duration_h=m["duration_h"])
        u.standalone_eur = out["totals"]["net"]
        u.standalone_arbitrage_eur = out["totals"]["arb"]
        u.standalone_flex_eur = out["totals"]["flex"]
        u.standalone_degradation_eur = out["totals"]["deg"]
        units.append(u)

        arm.total_eur += out["totals"]["net"]
        arm.arbitrage_eur += out["totals"]["arb"]
        arm.flexibility_eur += out["totals"]["flex"]
        arm.degradation_eur += out["totals"]["deg"]
        for k, v in out["by_service"].items():
            arm.revenue_by_service[k] = arm.revenue_by_service.get(k, 0.0) + v
        arm.n_optimal += out["n_optimal"]
        arm.n_solves += out["n_solves"]
        # One-battery solves, so every per-unit array has a single entry that
        # belongs to unit i.
        for key in ("band_by_service", "act_by_service", "hours_by_service"):
            for svc, vals in out[key].items():
                detail[key].setdefault(svc, [0.0] * len(meta))
                detail[key][svc][i] = float(vals[0]) if vals else 0.0

    arm.n_days = n_days
    arm.seconds = time.time() - t0
    return arm, units, detail


def run_aggregated_arm(fleet, window, n_days, sustain, verbose=True):
    """The whole portfolio at once: the threshold applies to the SUM."""
    out = _solve_window(fleet, window, n_days, sustain, tag="aggregated",
                        verbose=verbose)
    arm = ArmResult(
        label="aggregated",
        total_eur=out["totals"]["net"], arbitrage_eur=out["totals"]["arb"],
        flexibility_eur=out["totals"]["flex"],
        degradation_eur=out["totals"]["deg"],
        revenue_by_service=out["by_service"], n_days=n_days,
        n_optimal=out["n_optimal"], n_solves=out["n_solves"],
        seconds=out["seconds"])
    return arm, out


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def run_study(window, n_days, year_label, portfolio_spec=DEFAULT_PORTFOLIO,
              sustain: SustainDurations = DEFAULT_SUSTAIN,
              verbose=True) -> AggregationStudy:
    fleet, meta = build_portfolio(portfolio_spec)
    total_mw = sum(m["power_mw"] for m in meta)
    total_mwh = sum(m["capacity_mwh"] for m in meta)
    if verbose:
        print("=" * 70)
        print(f"AGGREGATION STUDY — {year_label}")
        print(f"  portfolio: {len(meta)} units, {total_mw:.1f} MW, "
              f"{total_mwh:.1f} MWh")
        for count, p, d, lab in portfolio_spec:
            alone = "clears 1 MW alone" if p >= 1.0 else "BELOW threshold"
            print(f"    {count} x {p} MW / {p*d} MWh ({d:g}h)  {alone}")
        print(f"  sustain: {sustain.describe()}")
        print(f"  days: {n_days}")
        print("=" * 70)

    arm_solo, units, solo_detail = run_standalone_arm(
        fleet, meta, window, n_days, sustain, verbose)
    arm_agg, agg_out = run_aggregated_arm(fleet, window, n_days, sustain,
                                          verbose)
    agg_arb = agg_out["per_unit_arb"]
    agg_deg = agg_out["per_unit_deg"]
    agg_band = agg_out["per_unit_band"]

    # ALLOCATING THE JOINT RESULT BACK TO UNITS.
    #
    # Arbitrage and degradation are per-unit quantities the optimizer reports
    # directly. Service revenue is NOT: it is earned by the aggregate bid, so
    # no unique split exists and a CONVENTION has to be chosen. That choice is
    # not cosmetic — it decides who appears to gain and who to lose — so three
    # are computed and compared rather than one being asserted.
    #
    #   by_power  proportional to rated power. The obvious rule, and the one
    #             the literature classes among "simple rules": it cannot
    #             reflect marginal contribution and is known to under-reward
    #             real contributors. A unit is paid for what it IS, not for
    #             what it DID.
    #   by_band   proportional to the capacity each unit actually reserved,
    #             summed over hours and products. This is what a settlement
    #             process would use, and the optimizer produces it.
    #   floor     every unit first receives its standalone value, and only the
    #             surplus is shared (by band). This enforces INDIVIDUAL
    #             RATIONALITY, the minimum requirement in the cooperative-game
    #             literature: nobody may end up worse off than alone, or that
    #             member simply leaves the portfolio.
    #
    # The standalone arm doubles as the individual-rationality benchmark, so
    # the check costs nothing extra.
    flex_pool = arm_agg.flexibility_eur

    w_power = np.array([m["power_mw"] for m in meta], dtype=float)
    w_power = w_power / w_power.sum() if w_power.sum() > 0 else w_power
    w_band = np.asarray(agg_band, dtype=float)
    w_band = w_band / w_band.sum() if w_band.sum() > 0 else w_power.copy()

    base = np.array([float(agg_arb[i]) - float(agg_deg[i])
                     for i in range(len(units))])
    solo = np.array([u.standalone_eur for u in units])

    alloc_power = base + flex_pool * w_power
    alloc_band = base + flex_pool * w_band

    # Reserve floor: guarantee the standalone value, share what is left over
    # by contribution. If the joint result cannot cover every floor the
    # coalition is not viable and the guarantee is dropped, which is itself
    # worth reporting rather than silently patching.
    total_joint = float(base.sum() + flex_pool)
    if total_joint >= solo.sum():
        surplus = total_joint - solo.sum()
        alloc_floor = solo + surplus * w_band
    else:
        alloc_floor = alloc_band.copy()

    for i, u in enumerate(units):
        u.aggregated_arbitrage_eur = float(agg_arb[i])
        u.aggregated_degradation_eur = float(agg_deg[i])
        u.reserved_band_mwh = float(agg_band[i])
        u.band_share = float(w_band[i])
        u.power_share = float(w_power[i])
        u.aggregated_eur = float(alloc_band[i])          # headline convention
        u.aggregated_by_power_eur = float(alloc_power[i])
        u.aggregated_by_band_eur = float(alloc_band[i])
        u.aggregated_floor_eur = float(alloc_floor[i])

    return AggregationStudy(
        year_label=year_label, sustain_label=sustain.label,
        portfolio=meta, standalone=arm_solo, aggregated=arm_agg, units=units,
        detail={
            "aggregated": {
                "band_by_service": agg_out["band_by_service"],
                "act_by_service": agg_out["act_by_service"],
                "hours_by_service": agg_out["hours_by_service"],
                "per_unit_arbitrage": list(map(float, agg_arb)),
                "per_unit_degradation": list(map(float, agg_deg)),
            },
            "standalone": solo_detail,
        })


def print_report(study: AggregationStudy):
    s, a = study.standalone, study.aggregated
    print()
    print("=" * 70)
    print(f"RESULTS — {study.year_label}")
    print("=" * 70)
    print(f"  {'':<26}{'standalone':>16}{'aggregated':>16}")
    for name, x, y in (("net profit", s.total_eur, a.total_eur),
                       ("  arbitrage", s.arbitrage_eur, a.arbitrage_eur),
                       ("  services", s.flexibility_eur, a.flexibility_eur),
                       ("  degradation", -s.degradation_eur,
                        -a.degradation_eur)):
        print(f"  {name:<26}{x:>16,.0f}{y:>16,.0f}")
    print(f"\n  VALUE OF AGGREGATION: {study.aggregation_value_eur:,.0f} EUR "
          f"({study.aggregation_value_pct:+.1f}%)")

    print(f"\n  by unit class:")
    print(f"  {'class':<12}{'standalone':>14}{'aggregated':>14}{'gain':>14}")
    seen = {}
    for u in study.units:
        d = seen.setdefault(u.label, [0.0, 0.0, 0])
        d[0] += u.standalone_eur
        d[1] += u.aggregated_eur
        d[2] += 1
    for lab, (x, y, n) in seen.items():
        gain = "n/a" if abs(x) < 1e-9 else f"{100*(y/x-1):+.0f}%"
        print(f"  {lab:<12}{x:>14,.0f}{y:>14,.0f}{gain:>14}")

    print(f"\n  allocation of the joint service revenue — three conventions:")
    print(f"  {'class':<12}{'standalone':>13}{'by power':>13}"
          f"{'by band':>13}{'floor':>13}")
    agg = {}
    for u in study.units:
        d = agg.setdefault(u.label, [0.0, 0.0, 0.0, 0.0])
        d[0] += u.standalone_eur
        d[1] += u.aggregated_by_power_eur
        d[2] += u.aggregated_by_band_eur
        d[3] += u.aggregated_floor_eur
    for lab, (x, p, b, fl) in agg.items():
        f = lambda y: ("n/a" if abs(x) < 1e-9 else f"{100*(y/x-1):+.0f}%")
        print(f"  {lab:<12}{x:>13,.0f}{p:>13,.0f}{b:>13,.0f}{fl:>13,.0f}")
        print(f"  {'':<12}{'':>13}{f(p):>13}{f(b):>13}{f(fl):>13}")

    print(f"\n  INDIVIDUAL RATIONALITY (units ending below their "
          f"standalone value):")
    for rule, key in (("by power", "aggregated_by_power_eur"),
                      ("by band", "aggregated_by_band_eur"),
                      ("floor", "aggregated_floor_eur")):
        losers = [u for u in study.units
                  if getattr(u, key) < u.standalone_eur - 1e-6]
        worst = (min(100 * (getattr(u, key) / u.standalone_eur - 1)
                     for u in losers) if losers else 0.0)
        print(f"    {rule:<10}{len(losers):>3} of {len(study.units)} units"
              + (f", worst {worst:+.1f}%" if losers else ""))

    print(f"\n  contribution vs payment (by band), per class:")
    print(f"  {'class':<12}{'band share':>13}{'power share':>13}")
    sh = {}
    for u in study.units:
        d = sh.setdefault(u.label, [0.0, 0.0])
        d[0] += u.band_share
        d[1] += u.power_share
    for lab, (bs, ps) in sh.items():
        print(f"  {lab:<12}{100*bs:>12.1f}%{100*ps:>12.1f}%")

    print(f"\n  solver: standalone {s.n_optimal}/{s.n_solves} optimal "
          f"({s.seconds/60:.1f} min), aggregated {a.n_optimal}/{a.n_solves} "
          f"({a.seconds/60:.1f} min)")
    print("=" * 70)


def save(study: AggregationStudy, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"aggregation_{study.year_label}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(asdict(study), f, indent=2, default=str)
    cols = ["index", "label", "power_mw", "duration_h", "standalone_eur",
            "standalone_arbitrage_eur", "standalone_flex_eur",
            "aggregated_eur", "aggregated_arbitrage_eur",
            "reserved_band_mwh", "band_share", "power_share",
            "aggregated_by_power_eur", "aggregated_by_band_eur",
            "aggregated_floor_eur"]
    with open(stem + "_units.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols + ["aggregation_gain_eur",
                                 "aggregation_gain_pct"]) + "\n")
        for u in study.units:
            d = asdict(u)
            row = [f"{d[c]}" for c in cols]
            row += [f"{u.aggregation_gain_eur:.2f}",
                    f"{u.aggregation_gain_pct:.2f}"]
            f.write(",".join(row) + "\n")
    print(f"  written: {stem}.json, {stem}_units.csv")
    return stem


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pun-xlsx", nargs="*", default=None,
                    help="GME day-ahead workbook(s)")
    ap.add_argument("--msd-xlsx", nargs="*", default=None)
    ap.add_argument("--price-column", default="PUN")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--year", type=int, action="append",
                    help="repeat for each year, e.g. --year 2023 --year 2024")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--tau", type=float, default=None,
                    help="override tau_FCR in hours")
    ap.add_argument("--outdir", default="aggregation_results")
    args = ap.parse_args(argv)

    if args.pun_xlsx is None and not args.synthetic:
        print("no market data given. Pass --pun-xlsx (and --msd-xlsx), or "
              "--synthetic to say explicitly that generated prices are wanted.")
        return 2
    if args.pun_xlsx is not None and len(args.pun_xlsx) == 1:
        args.pun_xlsx = args.pun_xlsx[0]

    sustain = (SustainDurations.with_tau_fcr(args.tau) if args.tau
               else DEFAULT_SUSTAIN)
    years = args.year or [2023, 2024]

    from italian_market_data import (make_market_window_from_real_pun,
                                     make_synthetic_market_window)
    studies = []
    for y in years:
        start = datetime(y, 1, 1)
        end = start + timedelta(days=args.days)
        print(f"\n[data] building market window for {y}...")
        if args.synthetic:
            window = make_synthetic_market_window(
                start_date=start, end_date=end, seed=0, directional=True)
        else:
            window = make_market_window_from_real_pun(
                start_date=start, end_date=end,
                pun_xlsx_path=args.pun_xlsx,
                price_column=args.price_column,
                msd_xlsx_paths=args.msd_xlsx,
                directional_services=True)
        st = run_study(window, args.days, str(y), sustain=sustain)
        print_report(st)
        save(st, args.outdir)
        studies.append(st)

    return 0


if __name__ == "__main__":
    sys.exit(main())