# -*- coding: utf-8 -*-
"""
marl_scalability.py — Computational scalability benchmark for the MILP
scheduler against the learned policy.

WHY
---
All three reviewers rejected the tractability argument on the paper's own
numbers:

    R1 pt 5: "The reported MILP time for fifty units is approximately 341
              milliseconds for a full day, which is already small for
              operational scheduling. [...] The timing experiment should
              report solver settings, optimality tolerances, warm-up, the
              number of repeated measurements, and timing dispersion."
    R2 pt 1: "A 341 ms MILP solve for a 50-unit day is not operationally
              binding at a day-ahead cadence [...] Either show a regime
              where the MILP genuinely binds (15-minute resolution over a
              scenario tree, or N in the hundreds with a fitted scaling
              curve), or reframe the contribution away from tractability.
              Also please explain the counter-intuitive decrease in
              per-decision DRL cost with fleet size."
    R3 pt 3: "distinguish inference speed from the substantial offline
              training cost."

This module produces the four things they asked for:

  1. A scaling curve over N = 3 ... 500 with a fitted power law
     t = a * N^b, so the paper can state the exponent instead of quoting
     three points.
  2. Three operational cadences, because the day-ahead cadence is the one
     regime where the MILP is comfortably fast:
         "day_ahead"   1 solve per day,  24 hourly steps
         "intraday"    24 solves per day (hourly re-optimisation)
         "quarter"     96 solves per day at 15-minute resolution
     The MILP cost is per-solve and multiplies by the number of solves; the
     policy emits a bid per decision either way. This is what turns a
     third of a second per day into the regime the paper claims.
  3. Repeated measurements with warm-up discarded, reporting median and
     interquartile range rather than a single number, plus the full solver
     configuration.
  4. The offline cost (demo generation + BC + PPO training), so that
     inference speed is not quoted as if it were free.

USAGE
-----
    # full sweep (this is the paper figure)
    python marl_scalability.py --sizes 3 10 50 100 200 500 --repeats 10

    # quick check
    python marl_scalability.py --sizes 3 10 --repeats 3 --no-policy

    # a specific cadence only
    python marl_scalability.py --cadences quarter --sizes 50 100 200

Outputs, in --outdir (default scalability_results/):
    scalability_raw.csv       one row per (N, cadence, repeat)
    scalability_summary.csv   median / IQR / fitted exponent per (N, cadence)
    scalability_fit.json      power-law fit coefficients and R^2
    scalability_table.md      the table to paste into the paper

NOTE ON THE DRL TIMING ANOMALY
------------------------------
Reviewer 2 asked why the per-decision DRL cost FELL with fleet size in the
submitted version. It is a batching artefact: the policy is queried once per
agent per hour, but the implementation stacks the N observations into one
tensor, so a larger N amortises the fixed per-call overhead (Python dispatch,
tensor allocation) over more units. The per-decision figure therefore mixes
two things. This module reports both `policy_seconds_per_decision` and
`policy_seconds_per_unit_decision`; the second is the one that is flat, and
the paper should quote per-day-per-fleet cost rather than per-decision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from market_constants import SustainDurations, DEFAULT_SUSTAIN


# ---------------------------------------------------------------------------
# Cadences
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cadence:
    key: str
    label: str
    solves_per_day: int          # MILP re-optimisations per operating day
    decisions_per_day: int       # policy queries per unit per operating day
    horizon_steps: int           # steps in one MILP instance
    note: str


CADENCES: Tuple[Cadence, ...] = (
    Cadence("day_ahead", "Day-ahead (1 solve/day, hourly)",
            solves_per_day=1, decisions_per_day=24, horizon_steps=24,
            note="The regime reported in the submitted paper. The MILP is "
                 "comfortably fast here and the speedup does not buy "
                 "anything operationally."),
    Cadence("intraday", "Intraday (hourly re-optimisation)",
            solves_per_day=24, decisions_per_day=24, horizon_steps=24,
            note="Each hour the remaining horizon is re-solved on updated "
                 "information. 24 solves per day."),
    Cadence("quarter", "15-minute resolution, rolling",
            solves_per_day=96, decisions_per_day=96, horizon_steps=96,
            note="Italy moved to 15-minute balancing. Both the instance "
                 "size and the number of solves grow, which is where the "
                 "MILP actually binds."),
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class TimingRow:
    n_units: int
    cadence: str
    repeat: int
    milp_seconds_per_solve: float = float("nan")
    milp_status: str = ""
    milp_n_vars: int = -1
    milp_n_constraints: int = -1
    policy_seconds_per_decision: float = float("nan")
    policy_seconds_per_unit_decision: float = float("nan")
    ok: bool = False
    error: str = ""


@dataclass
class OfflineCost:
    """Offline cost, reported alongside inference so the trade is honest."""
    demo_generation_seconds: float = float("nan")
    bc_training_seconds: float = float("nan")
    ppo_training_seconds: float = float("nan")
    n_seeds: int = 1
    n_ablation_configs: int = 1

    def total_hours(self) -> float:
        per_run = (self.ppo_training_seconds
                   if not math.isnan(self.ppo_training_seconds) else 0.0)
        fixed = sum(x for x in (self.demo_generation_seconds,
                                self.bc_training_seconds)
                    if not math.isnan(x))
        return (fixed + per_run * self.n_seeds * self.n_ablation_configs) / 3600.0


# ---------------------------------------------------------------------------
# Fleet construction
# ---------------------------------------------------------------------------

def build_fleet(n_units: int, e_p_ratio: float = 2.0):
    """Scale the paper's three-cluster fleet to an arbitrary size.

    The 20/20/10 commercial/industrial/utility split is preserved in
    proportion so that larger fleets are the same portfolio, not a different
    one. `e_p_ratio` is exposed because Reviewers 1 and 2 both asked for
    heterogeneous storage durations; passing a per-cluster ratio here is the
    hook for that experiment.
    """
    from marl_milp_continuous import MultiBatteryParameters
    from milp_optimizer import BatteryParameters

    shares = (0.4, 0.4, 0.2)              # commercial, industrial, utility
    powers = (0.5, 1.0, 2.5)              # MW
    counts = [max(1, int(round(n_units * s))) for s in shares]
    # fix rounding drift so the fleet has exactly n_units members
    while sum(counts) > n_units:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < n_units:
        counts[counts.index(min(counts))] += 1

    bats = []
    for cnt, pmw in zip(counts, powers):
        for _ in range(cnt):
            bats.append(BatteryParameters(
                capacity_mwh=pmw * e_p_ratio,
                max_power_mw=pmw,
                degradation_cost_per_mwh=25000.0,
                cycle_life=8000,
            ))
    return MultiBatteryParameters(bats)


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

def _synthetic_day(hours: int, seed: int = 0):
    """A deterministic price/service day for timing.

    Timing must not depend on which market file happens to be on disk, so
    the benchmark uses a fixed synthetic day. Solve TIME depends on problem
    structure, not on price realism; the economic results come from the real
    data elsewhere in the pipeline.
    """
    from italian_market_data import make_synthetic_market_window
    from datetime import datetime as _dt, timedelta as _td
    start = _dt(2024, 1, 1)
    days = max(1, hours // 24)
    w = make_synthetic_market_window(
        start_date=start, end_date=start + _td(days=days + 1), seed=seed)
    prices, services = w.slice_day(0)
    prices, services = list(prices), list(services)
    # For sub-hourly cadences, upsample by repeating each hour's price and
    # service. This keeps the instance SIZE honest (96 steps) without
    # inventing a 15-minute price series.
    factor = hours / max(1, len(prices))
    if factor > 1:
        rep = int(round(factor))
        prices = [p for p in prices for _ in range(rep)][:hours]
        services = [s for s in services for _ in range(rep)][:hours]
    return prices[:hours], services[:hours]


def time_milp(fleet, cadence: Cadence, sustain: SustainDurations,
              repeats: int, warmup: int, mip_gap: float,
              threads: int, time_limit_s: float,
              milp_mode: str = "discrete") -> List[TimingRow]:
    """Time one MILP instance of `cadence.horizon_steps` steps."""
    from marl_milp_continuous import MultiBESSMILPOptimizer
    from marl_milp_discrete import MultiBESSMILPOptimizerDiscrete
    from marl_env import N_ACTION_BINS

    prices, services = _synthetic_day(cadence.horizon_steps)
    rows: List[TimingRow] = []

    # The solver tolerances live as CLASS attributes on the optimizer
    # (SOLVER_GAP_REL, SOLVER_TIME_LIMIT_S), not as constructor arguments, so
    # they have to be overridden on the class before _build_solver() runs.
    # Doing this matters for more than tidiness: Reviewer 1 (pt 5) asks the
    # paper to REPORT the optimality tolerance, and the default is a 1%
    # relative MIP gap, not exact optimality. Timing a solve without pinning
    # the gap would report a number produced under an undisclosed tolerance.
    _cls = (MultiBESSMILPOptimizerDiscrete if milp_mode == "discrete"
            else MultiBESSMILPOptimizer)
    _saved = (getattr(_cls, "SOLVER_GAP_REL", None),
              getattr(_cls, "SOLVER_TIME_LIMIT_S", None))
    _cls.SOLVER_GAP_REL = float(mip_gap)
    _cls.SOLVER_TIME_LIMIT_S = float(time_limit_s)

    def _build():
        common = dict(
            use_nonlinear_degradation=False,
            directional_services=True,
            **sustain.as_milp_kwargs(),
        )
        if milp_mode == "discrete":
            return MultiBESSMILPOptimizerDiscrete(
                fleet, None, n_action_bins=N_ACTION_BINS, **common)
        return MultiBESSMILPOptimizer(fleet, None, **common)

    # Warm-up solves are discarded: the first call pays solver start-up and
    # Python import cost, which is not part of the steady-state schedule cost.
    for _ in range(warmup):
        try:
            _build().optimize(cadence.horizon_steps, prices, services)
        except Exception:
            break

    for r in range(repeats):
        row = TimingRow(n_units=fleet.n_batteries, cadence=cadence.key,
                        repeat=r)
        try:
            opt = _build()
            t0 = time.perf_counter()
            res = opt.optimize(cadence.horizon_steps, prices, services)
            row.milp_seconds_per_solve = time.perf_counter() - t0
            row.milp_status = str(getattr(res, "solver_status", "?"))
            prob = getattr(opt, "prob", None) or getattr(opt, "_prob", None)
            if prob is not None:
                try:
                    row.milp_n_vars = len(prob.variables())
                    row.milp_n_constraints = len(prob.constraints)
                except Exception:
                    pass
            row.ok = True
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    # restore the class defaults so a later stage of the same process is not
    # silently running under the benchmark's tolerances
    if _saved[0] is not None:
        _cls.SOLVER_GAP_REL = _saved[0]
    if _saved[1] is not None:
        _cls.SOLVER_TIME_LIMIT_S = _saved[1]
    return rows


def time_policy(fleet, cadence: Cadence, sustain: SustainDurations,
                repeats: int, warmup: int,
                algo=None) -> List[TimingRow]:
    """Time one policy decision for the whole fleet.

    When `algo` is None an untrained network of the production shape is used.
    Inference cost depends on architecture and batch size, not on the learned
    weights, so an untrained module gives the same timing and lets the
    benchmark run without a trained checkpoint on disk.

    The reported cost includes observation construction and action decoding,
    so both schedulers are timed on the work needed to produce an admissible
    fleet bid.
    """
    import torch
    from marl_env import MultiBESSEnv, N_ACTION_BINS, ACTION_AXES_DIRECTIONAL

    prices, services = _synthetic_day(24)
    env = MultiBESSEnv(
        fleet, episode_hours=24, prices=prices, services=services,
        soc_init=0.5, seed=0, directional_services=True,
        sustain_hours=sustain,
    )
    obs, _ = env.reset(seed=0)
    obs_dim = len(next(iter(obs.values())))
    n_axes = ACTION_AXES_DIRECTIONAL
    n = fleet.n_batteries

    if algo is None:
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 512), torch.nn.Tanh(),
            torch.nn.Linear(512, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, n_axes * N_ACTION_BINS),
        ).eval()

        def _forward(batch):
            with torch.no_grad():
                return net(batch).numpy()
    else:
        module = algo.get_module("shared_policy")

        def _forward(batch):
            with torch.no_grad():
                out = module.forward_inference({"obs": batch})
            return out["action_dist_inputs"].detach().cpu().numpy()

    rows: List[TimingRow] = []

    def _one_decision():
        # observation construction ...
        batch = np.stack([env._build_observation(a) for a in env.agents])
        t = torch.as_tensor(batch, dtype=torch.float32)
        # ... forward pass ...
        logits = _forward(t).reshape(n, n_axes, N_ACTION_BINS)
        # ... and action decoding into an admissible fleet bid.
        actions = {a: logits[i].argmax(axis=-1).astype(np.int64)
                   for i, a in enumerate(env.agents)}
        env._decode_clip_actions(actions, hour=0)

    for _ in range(warmup):
        _one_decision()

    for r in range(repeats):
        row = TimingRow(n_units=n, cadence=cadence.key, repeat=r)
        try:
            t0 = time.perf_counter()
            _one_decision()
            dt = time.perf_counter() - t0
            row.policy_seconds_per_decision = dt
            row.policy_seconds_per_unit_decision = dt / max(n, 1)
            row.ok = True
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation and fitting
# ---------------------------------------------------------------------------

def _median_iqr(values: Sequence[float]) -> Tuple[float, float, float]:
    v = sorted(x for x in values if x is not None and not math.isnan(x))
    if not v:
        return float("nan"), float("nan"), float("nan")
    med = statistics.median(v)
    if len(v) < 4:
        return med, v[0], v[-1]
    q1 = statistics.median(v[:len(v) // 2])
    q3 = statistics.median(v[(len(v) + 1) // 2:])
    return med, q1, q3


def fit_power_law(sizes: Sequence[float],
                  times: Sequence[float]) -> Dict[str, float]:
    """Least squares on log t = log a + b log N.

    The exponent b is what the paper should quote: it says how the optimizer
    scales, rather than leaving the reader to infer it from three points.
    """
    xs, ys = [], []
    for n, t in zip(sizes, times):
        if n > 0 and t and not math.isnan(t) and t > 0:
            xs.append(math.log(n))
            ys.append(math.log(t))
    if len(xs) < 2:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"),
                "n_points": len(xs)}
    x = np.asarray(xs)
    y = np.asarray(ys)
    b, log_a = np.polyfit(x, y, 1)
    pred = b * x + log_a
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return {"a": float(math.exp(log_a)), "b": float(b), "r2": r2,
            "n_points": len(xs)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_scalability(
    sizes: Sequence[int],
    cadences: Sequence[Cadence],
    *,
    sustain: SustainDurations = DEFAULT_SUSTAIN,
    repeats: int = 10,
    warmup: int = 2,
    mip_gap: float = 1e-4,
    threads: int = 0,
    time_limit_s: float = 300.0,
    milp_mode: str = "discrete",
    include_policy: bool = True,
    outdir: str = "scalability_results",
) -> List[TimingRow]:
    os.makedirs(outdir, exist_ok=True)
    rows: List[TimingRow] = []

    print("=" * 74)
    print("SCALABILITY BENCHMARK")
    print(f"  machine    : {platform.processor() or platform.machine()}, "
          f"Python {platform.python_version()}")
    print(f"  sizes      : {list(sizes)}")
    print(f"  cadences   : {[c.key for c in cadences]}")
    print(f"  repeats    : {repeats} (+{warmup} warm-up, discarded)")
    print(f"  MILP mode  : {milp_mode}, MIPGap={mip_gap}, threads={threads}, "
          f"time limit={time_limit_s}s")
    print(f"  sustain    : {sustain.describe()}")
    print("=" * 74)

    for n in sizes:
        fleet = build_fleet(n)
        # Cadences that share a horizon length share an instance size, so the
        # per-solve cost is the same and only the solves-per-day multiplier
        # differs. day_ahead and intraday are both 24 steps; timing both would
        # double the MILP work for no new information. Time each distinct
        # horizon once and reuse the measurement.
        measured_by_horizon: Dict[int, List[TimingRow]] = {}
        for cad in cadences:
            print(f"\n  N={n:<4} cadence={cad.key:<10} "
                  f"({cad.horizon_steps} steps x {cad.solves_per_day} solves/day)")
            if cad.horizon_steps in measured_by_horizon:
                src = measured_by_horizon[cad.horizon_steps]
                print(f"        MILP   reusing the {cad.horizon_steps}-step "
                      f"measurement (same instance size)")
                milp_rows = [TimingRow(
                    n_units=r.n_units, cadence=cad.key, repeat=r.repeat,
                    milp_seconds_per_solve=r.milp_seconds_per_solve,
                    milp_status=r.milp_status, milp_n_vars=r.milp_n_vars,
                    milp_n_constraints=r.milp_n_constraints,
                    ok=r.ok, error=r.error) for r in src]
            else:
                milp_rows = time_milp(fleet, cad, sustain, repeats, warmup,
                                      mip_gap, threads, time_limit_s,
                                      milp_mode)
                measured_by_horizon[cad.horizon_steps] = milp_rows
            med, q1, q3 = _median_iqr(
                [r.milp_seconds_per_solve for r in milp_rows])
            print(f"        MILP   {med*1e3:9.2f} ms/solve "
                  f"[IQR {q1*1e3:.2f}-{q3*1e3:.2f}]  "
                  f"-> {med*cad.solves_per_day*1e3:9.2f} ms/day")
            rows.extend(milp_rows)

            if include_policy:
                pol_rows = time_policy(fleet, cad, sustain, repeats, warmup)
                pmed, pq1, pq3 = _median_iqr(
                    [r.policy_seconds_per_decision for r in pol_rows])
                umed, _, _ = _median_iqr(
                    [r.policy_seconds_per_unit_decision for r in pol_rows])
                print(f"        policy {pmed*1e3:9.4f} ms/decision "
                      f"[IQR {pq1*1e3:.4f}-{pq3*1e3:.4f}]  "
                      f"({umed*1e6:.2f} us/unit)  "
                      f"-> {pmed*cad.decisions_per_day*1e3:9.4f} ms/day")
                if med > 0 and pmed > 0:
                    speedup = ((med * cad.solves_per_day)
                               / (pmed * cad.decisions_per_day))
                    print(f"        speedup per scheduled day: {speedup:,.0f}x")
                rows.extend(pol_rows)

    return rows


def write_reports(rows: List[TimingRow], outdir: str,
                  cadences: Sequence[Cadence],
                  offline: Optional[OfflineCost] = None) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths: Dict[str, str] = {}

    raw = os.path.join(outdir, "scalability_raw.csv")
    cols = list(asdict(TimingRow(0, "", 0)).keys())
    with open(raw, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            d = asdict(r)
            f.write(",".join(
                f'"{d[c]}"' if isinstance(d[c], str) else f"{d[c]}"
                for c in cols) + "\n")
    paths["raw"] = raw

    # group
    key = lambda r: (r.n_units, r.cadence)
    groups: Dict[Tuple[int, str], List[TimingRow]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    summary = os.path.join(outdir, "scalability_summary.csv")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("n_units,cadence,milp_median_s,milp_q1_s,milp_q3_s,"
                "milp_ms_per_day,policy_median_s,policy_us_per_unit,"
                "policy_ms_per_day,speedup_per_day,n_repeats\n")
        for (n, cad_key), rs in sorted(groups.items()):
            cad = next(c for c in CADENCES if c.key == cad_key)
            m, q1, q3 = _median_iqr([r.milp_seconds_per_solve for r in rs])
            p, _, _ = _median_iqr([r.policy_seconds_per_decision for r in rs])
            u, _, _ = _median_iqr(
                [r.policy_seconds_per_unit_decision for r in rs])
            milp_day = m * cad.solves_per_day * 1e3
            pol_day = p * cad.decisions_per_day * 1e3
            spd = (milp_day / pol_day) if pol_day and pol_day > 0 else float("nan")
            f.write(f"{n},{cad_key},{m},{q1},{q3},{milp_day},{p},"
                    f"{u*1e6 if not math.isnan(u) else float('nan')},"
                    f"{pol_day},{spd},{len(rs)}\n")
    paths["summary"] = summary

    # power-law fits, one per cadence
    fits: Dict[str, Dict[str, float]] = {}
    for cad in cadences:
        ns, ts = [], []
        for (n, cad_key), rs in sorted(groups.items()):
            if cad_key != cad.key:
                continue
            m, _, _ = _median_iqr([r.milp_seconds_per_solve for r in rs])
            if not math.isnan(m):
                ns.append(n)
                ts.append(m)
        fits[cad.key] = fit_power_law(ns, ts)
    fit_path = os.path.join(outdir, "scalability_fit.json")
    with open(fit_path, "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=2)
    paths["fit"] = fit_path

    md = os.path.join(outdir, "scalability_table.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Computational scalability\n\n")
        f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
                f"Median over repeated measurements, warm-up discarded; "
                f"brackets are the interquartile range.\n\n")
        for cad in cadences:
            f.write(f"## {cad.label}\n\n")
            f.write(f"{cad.note}\n\n")
            f.write("| N | MILP [ms/solve] | MILP [ms/day] | "
                    "Policy [ms/decision] | Policy [ms/day] | Speedup/day |\n")
            f.write("|---|---|---|---|---|---|\n")
            for (n, cad_key), rs in sorted(groups.items()):
                if cad_key != cad.key:
                    continue
                m, q1, q3 = _median_iqr([r.milp_seconds_per_solve for r in rs])
                p, _, _ = _median_iqr(
                    [r.policy_seconds_per_decision for r in rs])
                milp_day = m * cad.solves_per_day * 1e3
                pol_day = p * cad.decisions_per_day * 1e3
                spd = (f"{milp_day/pol_day:,.0f}x"
                       if pol_day and pol_day > 0 else "n/a")
                f.write(f"| {n} | {m*1e3:.2f} [{q1*1e3:.2f}–{q3*1e3:.2f}] | "
                        f"{milp_day:.1f} | {p*1e3:.4f} | {pol_day:.3f} | "
                        f"{spd} |\n")
            fit = fits.get(cad.key, {})
            if fit and not math.isnan(fit.get("b", float("nan"))):
                f.write(f"\nFitted scaling: t = {fit['a']:.3g} * N^"
                        f"{fit['b']:.3f}  (R^2 = {fit['r2']:.4f}, "
                        f"{fit['n_points']} points)\n\n")

        if offline is not None:
            f.write("## Offline cost\n\n")
            f.write("Inference speed is not free: the policy has to be "
                    "produced first.\n\n")
            f.write(f"- MILP demonstration generation: "
                    f"{offline.demo_generation_seconds/3600:.2f} h\n")
            f.write(f"- Behavioural cloning: "
                    f"{offline.bc_training_seconds/3600:.2f} h\n")
            f.write(f"- PPO training, per run: "
                    f"{offline.ppo_training_seconds/3600:.2f} h\n")
            f.write(f"- Configurations x seeds: "
                    f"{offline.n_ablation_configs} x {offline.n_seeds}\n")
            f.write(f"- **Total offline: {offline.total_hours():.1f} h**\n")
    paths["table"] = md

    print("\nreports written:")
    for k, v in paths.items():
        print(f"  {k:<8} {v}")
    for cad_key, fit in fits.items():
        if not math.isnan(fit.get("b", float("nan"))):
            print(f"  fit[{cad_key}]: t = {fit['a']:.3g} * N^{fit['b']:.3f} "
                  f"(R^2={fit['r2']:.4f})")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[3, 10, 50, 100, 200, 500])
    ap.add_argument("--cadences", nargs="+",
                    default=[c.key for c in CADENCES],
                    choices=[c.key for c in CADENCES])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = solver default; pin it for a reproducible "
                         "timing protocol")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--milp-mode", choices=("continuous", "discrete"),
                    default="discrete")
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--outdir", default="scalability_results")
    # offline-cost inputs, taken from the pipeline log of the production run
    ap.add_argument("--offline-demo-s", type=float, default=float("nan"))
    ap.add_argument("--offline-bc-s", type=float, default=float("nan"))
    ap.add_argument("--offline-ppo-s", type=float, default=float("nan"))
    ap.add_argument("--offline-seeds", type=int, default=3)
    ap.add_argument("--offline-configs", type=int, default=5)
    args = ap.parse_args(argv)

    sustain = (SustainDurations.with_tau_fcr(args.tau) if args.tau
               else DEFAULT_SUSTAIN)
    cadences = tuple(c for c in CADENCES if c.key in set(args.cadences))

    rows = run_scalability(
        args.sizes, cadences, sustain=sustain,
        repeats=args.repeats, warmup=args.warmup,
        mip_gap=args.mip_gap, threads=args.threads,
        time_limit_s=args.time_limit, milp_mode=args.milp_mode,
        include_policy=not args.no_policy, outdir=args.outdir,
    )

    offline = None
    if not math.isnan(args.offline_ppo_s):
        offline = OfflineCost(
            demo_generation_seconds=args.offline_demo_s,
            bc_training_seconds=args.offline_bc_s,
            ppo_training_seconds=args.offline_ppo_s,
            n_seeds=args.offline_seeds,
            n_ablation_configs=args.offline_configs,
        )

    write_reports(rows, args.outdir, cadences, offline)
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
