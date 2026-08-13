# -*- coding: utf-8 -*-
"""
preflight.py — validate a run configuration before spending a day on it.

WHY
---
A production cell costs roughly a day. Every configuration error found in the
first minute of that day is a day lost, and this project has already lost runs
to a missing dataclass field and to a keyword the callee did not accept. The
checks below construct everything a run touches, in the cheapest possible
form, so a mistake surfaces in seconds.

It deliberately mirrors the ablation's own command line, so the invocation
under test is the invocation that will run:

    python preflight.py --durations 4 2 1 --connection-fraction 0.6 \\
                        --centralized-critic --per-cluster-policies

WHAT IT CHECKS
--------------
  1. the fleet builds and its totals are what the flags imply
  2. the sustain scenario is inside the regulatory range
  3. the coupling reaches BOTH the MILP and the environment, and the MILP
     schedule survives the environment's projection unchanged
  4. the observation width is consistent across every stage
  5. the MILP solves one short horizon and reports Optimal
  6. the behavioural-clone input dimension matches the environment
  7. the RLlib configuration builds, including the centralised critic module
     spec and the per-cluster policy mapping

Steps 6 and 7 need torch and RLlib. Where they are missing the check is
reported as SKIPPED rather than passed, because a skipped check is not
evidence of anything and saying otherwise is how a run gets launched on an
untested path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

RESULTS = []


def report(name, ok, detail=""):
    tag = "PASS " if ok is True else ("SKIP " if ok is None else "FAIL ")
    RESULTS.append((name, ok))
    print(f"  {tag} {name}" + (f"  |  {detail}" if detail else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--durations", type=float, nargs=3, default=None,
                    metavar=("COMMERCIAL", "INDUSTRIAL", "UTILITY"))
    ap.add_argument("--connection-limit", type=float, default=None)
    ap.add_argument("--connection-fraction", type=float, default=None)
    ap.add_argument("--centralized-critic", action="store_true")
    ap.add_argument("--per-cluster-policies", action="store_true")
    ap.add_argument("--no-duration-features", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--milp-mode", choices=("continuous", "discrete"),
                    default="discrete")
    ap.add_argument("--horizon", type=int, default=4,
                    help="hours for the trial MILP solve; keep it small")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("=" * 70)
    print("PREFLIGHT")
    print("=" * 70)

    # ---------------------------------------------------------------- 0
    # Static first: a keyword the callee rejects has twice killed a run at the
    # line that used it, minutes or hours in. This costs a second.
    try:
        from check_call_signatures import main as _sigcheck
        import contextlib, io as _io
        _buf = _io.StringIO()
        with contextlib.redirect_stdout(_buf):
            rc = _sigcheck(os.path.dirname(os.path.abspath(__file__)) or ".")
        report("every call passes arguments its callee accepts", rc == 0,
               _buf.getvalue().strip().splitlines()[-1] if _buf.getvalue() else "")
        if rc != 0:
            for ln in _buf.getvalue().strip().splitlines()[2:]:
                print(f"        {ln.strip()}")
    except Exception as exc:
        report("every call passes arguments its callee accepts", None,
               f"{type(exc).__name__}: {exc}")

    try:
        from check_world_consistency import main as _worldcheck
        _buf = _io.StringIO()
        with contextlib.redirect_stdout(_buf):
            rc = _worldcheck(os.path.dirname(os.path.abspath(__file__)) or ".")
        report("every stage is built in the same world", rc == 0,
               _buf.getvalue().strip().splitlines()[0] if _buf.getvalue() else "")
        if rc != 0:
            for ln in _buf.getvalue().strip().splitlines()[2:]:
                print(f"        {ln.strip()}")
    except Exception as exc:
        report("every stage is built in the same world", None,
               f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 1
    from marl_milp_continuous import MultiBatteryParameters
    durations = tuple(args.durations) if args.durations else None
    fleet = MultiBatteryParameters.heterogeneous_fleet(
        seed=0, durations=durations)
    tp = sum(b.max_power_mw for b in fleet.batteries)
    te = sum(b.capacity_mwh for b in fleet.batteries)
    ratios = sorted({round(b.capacity_mwh / b.max_power_mw, 3)
                     for b in fleet.batteries})
    report("fleet builds", True,
           f"{fleet.n_batteries} units, {tp:.1f} MW, {te:.1f} MWh, E/P {ratios}")
    if durations:
        report("durations are heterogeneous", len(ratios) > 1,
               f"distinct E/P ratios: {ratios}")

    # ---------------------------------------------------------------- 2
    from market_constants import SustainDurations, DEFAULT_SUSTAIN
    sustain = (SustainDurations.with_tau_fcr(args.tau) if args.tau
               else DEFAULT_SUSTAIN)
    report("sustain scenario", sustain.is_sogl_compliant(), sustain.describe())

    # ---------------------------------------------------------------- 3
    from fleet_coupling import (FleetCoupling, direction_envelopes,
                                project_to_connection)
    if args.connection_limit is not None and args.connection_fraction is not None:
        report("coupling flags", False,
               "give either --connection-limit or --connection-fraction")
        return _finish(t0)
    if args.connection_limit is not None:
        coupling = FleetCoupling.connection(args.connection_limit)
    elif args.connection_fraction is not None:
        coupling = FleetCoupling.as_fraction_of_fleet(
            fleet, args.connection_fraction)
    else:
        coupling = FleetCoupling.uncoupled()
    report("coupling", True, coupling.describe())
    if coupling.is_active:
        binds = coupling.connection_limit_mw < tp
        report("the coupling can actually bind", binds,
               f"limit {coupling.connection_limit_mw:.1f} MW against "
               f"{tp:.1f} MW of rated power"
               + ("" if binds else "  <-- it will never bind"))

    # ---------------------------------------------------------------- 4
    from marl_env import MultiBESSEnv, OBS_DIM, GLOBAL_STATE_DIM
    duration_features = not args.no_duration_features
    env = MultiBESSEnv(
        fleet, episode_hours=24, soc_init=0.5, seed=0,
        directional_services=True, sustain_hours=sustain, coupling=coupling,
        duration_features=duration_features,
        centralized_critic=args.centralized_critic)
    env.reset(seed=0)
    width = env.observation_space("bess_0").shape[0]
    built = len(env._build_observation("bess_0"))
    expected = OBS_DIM + (GLOBAL_STATE_DIM if args.centralized_critic else 0)
    report("observation width is consistent", width == built == expected,
           f"declared {width}, built {built}, expected {expected}")

    # ---------------------------------------------------------------- 5
    from italian_market_data import make_synthetic_market_window
    from marl_milp_continuous import MultiBESSMILPOptimizer
    from marl_milp_discrete import MultiBESSMILPOptimizerDiscrete
    from marl_env import N_ACTION_BINS
    start = datetime(2024, 1, 1)
    w = make_synthetic_market_window(
        start_date=start, end_date=start + timedelta(days=2), seed=0,
        directional=True)
    prices, services = (lambda p, s: (list(p), list(s)))(*w.slice_day(0))
    H = max(1, int(args.horizon))
    common = dict(use_nonlinear_degradation=False, directional_services=True,
                  coupling=coupling, **sustain.as_milp_kwargs())
    opt = (MultiBESSMILPOptimizerDiscrete(fleet, None,
                                          n_action_bins=N_ACTION_BINS, **common)
           if args.milp_mode == "discrete"
           else MultiBESSMILPOptimizer(fleet, None, **common))
    t_solve = time.time()
    res = opt.optimize(H, prices[:H], services[:H])
    solve_s = time.time() - t_solve
    report(f"the MILP solves {H} hours", res.solver_status == "Optimal",
           f"status {res.solver_status}, {solve_s:.1f}s "
           f"-> about {solve_s*24/H/60:.1f} min per 24h day")

    n_conn = sum(1 for k in opt.problem.constraints
                 if k.startswith("ConnUp") or k.startswith("ConnDn"))
    if coupling.is_active:
        report("the MILP carries the connection constraints", n_conn == 2 * H,
               f"{n_conn} constraints for {H} hours")

    # MILP schedule must pass the environment's projection untouched
    RES = res.flexibility_reservations
    KM = {"R_fcr": "FCR", "R_afrr_up": "aFRR_up", "R_afrr_dn": "aFRR_dn",
          "R_mfrr_up": "mFRR_up", "R_mfrr_dn": "mFRR_dn"}
    touched = 0
    for t in range(H):
        bid = {}
        for i in range(fleet.n_batteries):
            net = res.power_trajectories[i][t]
            b = {"P_discharge": max(0.0, net), "P_charge": max(0.0, -net)}
            for ours, theirs in KM.items():
                b[ours] = RES[theirs][i][t]
            bid[f"bess_{i}"] = b
        if project_to_connection(bid, coupling.connection_limit_mw)["binding"]:
            touched += 1
    report("MILP/env parity: the projection is a no-op on the MILP schedule",
           touched == 0, f"{touched} of {H} hours would be re-scaled")

    # ---------------------------------------------------------------- 6
    try:
        from marl_bc import BCPolicyNet
        net = BCPolicyNet(obs_dim=width, n_axes=7)
        import torch
        with torch.no_grad():
            out = net(torch.zeros(2, width, dtype=torch.float32))
        report("the clone accepts the environment's observation", True,
               f"input {width}, output {tuple(out.shape)}")
    except ImportError as exc:
        report("the clone accepts the environment's observation", None,
               f"torch unavailable here: {exc}")
    except Exception as exc:
        report("the clone accepts the environment's observation", False,
               f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 7
    try:
        from marl_trainer import build_default_config, cluster_sizes_from_fleet
        cfg = build_default_config(
            n_batteries=fleet.n_batteries, episode_hours=24,
            directional_services=True, sustain_hours=sustain,
            coupling=coupling, duration_features=duration_features,
            centralized_critic=args.centralized_critic,
            per_cluster_sizes=(cluster_sizes_from_fleet(fleet)
                               if args.per_cluster_policies else None),
        )
        report("the RLlib configuration builds", True,
               f"policies: {sorted(cfg.policies) if cfg.policies else 'default'}")
        if args.centralized_critic:
            spec = getattr(cfg, "rl_module_spec", None)
            n_mod = len(getattr(spec, "rl_module_specs", {}) or {})
            n_pol = len(cfg.policies or [])
            report("one centralised module per trained policy",
                   n_mod == n_pol and n_mod > 0,
                   f"{n_mod} module specs for {n_pol} policies")
    except ImportError as exc:
        report("the RLlib configuration builds", None,
               f"ray/torch unavailable here: {exc}")
    except Exception as exc:
        report("the RLlib configuration builds", False,
               f"{type(exc).__name__}: {exc}")

    return _finish(t0)


def _finish(t0):
    print("=" * 70)
    failed = [n for n, ok in RESULTS if ok is False]
    skipped = [n for n, ok in RESULTS if ok is None]
    print(f"{len(RESULTS)} checks in {time.time()-t0:.1f}s: "
          f"{len(RESULTS)-len(failed)-len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    for n in skipped:
        print(f"  SKIPPED (not evidence of anything): {n}")
    if failed:
        print("\nDO NOT START THE RUN. Failed:")
        for n in failed:
            print(f"  - {n}")
        return 1
    if skipped:
        print("\nNo failures, but the skipped checks were never exercised "
              "here. Re-run this on the training machine before committing "
              "a day of compute.")
    else:
        print("\nAll clear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())