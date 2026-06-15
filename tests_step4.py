"""
Step 4 verification: physics-based linear degradation rate in multi-BESS MILP.

The corrected Step 4 approach: replace the legacy amortised constant rate
`degradation_cost_per_mwh / (2 * cycle_life)` with a per-BESS LINEAR rate
computed from the Schmalstieg+Naumann LFP cycle aging model as the AVERAGE
marginal cost over the expected daily throughput at the BESS's current
cumulative throughput Q0. The MILP within each daily horizon is still an
LP (linear cost rate), but the COEFFICIENT now reflects:

  - per-BESS heterogeneity (capacity, cycle_life)
  - the BESS's current age (Q0 in FCE)
  - assumed operating DoD

What it still misses (and DRL captures):
  - DoD dependence (fourth-power stress, c_dod_a=4)
  - SOC-dependent calendar aging
  - Within-horizon non-linearity (small)
"""

from __future__ import annotations

import sys
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")

_FAILED = []
SOLVE_TIME_TARGET_S = 60.0


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def _services_24h(award=1.0):
    from flexibility_market import FlexibilityService, ServiceType
    return [
        [
            FlexibilityService(
                service_type=ServiceType.FCR, capacity_price=20.0,
                energy_price=80.0, activation_probability=0.10,
                response_time=30, min_capacity=1.0,
                max_duration=4, min_duration=1, award_probability=award,
            ),
            FlexibilityService(
                service_type=ServiceType.AFRR, capacity_price=55.0,
                energy_price=120.0, activation_probability=0.30,
                response_time=200, min_capacity=1.0,
                max_duration=4, min_duration=1, award_probability=award,
            ),
            FlexibilityService(
                service_type=ServiceType.MFRR, capacity_price=15.0,
                energy_price=100.0, activation_probability=0.20,
                response_time=900, min_capacity=1.0,
                max_duration=4, min_duration=1, award_probability=award,
            ),
        ]
        for _ in range(24)
    ]


def _prices_24h(base=80.0, swing=40.0):
    return [base + swing * np.sin(2 * np.pi * (t - 6) / 24) for t in range(24)]


def test_avg_rate_properties():
    print("\n--- Step 4.1: average physics rate properties ---")
    from degradation_model import LFPBatteryState
    state = LFPBatteryState(capacity_mwh=4.0)
    state.fce_cumulative = 0.0
    r0 = state.avg_marginal_cycle_cost_per_mwh(
        dod=0.6, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    print(f"    fresh (FCE0=0):     rate = {r0:.2f} EUR/MWh")
    _check(0.0 < r0 < 1e6,
           "avg rate at FCE0=0 is finite and positive",
           f"got {r0:.2f}")
    state.fce_cumulative = 2000.0
    r2k = state.avg_marginal_cycle_cost_per_mwh(
        dod=0.6, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    print(f"    mid-life (FCE0=2000): rate = {r2k:.4f} EUR/MWh")
    _check(r2k < r0,
           "rate decreases monotonically as Q0 grows",
           f"FCE0=0 ({r0:.2f}) > FCE0=2000 ({r2k:.4f})")
    state.fce_cumulative = 8000.0
    r8k = state.avg_marginal_cycle_cost_per_mwh(
        dod=0.6, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    print(f"    EoL    (FCE0=8000): rate = {r8k:.4f} EUR/MWh")
    _check(r8k < r2k,
           "rate continues to decrease as Q0 grows further")
    state.fce_cumulative = 1000.0
    r_low = state.avg_marginal_cycle_cost_per_mwh(
        dod=0.3, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    r_mid = state.avg_marginal_cycle_cost_per_mwh(
        dod=0.6, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    r_high = state.avg_marginal_cycle_cost_per_mwh(
        dod=1.0, Q_estimated_daily_mwh=8.0, replacement_cost=200000.0,
    )
    print(f"    DoD sensitivity at FCE0=1000:")
    print(f"      DoD 0.3: {r_low:.4f} EUR/MWh")
    print(f"      DoD 0.6: {r_mid:.4f} EUR/MWh")
    print(f"      DoD 1.0: {r_high:.4f} EUR/MWh")
    _check(r_low < r_mid < r_high,
           "rate strictly increases with DoD")


def test_linear_mode_regression():
    print("\n--- Step 4.2: linear-mode MILP regression ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp, bp])
    prices = _prices_24h()
    services = _services_24h(award=1.0)
    opt_legacy = MultiBESSMILPOptimizer(fleet, None)
    res_legacy = opt_legacy.optimize(24, prices, services)
    opt_off = MultiBESSMILPOptimizer(fleet, None, use_nonlinear_degradation=False)
    res_off = opt_off.optimize(24, prices, services)
    delta = abs(res_legacy.net_profit - res_off.net_profit)
    _check(delta < 1e-4,
           "use_nonlinear_degradation=False reproduces legacy mode",
           f"|delta|={delta:.6f}")


def test_nonlinear_mode_consistency():
    print("\n--- Step 4.3: non-linear mode runs and reports consistently ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp])
    prices = _prices_24h()
    services = _services_24h(award=1.0)
    opt = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=True,
        nonlinear_replacement_cost=200000.0, nonlinear_dod_assumed=0.6,
        fce_cumulative_initial=[500.0],
    )
    res = opt.optimize(24, prices, services)
    _check(res.solver_status == "Optimal",
           "non-linear MILP solves to Optimal")
    rate = opt._physics_rates_eur_per_mwh[0]
    total_tp = sum(opt.variables['P_charge'][(0, t)].varValue
                   + opt.variables['P_discharge'][(0, t)].varValue
                   + opt.variables['A_fcr'][(0, t)].varValue
                   + opt.variables['A_afrr'][(0, t)].varValue
                   + opt.variables['A_mfrr'][(0, t)].varValue
                   for t in range(24))
    expected_deg = rate * total_tp
    print(f"    per-BESS physics rate: {rate:.4f} EUR/MWh")
    print(f"    realised throughput:   {total_tp:.2f} MWh")
    print(f"    reported degradation:  {res.degradation_cost:.2f} EUR")
    print(f"    expected (rate * tp):  {expected_deg:.2f} EUR")
    _check(abs(res.degradation_cost - expected_deg) < 1e-3,
           "degradation_cost = rate * throughput",
           f"|delta|={abs(res.degradation_cost - expected_deg):.6f}")


def test_fresh_vs_used_behaviour():
    print("\n--- Step 4.4: fresh battery has higher rate than used ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp])
    prices = _prices_24h()
    services = _services_24h(award=1.0)
    opt_fresh = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=True,
        nonlinear_dod_assumed=0.6, fce_cumulative_initial=[0.0],
    )
    res_fresh = opt_fresh.optimize(24, prices, services)
    rate_fresh = opt_fresh._physics_rates_eur_per_mwh[0]
    opt_used = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=True,
        nonlinear_dod_assumed=0.6, fce_cumulative_initial=[3000.0],
    )
    res_used = opt_used.optimize(24, prices, services)
    rate_used = opt_used._physics_rates_eur_per_mwh[0]
    print(f"    fresh (FCE0=0):    rate = {rate_fresh:.4f} EUR/MWh, "
          f"profit = {res_fresh.net_profit:.2f}, "
          f"deg = {res_fresh.degradation_cost:.2f}")
    print(f"    used  (FCE0=3000): rate = {rate_used:.4f} EUR/MWh, "
          f"profit = {res_used.net_profit:.2f}, "
          f"deg = {res_used.degradation_cost:.2f}")
    _check(rate_fresh > rate_used,
           "fresh rate > used rate",
           f"fresh={rate_fresh:.4f} > used={rate_used:.4f}")
    _check(res_used.net_profit >= res_fresh.net_profit - 1e-3,
           "used battery achieves >= profit of fresh battery",
           f"used={res_used.net_profit:.2f} >= fresh={res_fresh.net_profit:.2f}")


def test_heterogeneous_rates():
    print("\n--- Step 4.5: heterogeneous fleet has differentiated rates ---")
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    fce_init = [100.0] * fleet.n_batteries
    opt = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=True,
        nonlinear_dod_assumed=0.6, fce_cumulative_initial=fce_init,
    )
    rates = opt._physics_rates_eur_per_mwh
    cluster_rates = {'commercial': [], 'industrial': [], 'utility': []}
    for i in range(fleet.n_batteries):
        cluster_rates[fleet.cluster_of(i)].append(rates[i])
    avg_rates = {c: float(np.mean(rs)) for c, rs in cluster_rates.items()}
    print(f"    average physics rate by cluster (EUR/MWh):")
    for c, r in avg_rates.items():
        print(f"      {c:11s}: {r:.4f}")
    # Daily throughput estimate scales with capacity (2 * capacity_mwh).
    # The chord slope (avg_rate) depends on FCE_0 and dQ_FCE which both
    # scale identically with capacity, so the rate is approximately the
    # SAME across clusters per MWh. The test verifies the absolute values
    # are positive and within an order of magnitude of each other.
    _check(all(r > 0 for r in avg_rates.values()),
           "all clusters have positive degradation rates")
    max_r, min_r = max(avg_rates.values()), min(avg_rates.values())
    _check(max_r / min_r < 10.0,
           "cluster rates differ by less than 10x (sizing effect)",
           f"max/min = {max_r/min_r:.2f}")
    # Expected ordering: smaller BESS see higher per-MWh rate because each
    # MWh of throughput corresponds to a larger fraction of an FCE for them
    _check(avg_rates['commercial'] > avg_rates['utility'],
           "commercial cluster has higher per-MWh rate than utility "
           "(smaller capacity => higher FCE per MWh)",
           f"comm={avg_rates['commercial']:.4f} > "
           f"util={avg_rates['utility']:.4f}")


def test_solve_time_n50_nonlinear():
    print("\n--- Step 4.6: N=50 non-linear MILP solves within target ---")
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    prices = _prices_24h()
    services = _services_24h(award=1.0)
    fce_init = [100.0] * fleet.n_batteries
    opt = MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=True,
        nonlinear_dod_assumed=0.6, fce_cumulative_initial=fce_init,
    )
    t0 = time.time()
    res = opt.optimize(24, prices, services)
    elapsed = time.time() - t0
    print(f"    solve time: {elapsed:.2f}s (target < {SOLVE_TIME_TARGET_S}s)")
    print(f"    status: {res.solver_status}, net profit: {res.net_profit:.2f}")
    _check(res.solver_status == "Optimal",
           "N=50 non-linear MILP solves to Optimal")
    _check(elapsed < SOLVE_TIME_TARGET_S,
           f"solve time under {SOLVE_TIME_TARGET_S}s",
           f"got {elapsed:.2f}s")


def main():
    print("=" * 70)
    print("Step 4: physics-based linear degradation rate in multi-BESS MILP")
    print("=" * 70)
    test_avg_rate_properties()
    test_linear_mode_regression()
    test_nonlinear_mode_consistency()
    test_fresh_vs_used_behaviour()
    test_heterogeneous_rates()
    test_solve_time_n50_nonlinear()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 4 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
