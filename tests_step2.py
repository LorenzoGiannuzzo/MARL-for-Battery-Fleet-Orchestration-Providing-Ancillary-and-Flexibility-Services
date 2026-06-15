"""
Step 2 verification: heterogeneous N=50 fleet under BSP aggregation.

Tests:
  2.1 heterogeneous_fleet(seed=0) builds a 50-BESS portfolio with the
      expected cluster breakdown (20 commercial, 20 industrial, 10 utility)
      and total nameplate capacity (110 MWh) and power (55 MW). Cluster
      classification of each battery via cluster_of(i) is consistent with
      the sizing. Reproducibility of the SOC sampling with a fixed seed.
  2.2 The MILP solves to optimality at N=50 on a 24h horizon within a
      target wall-clock of 60 s (consumer CPU). This is the operational
      feasibility check: if CBC could not chew this in under a minute, the
      whole multi-BESS pipeline would be impractical for a daily rolling
      horizon over a year.
  2.3 Cluster allocation is sensible: larger BESS contribute proportionally
      more to the aggregate aFRR bid than smaller BESS. Specifically, the
      utility cluster (10 BESS, 25 MW nameplate) should contribute more
      total aFRR reservation across the day than the commercial cluster
      (20 BESS, 10 MW nameplate), even though commercial has more BESS.
      This validates that the optimiser rewards capacity, not headcount.
  2.4 The aggregate aFRR bid hourly trajectory is plausibly bounded by the
      fleet nameplate, and the per-BESS hourly reservation is bounded by
      each BESS's own P_max (no cross-contamination of constraints).
  2.5 Cumulative daily solve over a 5-day rolling horizon stays within
      target. Conservative ceiling: 5 minutes wall-clock for 5 days, i.e.
      under 1 min/day on average, consistent with Test 2.2.

Run:

    python tests_step2.py
"""

from __future__ import annotations

import sys
import time
import warnings
import numpy as np

warnings.simplefilter("ignore")

_FAILED = []
SOLVE_TIME_TARGET_S = 60.0  # per-day MILP solve time ceiling


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def _services_24h(award_full=True):
    """Build a 24h service catalog. award_full=True keeps the aFRR award
    probability at 1.0 across all hours (deterministic comparison)."""
    from flexibility_market import FlexibilityService, ServiceType
    services = []
    for t in range(24):
        services.append([
            FlexibilityService(
                service_type=ServiceType.FCR, capacity_price=20.0,
                energy_price=80.0, activation_probability=0.10,
                response_time=30, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=1.0 if award_full else 0.5,
            ),
            FlexibilityService(
                service_type=ServiceType.AFRR, capacity_price=55.0,
                energy_price=120.0, activation_probability=0.30,
                response_time=200, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=1.0 if award_full else 0.5,
            ),
            FlexibilityService(
                service_type=ServiceType.MFRR, capacity_price=15.0,
                energy_price=100.0, activation_probability=0.20,
                response_time=900, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=1.0 if award_full else 0.5,
            ),
        ])
    return services


def _prices_24h(base=80.0, swing=40.0):
    return [base + swing * np.sin(2 * np.pi * (t - 6) / 24) for t in range(24)]


def test_fleet_construction():
    print("\n--- Step 2.1: heterogeneous_fleet construction and breakdown ---")
    from milp_optimizer_multi import MultiBatteryParameters

    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    _check(fleet.n_batteries == 50,
           "default heterogeneous fleet has 50 BESS",
           f"got {fleet.n_batteries}")

    bd = fleet.cluster_breakdown()
    print(f"    cluster breakdown: {bd}")
    _check(bd == {'commercial': 20, 'industrial': 20, 'utility': 10},
           "cluster counts are (20, 20, 10) as designed",
           f"got {bd}")

    expected_pmax = 20 * 0.5 + 20 * 1.0 + 10 * 2.5  # 10 + 20 + 25 = 55 MW
    expected_emwh = 20 * 1.0 + 20 * 2.0 + 10 * 5.0  # 20 + 40 + 50 = 110 MWh
    _check(abs(fleet.total_max_power_mw - expected_pmax) < 1e-9,
           f"total nameplate power = {expected_pmax} MW",
           f"got {fleet.total_max_power_mw}")
    _check(abs(fleet.total_capacity_mwh - expected_emwh) < 1e-9,
           f"total nameplate energy = {expected_emwh} MWh",
           f"got {fleet.total_capacity_mwh}")

    # Reproducibility of SOC sampling
    fleet_a = MultiBatteryParameters.heterogeneous_fleet(seed=42)
    fleet_b = MultiBatteryParameters.heterogeneous_fleet(seed=42)
    soc_a = [b.initial_soc for b in fleet_a.batteries]
    soc_b = [b.initial_soc for b in fleet_b.batteries]
    _check(soc_a == soc_b,
           "SOC sampling is reproducible with a fixed seed")

    # Cluster classification of each battery is consistent with its sizing
    consistent = all(
        (fleet.cluster_of(i) == 'commercial' and fleet.batteries[i].capacity_mwh == 1.0) or
        (fleet.cluster_of(i) == 'industrial' and fleet.batteries[i].capacity_mwh == 2.0) or
        (fleet.cluster_of(i) == 'utility'    and fleet.batteries[i].capacity_mwh == 5.0)
        for i in range(50)
    )
    _check(consistent, "cluster_of(i) matches the sizing of each BESS")


def test_solve_time_n50():
    print("\n--- Step 2.2: MILP solves N=50 in target wall-clock ---")
    from milp_optimizer_multi import MultiBatteryParameters, MultiBESSMILPOptimizer

    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    prices = _prices_24h()
    services = _services_24h(award_full=True)

    opt = MultiBESSMILPOptimizer(fleet, None)
    t0 = time.time()
    res = opt.optimize(time_horizon=24, energy_prices=prices,
                        flexibility_services=services)
    elapsed = time.time() - t0

    print(f"    solve time: {elapsed:.2f}s (target < {SOLVE_TIME_TARGET_S:.0f}s)")
    print(f"    solver status: {res.solver_status}")
    print(f"    aggregate net profit (1 day): {res.net_profit:.2f} EUR")
    print(f"    aggregate aFRR reservation total: "
          f"{sum(res.aggregate_reservations['aFRR']):.2f} MWh")

    _check(res.solver_status == "Optimal",
           "CBC returns Optimal on N=50 fleet")
    _check(elapsed < SOLVE_TIME_TARGET_S,
           f"solve time under {SOLVE_TIME_TARGET_S}s",
           f"got {elapsed:.2f}s")


def test_cluster_allocation():
    print("\n--- Step 2.3: utility cluster contributes more aFRR than commercial ---")
    from milp_optimizer_multi import MultiBatteryParameters, MultiBESSMILPOptimizer

    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    prices = _prices_24h()
    services = _services_24h(award_full=True)
    opt = MultiBESSMILPOptimizer(fleet, None)
    res = opt.optimize(time_horizon=24, energy_prices=prices,
                        flexibility_services=services)

    # Aggregate aFRR reservation by cluster, summed across hours
    afrr_by_cluster = {'commercial': 0.0, 'industrial': 0.0, 'utility': 0.0}
    for i in range(fleet.n_batteries):
        cluster = fleet.cluster_of(i)
        total_i = sum(res.flexibility_reservations['aFRR'][i])
        afrr_by_cluster[cluster] += total_i

    print(f"    aggregate aFRR MWh-day by cluster: {afrr_by_cluster}")

    # Utility cluster (10 BESS × 2.5 MW = 25 MW) should beat
    # commercial cluster (20 BESS × 0.5 MW = 10 MW)
    _check(afrr_by_cluster['utility'] > afrr_by_cluster['commercial'],
           "utility cluster contributes more aFRR than commercial cluster",
           f"util={afrr_by_cluster['utility']:.1f} > "
           f"comm={afrr_by_cluster['commercial']:.1f}")

    # Per-cluster contribution roughly proportional to total nameplate
    util_pmax = 10 * 2.5
    comm_pmax = 20 * 0.5
    expected_ratio = util_pmax / comm_pmax  # = 5.0
    actual_ratio = afrr_by_cluster['utility'] / max(afrr_by_cluster['commercial'], 1e-6)
    print(f"    aFRR ratio utility/commercial: {actual_ratio:.2f} "
          f"(nameplate ratio: {expected_ratio:.2f})")
    _check(abs(actual_ratio - expected_ratio) / expected_ratio < 0.10,
           "aFRR ratio matches the cluster nameplate ratio within 10%",
           f"actual {actual_ratio:.2f} vs expected {expected_ratio:.2f}")


def test_per_bess_bounds():
    print("\n--- Step 2.4: per-BESS reservations respect individual P_max ---")
    from milp_optimizer_multi import MultiBatteryParameters, MultiBESSMILPOptimizer

    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    prices = _prices_24h()
    services = _services_24h(award_full=True)
    opt = MultiBESSMILPOptimizer(fleet, None)
    res = opt.optimize(24, prices, services)

    # For every battery, every hour: sum of reservations <= its P_max
    violations = 0
    max_violation = 0.0
    for i in range(fleet.n_batteries):
        pmax = fleet.batteries[i].max_power_mw
        for t in range(24):
            tot = (res.flexibility_reservations['FCR'][i][t]
                   + res.flexibility_reservations['aFRR'][i][t]
                   + res.flexibility_reservations['mFRR'][i][t])
            if tot > pmax + 1e-4:
                violations += 1
                max_violation = max(max_violation, tot - pmax)
    _check(violations == 0,
           "no per-BESS reservation exceeds individual P_max",
           f"{violations} violations, max = {max_violation:.4f} MW")

    # Aggregate aFRR per hour bounded by fleet nameplate
    pmax_total = fleet.total_max_power_mw
    over = max(res.aggregate_reservations['aFRR']) - pmax_total
    _check(over <= 1e-4,
           f"aggregate aFRR per hour <= fleet nameplate ({pmax_total} MW)",
           f"max excess = {over:.4f} MW")


def test_rolling_5days():
    print("\n--- Step 2.5: 5-day rolling solve stays within budget ---")
    from milp_optimizer_multi import MultiBatteryParameters, MultiBESSMILPOptimizer

    fleet = MultiBatteryParameters.heterogeneous_fleet(seed=0)
    services = _services_24h(award_full=True)

    daily_times = []
    cumulative_net = 0.0
    opt = MultiBESSMILPOptimizer(fleet, None)
    current_soc = [b.initial_soc for b in fleet.batteries]
    for d in range(5):
        prices = _prices_24h(base=80.0 + d * 5.0, swing=40.0)
        t0 = time.time()
        res = opt.optimize(24, prices, services, initial_soc=current_soc)
        daily_times.append(time.time() - t0)
        cumulative_net += res.net_profit
        # Roll the SOC forward
        current_soc = [traj[-1] for traj in res.soc_trajectories]

    avg_t = float(np.mean(daily_times))
    max_t = float(max(daily_times))
    print(f"    per-day solve times: {[round(x, 2) for x in daily_times]} s")
    print(f"    avg = {avg_t:.2f}s, max = {max_t:.2f}s")
    print(f"    cumulative 5-day net profit: {cumulative_net:.2f} EUR")

    _check(max_t < SOLVE_TIME_TARGET_S,
           f"max per-day solve time under {SOLVE_TIME_TARGET_S}s",
           f"max = {max_t:.2f}s")
    _check(avg_t < SOLVE_TIME_TARGET_S * 0.5,
           f"average per-day solve time under {SOLVE_TIME_TARGET_S/2}s",
           f"avg = {avg_t:.2f}s")
    # Extrapolate to a full year
    yearly_budget_s = avg_t * 366
    print(f"    extrapolated full-year budget: {yearly_budget_s/60:.1f} minutes")


def main():
    print("=" * 70)
    print("Step 2: heterogeneous N=50 fleet multi-BESS MILP tests")
    print("=" * 70)
    test_fleet_construction()
    test_solve_time_n50()
    test_cluster_allocation()
    test_per_bess_bounds()
    test_rolling_5days()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 2 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
