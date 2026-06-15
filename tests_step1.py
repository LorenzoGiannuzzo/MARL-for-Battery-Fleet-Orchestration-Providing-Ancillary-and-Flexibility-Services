"""
Step 1 verification: multi-BESS MILP with BSP aggregation.

Tests:
  1.1 N=1 multi-BESS reproduces the single-BESS MILP NET PROFIT to numerical
      precision. NB: the formulations are not bit-identical because the
      multi-BESS code uses an aggregate participation binary (B_agg_s,t)
      rather than the per-BESS B_s,t from Block 4. With N=1 they are
      mathematically equivalent and the optima coincide; the test asserts
      this directly. Sanity check that the multi-BESS pipeline is sound.
  1.2 N=2 with two IDENTICAL batteries produces a net profit equal to twice
      the N=1 case. This validates that linear scaling holds, that
      constraints are correctly indexed per-BESS, and that the aggregation
      is correctly computed. Critical consistency check.
  1.3 The per-service flexibility revenue breakdown sums exactly to the
      total flexibility_revenue (carries over from Block 6 to the multi-BESS
      formulation). Critical for downstream reporting.
  1.4 With N=2 batteries having DIFFERENT max_power_mw, the optimiser
      allocates more reservation to the larger battery, all else equal.
      Sanity check that heterogeneity is respected.
  1.5 With N=3 small batteries (each max_power_mw < 1 MW), the AGGREGATE
      minimum capacity of 1 MW can still be met by combining contributions
      across BESS, even though no individual BESS could meet it alone. This
      is the central feature of the BSP framing.

Run:

    python tests_step1.py
"""

from __future__ import annotations

import sys
import warnings
import numpy as np

warnings.simplefilter("ignore")

_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def _services_24h(min_cap=1.0):
    """Build a 24h flexibility-service catalog with all three services.

    Tariffs chosen so that aFRR dominates FCR and mFRR (matching the real
    Italian setup). Award probability set to 1.0 so the test results are
    deterministic.
    """
    from flexibility_market import FlexibilityService, ServiceType
    services = []
    for t in range(24):
        services.append([
            FlexibilityService(
                service_type=ServiceType.FCR, capacity_price=20.0,
                energy_price=80.0, activation_probability=0.10,
                response_time=30, min_capacity=min_cap,
                max_duration=4, min_duration=1, award_probability=1.0,
            ),
            FlexibilityService(
                service_type=ServiceType.AFRR, capacity_price=55.0,
                energy_price=120.0, activation_probability=0.30,
                response_time=200, min_capacity=min_cap,
                max_duration=4, min_duration=1, award_probability=1.0,
            ),
            FlexibilityService(
                service_type=ServiceType.MFRR, capacity_price=15.0,
                energy_price=100.0, activation_probability=0.20,
                response_time=900, min_capacity=min_cap,
                max_duration=4, min_duration=1, award_probability=1.0,
            ),
        ])
    return services


def _prices_24h():
    return [50.0 + 30.0 * ((t % 24) - 12) / 12.0 for t in range(24)]


def test_n1_matches_single():
    print("\n--- Step 1.1: N=1 multi-BESS matches single-BESS NET PROFIT ---")
    from milp_optimizer import BatteryParameters, MILPOptimizer
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)

    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    prices = _prices_24h()
    services = _services_24h()

    # Single-BESS reference
    single = MILPOptimizer(bp, None)
    res_single = single.optimize(time_horizon=24, energy_prices=prices,
                                  flexibility_services=services)

    # Multi-BESS with N=1
    multi_params = MultiBatteryParameters(batteries=[bp])
    multi = MultiBESSMILPOptimizer(multi_params, None)
    res_multi = multi.optimize(time_horizon=24, energy_prices=prices,
                                flexibility_services=services)

    print(f"    single net profit = {res_single.net_profit:.4f}")
    print(f"    multi  net profit = {res_multi.net_profit:.4f}")
    delta = abs(res_single.net_profit - res_multi.net_profit)
    _check(delta < 1e-3,
           "N=1 multi reproduces single-BESS net profit",
           f"|delta|={delta:.6f}")
    _check(res_multi.n_batteries == 1, "n_batteries == 1")
    _check(res_multi.solver_status == "Optimal",
           "N=1 solver returns Optimal")


def test_n2_doubles_n1():
    print("\n--- Step 1.2: N=2 identical batteries produce exactly 2x N=1 ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)

    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    prices = _prices_24h()
    services = _services_24h()

    # N=1
    one = MultiBESSMILPOptimizer(MultiBatteryParameters([bp]), None)
    r1 = one.optimize(24, prices, services)

    # N=2 (identical)
    two = MultiBESSMILPOptimizer(MultiBatteryParameters([bp, bp]), None)
    r2 = two.optimize(24, prices, services)

    expected = 2.0 * r1.net_profit
    delta_rel = abs(r2.net_profit - expected) / max(abs(expected), 1.0)
    print(f"    N=1 net = {r1.net_profit:.4f}")
    print(f"    N=2 net = {r2.net_profit:.4f}")
    print(f"    expected 2*N=1 = {expected:.4f}, relative delta = {delta_rel:.2e}")
    _check(delta_rel < 1e-3,
           "N=2 net profit equals 2*N=1 (linear scaling)",
           f"rel delta = {delta_rel:.4e}")
    _check(r2.n_batteries == 2, "n_batteries == 2")

    # Per-BESS profits should split roughly evenly between the two identical
    # batteries (allow some tolerance for solver tie-breaking)
    arb_split = r2.per_battery_arbitrage_profit
    print(f"    per-BESS arbitrage profit: {arb_split}")
    total_arb = sum(arb_split)
    if abs(total_arb) > 0.01:
        share_min = min(arb_split) / total_arb
        _check(share_min >= 0.20,
               "neither identical BESS is starved of arbitrage",
               f"shares = {[a/total_arb for a in arb_split]}")


def test_breakdown_sums():
    print("\n--- Step 1.3: per-service flex revenue sums to total ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    prices = _prices_24h()
    services = _services_24h()

    opt = MultiBESSMILPOptimizer(MultiBatteryParameters([bp, bp]), None)
    res = opt.optimize(24, prices, services)
    bd = res.flexibility_revenue_by_service
    bd_sum = sum(bd.values())
    print(f"    breakdown = {bd}")
    print(f"    sum = {bd_sum:.4f}, total = {res.flexibility_revenue:.4f}")
    _check(abs(bd_sum - res.flexibility_revenue) < 1e-4,
           "FCR + aFRR + mFRR == flexibility_revenue",
           f"|delta|={abs(bd_sum - res.flexibility_revenue):.6f}")

    # And the full reconciliation: arbitrage + flex_breakdown - degradation == net
    recon = res.arbitrage_profit + bd_sum - res.degradation_cost
    _check(abs(recon - res.net_profit) < 1e-3,
           "arbitrage + breakdown - degradation == net_profit",
           f"recon={recon:.4f} vs net={res.net_profit:.4f}")


def test_heterogeneous_allocation():
    print("\n--- Step 1.4: bigger battery receives bigger reservation ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)
    small = BatteryParameters(capacity_mwh=1.0, max_power_mw=0.5)
    large = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)

    prices = _prices_24h()
    services = _services_24h()

    opt = MultiBESSMILPOptimizer(MultiBatteryParameters([small, large]), None)
    res = opt.optimize(24, prices, services)

    # aFRR is the dominant service. Total aFRR reservation across hours.
    afrr_small = sum(res.flexibility_reservations['aFRR'][0])
    afrr_large = sum(res.flexibility_reservations['aFRR'][1])
    print(f"    small (0.5 MW) total aFRR reservation across 24h: {afrr_small:.2f}")
    print(f"    large (2.0 MW) total aFRR reservation across 24h: {afrr_large:.2f}")
    _check(afrr_large > afrr_small,
           "larger BESS contributes more aFRR than smaller BESS",
           f"large={afrr_large:.2f} > small={afrr_small:.2f}")
    # Hourly small reservation never exceeds small.max_power_mw
    per_hour_small = res.flexibility_reservations['aFRR'][0]
    _check(max(per_hour_small) <= 0.5 + 1e-6,
           "small BESS hourly aFRR <= its own P_max (0.5 MW)",
           f"max hourly = {max(per_hour_small):.4f}")


def test_aggregate_minimum_via_small_batteries():
    print("\n--- Step 1.5: aggregate 1 MW minimum met by combining small BESS ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import (MultiBatteryParameters,
                                      MultiBESSMILPOptimizer)

    # Three BESS each with P_max = 0.4 MW. Individually none can meet the
    # 1 MW aggregate minimum, but their sum (1.2 MW) can.
    small = BatteryParameters(capacity_mwh=1.0, max_power_mw=0.4)
    multi_params = MultiBatteryParameters([small, small, small])
    prices = _prices_24h()
    services = _services_24h()

    opt = MultiBESSMILPOptimizer(multi_params, None)
    res = opt.optimize(24, prices, services)
    print(f"    solver status: {res.solver_status}")
    print(f"    aggregate aFRR per hour: {[round(x, 3) for x in res.aggregate_reservations['aFRR'][:6]]}...")
    # For hours where the aggregator chooses to participate, the sum should
    # be >= 1 MW (the ARERA aggregate minimum)
    participating_hours = [t for t in range(24)
                            if res.aggregate_reservations['aFRR'][t] > 0.01]
    _check(len(participating_hours) > 0,
           "aggregator participates in aFRR for at least some hours",
           f"{len(participating_hours)} hours active")
    if participating_hours:
        min_active = min(res.aggregate_reservations['aFRR'][t]
                         for t in participating_hours)
        _check(min_active >= 1.0 - 1e-4,
               "every participating hour respects 1 MW aggregate minimum",
               f"min during active hours = {min_active:.4f}")


def main():
    print("=" * 70)
    print("Step 1: multi-BESS MILP regression and consistency tests")
    print("=" * 70)
    test_n1_matches_single()
    test_n2_doubles_n1()
    test_breakdown_sums()
    test_heterogeneous_allocation()
    test_aggregate_minimum_via_small_batteries()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 1 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
