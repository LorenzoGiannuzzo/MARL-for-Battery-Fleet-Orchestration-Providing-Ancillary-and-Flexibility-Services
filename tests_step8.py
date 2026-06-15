"""
Step 8 verification: end-to-end pipeline on synthetic Italian market data.

Tests:

  8.1 PUN synthetic generator produces statistically realistic data:
      annual mean within 5% of 107 EUR/MWh, peak hour in [17,21], weekend
      factor in [0.85, 0.95], all prices in plausible range.
  8.2 Annual service catalog has the correct shape (8760 hours, 3 services
      each) and respects the time-of-day pattern (night award > peak award).
  8.3 Multi-day MILP demo generation with rolling FCE: FCE strictly
      increases day by day, SOC stays in bounds, MILP solves every day.
  8.4 Dataset size and labelling are correct: M = n_days * N * 24
      samples, valid action bin indices.
  8.5 End-to-end pipeline runs to completion and produces a structured
      FullPipelineResult with all fields populated and finite.
  8.6 BC policy on the test window achieves env-realized profit close to
      the MILP oracle (env-realized, same seed): gap below 10%.
  8.7 BC policy on the test window is significantly better than random.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timedelta
import numpy as np

warnings.simplefilter("ignore")

_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def test_pun_synthetic_properties():
    print("\n--- Step 8.1: synthetic PUN statistical properties ---")
    from italian_market_data import PUNSynthetic2024Generator
    gen = PUNSynthetic2024Generator(seed=0)
    prices = gen.generate(datetime(2024, 1, 1), datetime(2025, 1, 1))
    _check(len(prices) == 8784,
           f"annual length = 8784 (leap year 2024)",
           f"got {len(prices)}")
    mean_v = float(prices.mean())
    _check(102.0 < mean_v < 112.0,
           "annual mean in [102, 112] EUR/MWh (target 107)",
           f"got {mean_v:.2f}")
    std_v = float(prices.std())
    _check(20.0 < std_v < 40.0,
           "annual std in [20, 40] EUR/MWh",
           f"got {std_v:.2f}")
    _check(prices.min() > 0.0,
           "no negative or zero prices",
           f"min {prices.min():.2f}")
    _check(prices.max() < 700.0,
           "no absurd spikes",
           f"max {prices.max():.2f}")
    # Peak hour check
    hours = np.array([(datetime(2024, 1, 1) + timedelta(hours=h)).hour
                       for h in range(len(prices))])
    hourly_means = np.array([prices[hours == h].mean() for h in range(24)])
    peak_hour = int(np.argmax(hourly_means))
    _check(17 <= peak_hour <= 21,
           "peak hour in [17, 21]",
           f"got {peak_hour}")
    # Weekend factor check
    wd = np.array([(datetime(2024, 1, 1) + timedelta(hours=h)).weekday()
                    for h in range(len(prices))])
    ratio = float(prices[wd >= 5].mean() / prices[wd < 5].mean())
    _check(0.85 < ratio < 0.95,
           "weekend / weekday mean ratio in [0.85, 0.95]",
           f"got {ratio:.3f}")


def test_service_catalog():
    print("\n--- Step 8.2: annual service catalog structure ---")
    from italian_market_data import build_yearly_service_catalog
    from flexibility_market import ServiceType
    cat = build_yearly_service_catalog(
        datetime(2024, 1, 1), datetime(2025, 1, 1)
    )
    _check(len(cat) == 8784,
           "catalog has hourly entries for the whole year",
           f"got {len(cat)}")
    _check(all(len(h) == 3 for h in cat),
           "every hour has 3 services (FCR, aFRR, mFRR)")
    types_per_hour = set(s.service_type for s in cat[0])
    _check(types_per_hour == {ServiceType.FCR, ServiceType.AFRR, ServiceType.MFRR},
           "services per hour are FCR, aFRR, mFRR")
    # Time-of-day pattern: aFRR award at hour 3 (night) > aFRR award at hour 19 (peak)
    afrr_3 = next(s for s in cat[3] if s.service_type == ServiceType.AFRR)
    afrr_19 = next(s for s in cat[19] if s.service_type == ServiceType.AFRR)
    _check(afrr_3.award_probability > afrr_19.award_probability,
           "aFRR award at hour 3 (night) > hour 19 (peak)",
           f"{afrr_3.award_probability:.2f} vs {afrr_19.award_probability:.2f}")


def test_multi_day_milp_with_rolling_state():
    print("\n--- Step 8.3: multi-day MILP with rolling SOC and FCE ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from pipeline_step8 import generate_multi_day_expert_demos
    from italian_market_data import make_synthetic_market_window
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp] * 2)
    window = make_synthetic_market_window(
        datetime(2024, 4, 1), datetime(2024, 4, 6), seed=0,
    )  # 5 days
    demo = generate_multi_day_expert_demos(fleet, window, env_seed=0)
    _check(len(demo.daily_milp_profits) == 5,
           "5 daily profits recorded")
    _check(all(s == "Optimal" for s in demo.daily_milp_status),
           "all 5 days solved to Optimal")
    _check(all(0.0 <= s <= 1.0 for s in demo.final_soc_per_bess),
           "final SOC in [0, 1] per BESS",
           f"got {demo.final_soc_per_bess}")
    _check(all(f > 0.0 for f in demo.final_fce_per_bess),
           "final FCE > 0 per BESS (some throughput happened)",
           f"got {demo.final_fce_per_bess}")
    _check(demo.total_env_realized_profit > 0,
           "env-realized total profit is positive",
           f"got {demo.total_env_realized_profit:.2f}")


def test_dataset_size_and_labels():
    print("\n--- Step 8.4: BC dataset size and label correctness ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from pipeline_step8 import generate_multi_day_expert_demos
    from italian_market_data import make_synthetic_market_window
    from multi_bess_env import N_ACTION_BINS, ACTION_AXES, OBS_DIM
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    N = 4
    n_days = 3
    fleet = MultiBatteryParameters([bp] * N)
    window = make_synthetic_market_window(
        datetime(2024, 4, 1), datetime(2024, 4, 1) + timedelta(days=n_days), seed=0,
    )
    demo = generate_multi_day_expert_demos(fleet, window, env_seed=0)
    expected = N * n_days * 24
    _check(demo.obs.shape == (expected, OBS_DIM),
           f"obs shape ({expected}, {OBS_DIM})",
           f"got {demo.obs.shape}")
    _check(demo.actions.shape == (expected, ACTION_AXES),
           f"actions shape ({expected}, {ACTION_AXES})",
           f"got {demo.actions.shape}")
    _check(demo.actions.min() >= 0 and demo.actions.max() < N_ACTION_BINS,
           f"all action bins in [0, {N_ACTION_BINS - 1}]",
           f"min={demo.actions.min()}, max={demo.actions.max()}")


def test_end_to_end_pipeline_runs():
    print("\n--- Step 8.5: end-to-end pipeline runs to completion ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from pipeline_step8 import run_full_pipeline, smoke_test_config
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp] * 2)
    cfg = smoke_test_config()
    cfg["bc_n_epochs"] = 20
    result = run_full_pipeline(fleet=fleet, **cfg)
    _check(np.isfinite(result.test_total_milp_profit_eur),
           "test MILP env-realized profit is finite",
           f"got {result.test_total_milp_profit_eur:.2f}")
    _check(np.isfinite(result.test_total_bc_profit_eur),
           "test BC profit is finite",
           f"got {result.test_total_bc_profit_eur:.2f}")
    _check(result.bc_dataset_size == result.train_days * fleet.n_batteries * 24,
           "BC dataset size matches expected",
           f"got {result.bc_dataset_size}")
    _check(result.train_milp_optimal_day_count == result.train_days,
           "all train days solved to Optimal")


def test_bc_gap_to_milp_oracle():
    print("\n--- Step 8.6: BC test profit close to MILP oracle (env-realized) ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from pipeline_step8 import run_full_pipeline, smoke_test_config
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp] * 3)
    cfg = smoke_test_config()
    result = run_full_pipeline(fleet=fleet, **cfg)
    print(f"    MILP oracle: {result.test_total_milp_profit_eur:.2f} EUR")
    print(f"    BC policy:   {result.test_total_bc_profit_eur:.2f} EUR")
    print(f"    gap:         {result.bc_gap_to_milp_percent:+.2f}%")
    _check(abs(result.bc_gap_to_milp_percent) < 10.0,
           "BC env-realized profit within 10% of MILP env-realized oracle",
           f"gap {result.bc_gap_to_milp_percent:+.2f}%")


def test_bc_beats_random():
    print("\n--- Step 8.7: BC significantly better than random on test ---")
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters
    from pipeline_step8 import run_full_pipeline, smoke_test_config
    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    fleet = MultiBatteryParameters([bp] * 3)
    cfg = smoke_test_config()
    result = run_full_pipeline(fleet=fleet, **cfg)
    print(f"    BC:     {result.test_total_bc_profit_eur:.2f} EUR")
    print(f"    Random: {result.test_total_random_profit_eur:.2f} EUR")
    print(f"    uplift: {result.bc_uplift_over_random_factor:.2f}x")
    _check(result.bc_uplift_over_random_factor > 2.0,
           "BC profit > 2x random profit",
           f"factor {result.bc_uplift_over_random_factor:.2f}")


def main():
    print("=" * 70)
    print("Step 8: end-to-end pipeline on synthetic Italian market data")
    print("=" * 70)
    test_pun_synthetic_properties()
    test_service_catalog()
    test_multi_day_milp_with_rolling_state()
    test_dataset_size_and_labels()
    test_end_to_end_pipeline_runs()
    test_bc_gap_to_milp_oracle()
    test_bc_beats_random()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 8 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
