"""
Block 2 and Block 4 verification tests.

Run from the repo root with:

    python tests_block2_block4.py

The script exercises:

  Block 2 (forecast-error calibration plug-in):
    1. ForecastErrorGenerator still produces sensible samples with the
       legacy default (no params argument).
    2. ForecastErrorGenerator.from_calibration_json correctly reads a JSON
       file produced by Block 1 and instantiates each of the four
       distribution branches.
    3. The Ornstein-Uhlenbeck process now respects a non-zero `mu`
       parameter (the empirical bias in gas-crisis 2022 is ~+2.5%).
    4. The OrnsteinUhlenbeckNoise class still works with the legacy
       constructor signature (no mu kwarg).

  Block 4 (ARERA minimum-capacity in MILP):
    5. With min_capacity = 1 MW, a feasible MILP either reserves 0 MW or
       reserves at least 1 MW; values in (0, 1) MW are infeasible.
    6. When a service is unavailable for an hour, the binary indicator
       and the reservation are forced to zero.

Each test prints PASS/FAIL with a short diagnostic. Exit status is 0 if
all tests pass, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


# ============================================================================
# Block 2 tests
# ============================================================================

def test_block2_backward_compat():
    print("\n--- Block 2.1: backward compatibility of ForecastErrorGenerator ---")
    from drl_flexibility_analysis import ForecastErrorGenerator
    for dist in ['uniform', 'normal', 'ornstein-uhlenbeck', 'fixed-bias']:
        np.random.seed(0)
        gen = ForecastErrorGenerator(dist)  # no params: legacy behaviour
        gen.reset()
        samples = np.array([gen.generate_error() for _ in range(500)])
        finite = np.isfinite(samples).all()
        in_range = (samples >= -1.5).all() and (samples <= 1.5).all()
        _check(finite and in_range,
               f"legacy {dist} samples are finite and in [-1.5, +1.5]",
               f"mean={samples.mean():+.4f}, std={samples.std():.4f}")


def test_block2_ou_with_mu():
    print("\n--- Block 2.2: OU now reverts to a non-zero mean ---")
    from drl_flexibility_analysis import OrnsteinUhlenbeckNoise
    # Reproducibility
    np.random.seed(42)
    # Long-run mean of 0.025 (gas-crisis 2022 calibrated bias), strong reversion
    ou = OrnsteinUhlenbeckNoise(theta=0.5, mu=0.025, sigma=0.05,
                                 clip_low=-1.0, clip_high=+1.0)
    samples = np.array([ou.sample() for _ in range(10_000)])
    # Drop the burn-in (first ~100 samples; with theta=0.5 this is way more
    # than enough for the chain to mix).
    samples = samples[100:]
    empirical_mean = float(samples.mean())
    # Tolerance: 2% of mu, plus a 0.005 absolute floor for tiny mu.
    tol = max(0.005, 0.02 * abs(0.025))
    _check(abs(empirical_mean - 0.025) < tol,
           "OU sample mean approaches mu=0.025",
           f"empirical={empirical_mean:+.5f}, target=+0.02500, tol={tol}")


def test_block2_legacy_ou_signature():
    print("\n--- Block 2.3: legacy OU(theta, sigma) signature still works ---")
    from drl_flexibility_analysis import OrnsteinUhlenbeckNoise
    np.random.seed(7)
    ou = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2)  # legacy positional
    samples = [ou.sample() for _ in range(10)]
    _check(all(np.isfinite(samples)),
           "legacy OU(theta, sigma) constructor is still callable",
           f"samples[:3]={samples[:3]}")


def test_block2_from_json():
    print("\n--- Block 2.4: ForecastErrorGenerator.from_calibration_json ---")
    from drl_flexibility_analysis import ForecastErrorGenerator
    # Create a minimal calibrations.json compatible with the Block-1 schema
    cal = {
        "normalized_2023_2024": {
            "normal":             {"mu": 0.006, "sigma": 0.10},
            "uniform":            {"loc":  0.006, "half_width": 0.20},
            "ornstein_uhlenbeck": {"theta": 0.43, "mu": 0.006, "sigma": 0.085},
            "fixed_bias":         {"bias": 0.006},
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cal, f)
        json_path = f.name

    for dist in ['uniform', 'normal', 'ornstein-uhlenbeck', 'fixed-bias']:
        gen = ForecastErrorGenerator.from_calibration_json(
            json_path=json_path, regime='normalized_2023_2024',
            error_type=dist, wide_clip=True)
        np.random.seed(123)
        gen.reset()
        samples = np.array([gen.generate_error() for _ in range(500)])
        _check(np.isfinite(samples).all(),
               f"from_calibration_json({dist}) generates finite samples",
               f"mean={samples.mean():+.4f}, std={samples.std():.4f}")

    Path(json_path).unlink()


# ============================================================================
# Block 4 tests
# ============================================================================

def _build_milp(min_capacity_mw=1.0, n_hours=24, fcr_available=True):
    """Helper: build a MILPOptimizer + a trivial price/service vector."""
    from milp_optimizer import BatteryParameters, MILPOptimizer
    from flexibility_market import FlexibilityService, ServiceType

    bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0)
    optim = MILPOptimizer(bp, None)  # MILP doesn't strictly need the market

    # Generate a synthetic FlexibilityService per hour. To keep the test
    # fast and the problem feasible, we set a small capacity tariff and
    # zero energy tariff (the MILP can choose to reserve or not).
    services = []
    for t in range(n_hours):
        per_hour = []
        if fcr_available:
            per_hour.append(FlexibilityService(
                service_type=ServiceType.FCR,
                capacity_price=10.0,
                energy_price=0.0,
                activation_probability=0.1,
                response_time=30,
                min_capacity=min_capacity_mw,
                max_duration=4,
                min_duration=1,
            ))
        services.append(per_hour)

    # Flat price profile -> arbitrage is unprofitable; the optimizer's
    # only revenue is from reserving FCR (or not).
    prices = [50.0] * n_hours
    return optim, prices, services


def test_block4_minimum_capacity():
    print("\n--- Block 4.1: MILP enforces R_fcr in {0} U [min_cap, P_max] ---")
    optim, prices, services = _build_milp(min_capacity_mw=1.0, n_hours=24)
    result = optim.optimize(time_horizon=24, energy_prices=prices,
                            flexibility_services=services)
    if result.solver_status != "Optimal":
        _check(False, "MILP solves to optimality on the toy instance",
               f"status={result.solver_status}")
        return

    fcr = np.array(result.flexibility_reservations.get('FCR', []))
    # Either 0 or in [1.0, 2.0] for every hour; allow a small numerical
    # tolerance because CBC may return values a hair below 1.0 or above 2.0.
    EPS = 1e-4
    is_zero    = np.abs(fcr) <= EPS
    is_in_band = (fcr >= 1.0 - EPS) & (fcr <= 2.0 + EPS)
    valid = is_zero | is_in_band
    bad = np.where(~valid)[0]
    _check(valid.all(),
           "every R_fcr is either 0 or in [min_cap, P_max]",
           f"violations at hours {list(bad)}; values={fcr[bad].tolist()[:5]}"
           if len(bad) else f"all {len(fcr)} hours obey the rule "
                              f"(unique values rounded: "
                              f"{sorted(set(np.round(fcr, 4).tolist()))[:6]})")


def test_block4_service_absent():
    print("\n--- Block 4.2: MILP forces R_s = 0 when service is unavailable ---")
    optim, prices, services = _build_milp(min_capacity_mw=1.0, n_hours=24,
                                          fcr_available=False)
    result = optim.optimize(time_horizon=24, energy_prices=prices,
                            flexibility_services=services)
    if result.solver_status != "Optimal":
        _check(False, "MILP solves on the no-FCR instance",
               f"status={result.solver_status}")
        return
    fcr = np.array(result.flexibility_reservations.get('FCR', []))
    _check(np.allclose(fcr, 0.0, atol=1e-6),
           "FCR reservation is identically zero when service is absent",
           f"max={float(np.abs(fcr).max()):.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Block 2 + Block 4 verification tests")
    print("=" * 70)
    test_block2_backward_compat()
    test_block2_ou_with_mu()
    test_block2_legacy_ou_signature()
    test_block2_from_json()
    test_block4_minimum_capacity()
    test_block4_service_absent()

    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
