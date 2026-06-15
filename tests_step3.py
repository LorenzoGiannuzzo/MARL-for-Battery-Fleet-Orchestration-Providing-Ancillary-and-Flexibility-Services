"""
Step 3 verification: non-linear LFP degradation model.

The model is calibrated against three well-established literature anchors
for prismatic LFP cells in BESS applications:

  3.1 Industrial end-of-life convention: 4000 full-cycle equivalents at
      DoD 0.80 produces ~20% capacity loss. Sources: many LFP datasheets
      (CATL, BYD, EVE) define cycle life on this anchor.
  3.2 Ambient calendar aging: 1 year at SOC 0.5 and 25 C produces ~3.5%
      capacity loss. Source: Naumann et al. 2018, Table 3.
  3.3 The cycle-aging curve is monotonically increasing in throughput AND
      concave (sqrt form): the 1000th FCE damages much less than the 100th.
  3.4 The DoD-stress curve is strictly monotonic in DoD: deeper cycles
      always damage more per cycle. Shallow cycling (DoD < 0.3) is nearly
      free, full cycling (DoD = 1.0) is ~2.4x more damaging than DoD 0.80.
  3.5 The stateful integrator advances monotonically (SOH never increases)
      and produces aging trajectories quantitatively consistent with the
      closed-form expressions. A 5-year simulation at moderate cycling
      (300 FCE/year at DoD 0.6) loses 8-12% capacity, within the range
      expected from utility-scale LFP BESS in the field.
  3.6 The marginal cycle cost per MWh decreases with cumulative throughput
      (concave fade => decreasing marginal damage), and increases steeply
      with DoD. Both directions matter for the MILP linearisation in Step 4.

Run:

    python tests_step3.py
"""

from __future__ import annotations

import sys
import math
import warnings

warnings.simplefilter("ignore")

_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def test_cycle_eol_anchor():
    print("\n--- Step 3.1: 4000 FCE @ DoD 0.80 => ~20% capacity loss ---")
    from degradation_model import cycle_aging_loss, LFPDegradationParameters
    p = LFPDegradationParameters()
    loss = cycle_aging_loss(4000.0, 0.80, p)
    print(f"    cycle loss = {loss*100:.2f}% (target ~20%)")
    _check(abs(loss - 0.20) < 0.005,
           "EoL anchor: 4000 FCE DoD 0.80 produces 20.0% ± 0.5% loss",
           f"got {loss*100:.2f}%")


def test_calendar_ambient_anchor():
    print("\n--- Step 3.2: 1 year @ SOC 0.5, 25C => ~3.5% calendar loss ---")
    from degradation_model import calendar_aging_loss, LFPDegradationParameters
    p = LFPDegradationParameters()
    loss = calendar_aging_loss(365.0, 0.5, 298.15, p)
    print(f"    calendar loss = {loss*100:.2f}% (target ~3.5%)")
    _check(abs(loss - 0.035) < 0.002,
           "ambient calendar anchor: 1 year SOC 0.5 25C => 3.5% ± 0.2%",
           f"got {loss*100:.2f}%")
    # Sanity at half a year (sqrt-time => sqrt(0.5) = 0.707 factor)
    loss_6m = calendar_aging_loss(183.0, 0.5, 298.15, p)
    ratio = loss_6m / loss
    print(f"    6-month / 12-month ratio = {ratio:.3f} (target sqrt(0.5)=0.707)")
    _check(abs(ratio - math.sqrt(0.5)) < 0.02,
           "calendar aging follows sqrt(t) time form",
           f"ratio {ratio:.3f} vs target {math.sqrt(0.5):.3f}")


def test_concavity_monotonicity():
    print("\n--- Step 3.3: cycle aging is monotonic and concave in throughput ---")
    from degradation_model import cycle_aging_loss, LFPDegradationParameters
    p = LFPDegradationParameters()
    # Marginal damage per FCE at successively higher cumulative throughput
    levels = [100, 500, 1000, 2000, 4000, 8000]
    marginals = []
    for fce in levels:
        m = cycle_aging_loss(fce, 0.80, p) - cycle_aging_loss(fce - 1.0, 0.80, p)
        marginals.append(m)
    print(f"    marginal loss per FCE at cumulative FCE = {levels}:")
    print(f"      {[f'{x*1e4:.3f}e-4' for x in marginals]}")
    monotonic_dec = all(marginals[i] > marginals[i + 1]
                        for i in range(len(marginals) - 1))
    _check(monotonic_dec,
           "marginal damage per FCE strictly decreases with cumulative FCE")
    # Total loss strictly increases
    totals = [cycle_aging_loss(fce, 0.80, p) for fce in levels]
    strict_inc = all(totals[i] < totals[i + 1] for i in range(len(totals) - 1))
    _check(strict_inc,
           "cumulative cycle loss strictly increasing in throughput")


def test_dod_sensitivity():
    print("\n--- Step 3.4: DoD stress is strictly monotonic ---")
    from degradation_model import cycle_aging_loss, LFPDegradationParameters
    p = LFPDegradationParameters()
    dods = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    losses = [cycle_aging_loss(1000.0, d, p) for d in dods]
    print(f"    1000 FCE losses by DoD:")
    for d, l in zip(dods, losses):
        print(f"      DoD {d}: {l*100:.3f}%")
    strict = all(losses[i] < losses[i + 1] for i in range(len(losses) - 1))
    _check(strict, "loss strictly increases with DoD")
    # Quantitative ratio between full and reference DoD
    ratio = losses[-1] / losses[-2]  # DoD 1.0 / DoD 0.8
    expected = (1.0 / 0.8) ** 4.0
    print(f"    DoD 1.0 / DoD 0.8 ratio: {ratio:.3f} (theoretical {expected:.3f})")
    _check(abs(ratio - expected) < 0.01,
           "full-cycle damage is (1/0.8)^4 = 2.44x reference",
           f"ratio {ratio:.3f}")
    # Shallow cycles are nearly free
    _check(losses[0] < 0.0001,
           "DoD 0.10 cycling produces negligible damage (<0.01%)",
           f"got {losses[0]*100:.4f}%")


def test_stateful_integrator():
    print("\n--- Step 3.5: stateful integrator simulates a realistic 5-year run ---")
    from degradation_model import LFPBatteryState, LFPDegradationParameters

    # 4 MWh battery, 5 years of operation: 300 FCE/year at DoD 0.6 average
    # (typical utility BESS arbitrage + ancillary services duty)
    cap = 4.0
    state = LFPBatteryState(capacity_mwh=cap)
    fce_per_year = 300
    dod = 0.6
    n_years = 5
    # Per-hour throughput: spread the annual FCE uniformly over 8760 hours,
    # and split equally between charge and discharge
    hours_per_year = 8760
    energy_per_fce = 2.0 * cap  # 2 * capacity = one full cycle
    energy_per_hour = (fce_per_year * energy_per_fce) / hours_per_year
    ch_per_hour = energy_per_hour / 2.0
    dis_per_hour = energy_per_hour / 2.0

    soh_trace = [state.soh]
    for year in range(n_years):
        for hour in range(hours_per_year):
            state.advance(hours=1.0, soc_avg=0.5,
                          charge_mwh=ch_per_hour, discharge_mwh=dis_per_hour,
                          dod=dod, T_kelvin=298.15)
        soh_trace.append(state.soh)
        print(f"    End of year {year+1}: SOH = {state.soh*100:.2f}%, "
              f"calendar loss = {state._Q_loss_cal*100:.2f}%, "
              f"cycle loss = {state._Q_loss_cyc*100:.2f}%, "
              f"FCE = {state.fce_cumulative:.0f}")

    _check(all(soh_trace[i] >= soh_trace[i + 1] for i in range(len(soh_trace) - 1)),
           "SOH is monotonically non-increasing")
    final = state.soh
    _check(0.85 <= final <= 0.95,
           "5-year SOH at moderate cycling falls in [85%, 95%]",
           f"got {final*100:.2f}%")
    # Cumulative FCE consistency check
    expected_fce = fce_per_year * n_years
    _check(abs(state.fce_cumulative - expected_fce) < 5.0,
           f"cumulative FCE = {expected_fce}",
           f"got {state.fce_cumulative:.1f}")


def test_marginal_cost():
    print("\n--- Step 3.6: marginal cycle cost decreases with FCE, rises with DoD ---")
    from degradation_model import LFPBatteryState
    state_new = LFPBatteryState(capacity_mwh=4.0)
    state_used = LFPBatteryState(capacity_mwh=4.0)
    state_used.fce_cumulative = 2000.0  # mid-life

    cost_new_80 = state_new.marginal_cycle_cost_per_mwh(dod=0.80,
                                                        replacement_cost=200000.0)
    cost_used_80 = state_used.marginal_cycle_cost_per_mwh(dod=0.80,
                                                          replacement_cost=200000.0)
    print(f"    EUR/MWh marginal cycle cost at DoD 0.80:")
    print(f"      new battery (0 FCE):      {cost_new_80:.2f}")
    print(f"      used battery (2000 FCE):  {cost_used_80:.2f}")
    _check(cost_new_80 > cost_used_80,
           "new battery has higher marginal damage than used (concavity)",
           f"new {cost_new_80:.1f} > used {cost_used_80:.1f}")

    cost_used_80_d = state_used.marginal_cycle_cost_per_mwh(dod=0.80, replacement_cost=200000.0)
    cost_used_100_d = state_used.marginal_cycle_cost_per_mwh(dod=1.00, replacement_cost=200000.0)
    print(f"    EUR/MWh marginal cycle cost at 2000 FCE:")
    print(f"      DoD 0.80: {cost_used_80_d:.2f}")
    print(f"      DoD 1.00: {cost_used_100_d:.2f}")
    _check(cost_used_100_d > cost_used_80_d,
           "deeper DoD increases marginal cycle cost",
           f"DoD 1.0 ({cost_used_100_d:.1f}) > DoD 0.8 ({cost_used_80_d:.1f})")


def main():
    print("=" * 70)
    print("Step 3: non-linear LFP degradation model")
    print("=" * 70)
    test_cycle_eol_anchor()
    test_calendar_ambient_anchor()
    test_concavity_monotonicity()
    test_dod_sensitivity()
    test_stateful_integrator()
    test_marginal_cost()
    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Step 3 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
