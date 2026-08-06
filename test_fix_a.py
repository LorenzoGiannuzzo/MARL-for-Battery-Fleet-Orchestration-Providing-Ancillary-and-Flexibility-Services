"""FIX A unit tests: JOINT sustain-budget enforcement in MultiBESSEnv.

Runs on the project machine with: python test_fix_a.py
(no data files needed; tests _decode_clip_actions directly).

Tests
-----
1. exploit_demo   : with enforce_joint_sustain=False (legacy), a saturating
                    bid jointly commits MORE energy than the SoC can sustain,
                    even though every per-service cap passes -> the hole.
2. joint_enforced : with the fix ON (default), the summed committed energy
                    per direction never exceeds the SoC budget.
3. milp_noop      : bids that satisfy the MILP's own (tighter) sustain
                    constraints pass through the decode UNCHANGED, so the
                    MILP replay and the BC clone are no-ops by construction.
"""
import numpy as np

from market_constants import DEFAULT_SUSTAIN, SustainDurations
from marl_env import MultiBESSEnv, N_ACTION_BINS

# Sustain windows under test. Read from the single source of truth
# (market_constants) instead of the hard-coded 4.0/1.0/2.0 this file used
# to carry: FIX A's joint budget must hold for ANY scenario, so the test
# has to track whatever scenario the env was given, not a fixed triple.
SUSTAIN = DEFAULT_SUSTAIN
FCR_T, AFRR_T, MFRR_T = SUSTAIN.fcr, SUSTAIN.afrr, SUSTAIN.mfrr
TOL = 1e-9


class _BP:
    """Duck-typed battery params: only the fields the decode reads."""
    def __init__(self, p_mw, e_mwh, eff):
        self.max_power_mw = p_mw
        self.capacity_mwh = e_mwh
        self.efficiency = eff


class _Fleet:
    def __init__(self, bats):
        self.batteries = bats
        self.n_batteries = len(bats)


def make_env(soc, p_mw=2.5, e_mwh=5.0, eff=0.95, enforce=True):
    """Build a minimal env instance around _decode_clip_actions only."""
    env = MultiBESSEnv.__new__(MultiBESSEnv)
    env.agents = ["bess_0"]
    env.agent_name_mapping = {"bess_0": 0}
    env.multi_params = _Fleet([_BP(p_mw, e_mwh, eff)])
    env._socs = np.array([soc], dtype=np.float64)
    env.soc_min, env.soc_max = 0.1, 0.9
    env.directional_services = True
    env.enable_commitments = False
    env.enforce_joint_sustain = bool(enforce)
    # __new__ bypasses __init__, so every attribute _decode_clip_actions
    # touches has to be set by hand. `sustain` joined that list when the
    # sustain windows moved out of hard-coded literals and into
    # market_constants.
    env.sustain = SUSTAIN
    return env


def committed_energy(out):
    up = (out["R_fcr"] * FCR_T + out["R_afrr_up"] * AFRR_T
          + out["R_mfrr_up"] * MFRR_T)
    dn = (out["R_fcr"] * FCR_T + out["R_afrr_dn"] * AFRR_T
          + out["R_mfrr_dn"] * MFRR_T)
    return up, dn


def saturating_action():
    """All reserve axes at max bin, no arbitrage: the farming bid."""
    a = np.zeros(7, dtype=np.int64)
    a[2] = N_ACTION_BINS - 1        # FCR
    a[3] = N_ACTION_BINS - 1        # aFRR_up
    a[5] = N_ACTION_BINS - 1        # mFRR_up
    return a


def test_exploit_demo():
    """With FIX A off, stacking all three reserves over-commits the SoC.

    The size of the hole depends on the sustain windows: each service checks
    the FULL headroom on its own, so the over-commitment scales with
    (tau_FCR + tau_aFRR + tau_mFRR). Under the legacy 4 h FCR window that was
    a ~3x hole; at the SO GL 15-minute window FCR contributes far less
    committed energy and the hole is smaller, but it is still a hole. The
    assertion therefore tests the PROPERTY (over-commitment exists) rather
    than a magnitude calibrated to one scenario, and the legacy magnitude is
    checked separately below.
    """
    soc, e_mwh, eff = 0.5, 5.0, 0.95
    env = make_env(soc, enforce=False)
    out = env._decode_clip_actions({"bess_0": saturating_action()})["bess_0"]
    up, _ = committed_energy(out)
    budget_up = (soc - 0.1) * e_mwh
    assert up > budget_up, (
        f"expected joint over-commitment with fix OFF, got {up:.3f} "
        f"vs budget {budget_up:.3f}")
    print(f"  [1] exploit demo (fix OFF, {SUSTAIN.label}): committed up = "
          f"{up:.2f} MWh vs sustainable {budget_up:.2f} MWh -> "
          f"{up/budget_up:.2f}x hole CONFIRMED")

    # Legacy 4 h scenario: the original ~3x hole must still reproduce, so
    # this test keeps its diagnostic power for the pre-revision runs.
    env_legacy = make_env(soc, enforce=False)
    env_legacy.sustain = SustainDurations.legacy_paper()
    out_l = env_legacy._decode_clip_actions(
        {"bess_0": saturating_action()})["bess_0"]
    tau = env_legacy.sustain
    up_l = (out_l["R_fcr"] * tau.fcr + out_l["R_afrr_up"] * tau.afrr
            + out_l["R_mfrr_up"] * tau.mfrr)
    assert up_l > 1.5 * budget_up, (
        f"legacy 4h scenario should show the large hole, got {up_l:.3f} "
        f"vs budget {budget_up:.3f}")
    print(f"      legacy 4h check: {up_l/budget_up:.2f}x hole CONFIRMED")


def test_joint_enforced():
    for soc in (0.15, 0.30, 0.50, 0.70, 0.88):
        for p_mw, e_mwh in ((2.5, 5.0), (1.0, 2.0), (0.5, 1.0)):
            eff = 0.95
            env = make_env(soc, p_mw, e_mwh, eff, enforce=True)
            rng = np.random.default_rng(42)
            for _ in range(200):
                a = rng.integers(0, N_ACTION_BINS, size=7, dtype=np.int64)
                out = env._decode_clip_actions({"bess_0": a})["bess_0"]
                up, dn = committed_energy(out)
                b_up = (soc - 0.1) * e_mwh
                b_dn = (0.9 - soc) * e_mwh / eff
                assert up <= b_up + TOL, (soc, p_mw, up, b_up)
                assert dn <= b_dn + TOL, (soc, p_mw, dn, b_dn)
    print("  [2] joint budget (fix ON): 3000 random bids across 5 SoC x 3 "
          "sizes, ZERO joint violations")


def test_milp_noop():
    """A bid inside the MILP's own limits must pass untouched.

    MILP conventions (tighter than or equal to the env budgets):
      up: sum r*T <= (soc - soc_min) * E * eta_d   (<= env budget, no eff)
      dn: sum r*T <= (soc_max - soc) * E / eta_c   (== env budget)
    """
    soc, p_mw, e_mwh, eff = 0.5, 2.5, 5.0, 0.95
    env = make_env(soc, p_mw, e_mwh, eff, enforce=True)
    # Construct a MILP-feasible-style bid on the grid: FCR 0.1, aFRR_up 0.2,
    # mFRR_up 0.2 of P_max -> committed up energy:
    #   (0.25*4 + 0.5*1 + 0.5*2) = 2.5 ... too big; scale down: use fractions
    #   FCR 0.0, aFRR_up 0.4 (1.0 MW * 1h = 1.0 MWh), mFRR_up 0.2 (0.5*2=1.0)
    #   total 2.0 MWh > MILP-tight 1.9 (=2.0*0.95). Use aFRR_up 0.3, mFRR_up 0.2
    #   -> 0.75 + 1.0 = 1.75 <= 1.9  OK
    a = np.zeros(7, dtype=np.int64)
    a[3] = 3   # aFRR_up = 0.3 * 2.5 = 0.75 MW -> 0.75 MWh
    a[5] = 2   # mFRR_up = 0.2 * 2.5 = 0.50 MW -> 1.00 MWh
    out = env._decode_clip_actions({"bess_0": a})["bess_0"]
    assert abs(out["R_afrr_up"] - 0.75) < 1e-9, out["R_afrr_up"]
    assert abs(out["R_mfrr_up"] - 0.50) < 1e-9, out["R_mfrr_up"]
    assert out["R_fcr"] == 0.0
    print("  [3] MILP-parity no-op: MILP-feasible bid passes decode "
          "UNCHANGED (0.75/0.50 MW preserved)")


if __name__ == "__main__":
    print("[test_fix_a] running on marl_env with FIX A")
    test_exploit_demo()
    test_joint_enforced()
    test_milp_noop()
    print("[test_fix_a] ALL PASSED")
